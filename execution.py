"""
This module provides functionality for executing and managing prompts in ComfyUI.

Key Components:
- PromptExecutor: Handles the execution of individual prompts
- PromptQueue: Manages a queue of prompts to be executed
- Validation: Functions for validating prompts and their inputs

The execution flow works as follows:
1. Prompts are validated using validate_prompt() 
2. Valid prompts are added to the PromptQueue
3. PromptExecutor executes prompts from the queue
4. Results are cached and status is tracked

Main Classes:
-------------

ExecutionResult (Enum):
    Represents the result status of executing a node:
    - SUCCESS: Node executed successfully
    - FAILURE: Node execution failed
    - PENDING: Node execution is pending/waiting

DuplicateNodeError (Exception):
    Raised when attempting to add a duplicate node ID to a graph

IsChangedCache:
    Caches whether nodes have changed between executions
    Used to optimize re-execution of unchanged nodes

CacheSet:
    Manages different types of caches used during execution:
    - outputs: Caches node output values
    - ui: Caches UI-related outputs
    - objects: Caches instantiated node objects

PromptExecutor:
    Handles execution of individual prompts
    - Manages caches
    - Tracks execution status
    - Handles errors
    - Reports progress

PromptQueue:
    Thread-safe queue for managing multiple prompts
    - Maintains execution history
    - Handles queue operations (add/remove/clear)
    - Tracks currently running prompts
    - Manages execution flags

Key Functions:
-------------

validate_prompt(prompt):
    Validates an entire prompt before execution
    - Checks for required node properties
    - Validates node connections
    - Returns validation status and any errors

validate_inputs(prompt, item, validated): 
    Validates inputs for a single node
    - Type checking
    - Value range validation
    - Custom validation rules

execute(server, dynprompt, caches, current_item, extra_data, executed, prompt_id, execution_list, pending_subgraph_results):
    Executes a single node
    - Handles caching
    - Manages execution state
    - Reports progress
    - Handles errors

get_input_data(inputs, class_def, unique_id, outputs, dynprompt, extra_data):
    Gets input data for a node
    - Resolves input connections
    - Handles lazy evaluation
    - Validates input types

Important Implementation Details:
-------------------------------

- Thread Safety: PromptQueue uses locks to ensure thread-safe operation
- Caching: Multiple cache levels optimize performance
- Error Handling: Detailed error reporting with stack traces
- Progress Tracking: Real-time execution status updates
- History: Maintains execution history with configurable size
- Validation: Extensive input validation before execution

The system is designed to be:
- Robust: Extensive error handling and validation
- Efficient: Multi-level caching and optimization
- Flexible: Supports dynamic prompt modification
- Maintainable: Clear separation of concerns
"""

import sys
import copy
import logging
import threading
import heapq
import time
import traceback
from enum import Enum
import inspect
from typing import List, Literal, NamedTuple, Optional

import torch
import nodes

import comfy.model_management
from comfy_execution.graph import get_input_info, ExecutionList, DynamicPrompt, ExecutionBlocker
from comfy_execution.graph_utils import is_link, GraphBuilder
from comfy_execution.caching import HierarchicalCache, LRUCache, CacheKeySetInputSignature, CacheKeySetID
from comfy_execution.validation import validate_node_input

class ExecutionResult(Enum):
    SUCCESS = 0
    FAILURE = 1
    PENDING = 2

class DuplicateNodeError(Exception):
    pass

class IsChangedCache:
    def __init__(self, dynprompt, outputs_cache):
        self.dynprompt = dynprompt
        self.outputs_cache = outputs_cache
        self.is_changed = {}

    def get(self, node_id):
        return False # GUO1
        if node_id in self.is_changed:
            return self.is_changed[node_id]

        node = self.dynprompt.get_node(node_id)
        class_type = node["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        if not hasattr(class_def, "IS_CHANGED"):
            self.is_changed[node_id] = False
            return self.is_changed[node_id]

        if "is_changed" in node:
            self.is_changed[node_id] = node["is_changed"]
            return self.is_changed[node_id]

        # Intentionally do not use cached outputs here. We only want constants in IS_CHANGED
        input_data_all, _ = get_input_data(node["inputs"], class_def, node_id, None)
        try:
            is_changed = _map_node_over_list(class_def, input_data_all, "IS_CHANGED")
            node["is_changed"] = [None if isinstance(x, ExecutionBlocker) else x for x in is_changed]
        except Exception as e:
            logging.warning("WARNING: {}".format(e))
            node["is_changed"] = float("NaN")
        finally:
            self.is_changed[node_id] = node["is_changed"]
        return self.is_changed[node_id]

class CacheSet:
    def __init__(self, lru_size=None):
        if (lru_size is None or lru_size == 0):
            print("using classic cache")
            self.init_classic_cache()
        else:
            print("using lru cache")
            self.init_lru_cache(lru_size)
        self.all = [self.outputs, self.ui, self.objects]

    # Useful for those with ample RAM/VRAM -- allows experimenting without
    # blowing away the cache every time
    def init_lru_cache(self, cache_size):
        self.outputs = LRUCache(CacheKeySetInputSignature, max_size=cache_size)
        self.ui = LRUCache(CacheKeySetInputSignature, max_size=cache_size)
        self.objects = HierarchicalCache(CacheKeySetID)

    # Performs like the old cache -- dump data ASAP
    def init_classic_cache(self):
        self.outputs = HierarchicalCache(CacheKeySetInputSignature)
        self.ui = HierarchicalCache(CacheKeySetInputSignature)
        self.objects = HierarchicalCache(CacheKeySetID)

    def recursive_debug_dump(self):
        result = {
            "outputs": self.outputs.recursive_debug_dump(),
            "ui": self.ui.recursive_debug_dump(),
        }
        return result

def deserialize_output(data):
    """Deserialize node outputs received from remote workers"""
    import torch
    import numpy as np
    from PIL import Image
    import base64
    import io

    type_name = data.get("type")
    value = data.get("value")

    if type_name in ("int", "float", "str", "bool"):
        return value

    elif type_name == "PIL.Image":
        img_data = base64.b64decode(value)
        img_buffer = io.BytesIO(img_data)
        return Image.open(img_buffer).convert(data["mode"])

    elif type_name == "torch.Tensor":
        # Decode base64 numpy array
        tensor_data = base64.b64decode(value)
        buffer = io.BytesIO(tensor_data)
        np_array = np.load(buffer)
        # Convert back to tensor
        return torch.from_numpy(np_array)

    elif type_name == "numpy.ndarray":
        array_data = base64.b64decode(value)
        buffer = io.BytesIO(array_data)
        return np.load(buffer)

    elif type_name == "dict":
        return {k: deserialize_output(v) for k, v in value.items()}

    elif type_name in ("list", "tuple"):
        deserialized = [deserialize_output(x) for x in value]
        return tuple(deserialized) if type_name == "tuple" else deserialized

    else:
        return value

async def get_remote_node_output(server, node_id, worker_port):
    """Fetch node output from a remote worker"""
    # Find worker with matching port
    worker = None
    for worker_id, w in server.distributed_manager.worker_instances.items():
        if w.port == worker_port:
            worker = w
            break
            
    if not worker:
        raise Exception(f"No worker found for port {worker_port}")
        
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{worker.base_url}/distributed/node_output/{node_id}"
            async with session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    if result["status"] == "success":
                        # Deserialize the output data
                        output_data = [deserialize_output(x) for x in result["output_data"]]
                        return output_data
                    else:
                        raise Exception(f"Error getting remote output: {result.get('error', 'Unknown error')}")
                else:
                    raise Exception(f"Failed to get remote output, status: {response.status}")
    except Exception as e:
        logging.error(f"Error fetching remote node output: {e}")
        raise

def get_input_data(inputs, class_def, unique_id, outputs=None, dynprompt=None, extra_data={}):
    print("<"*50)
    print(f"\nGetting input data for node {unique_id} of type {class_def.__name__}")
    print("outputs:.:", outputs.all_node_ids())
    if outputs:
        print("while having outputs::", outputs.all_node_ids())
        for node_id_debug in outputs.all_node_ids():
            print(f"{node_id_debug}={outputs.get(node_id_debug)}")        
    
    valid_inputs = class_def.INPUT_TYPES()
    print(f"Valid input types: {valid_inputs}")
    
    input_data_all = {}
    missing_keys = {}
    
    print("\nProcessing regular inputs:")
    for x in inputs:
        print(f"\nProcessing input '{x}':")
        input_data = inputs[x]
        print(f"Raw input data: {input_data}")
        
        input_type, input_category, input_info = get_input_info(class_def, x, valid_inputs)
        print(f"Input type: {input_type}, category: {input_category}, info: {input_info}")
        
        def mark_missing():
            print(f"Marking input '{x}' as missing")
            missing_keys[x] = True
            input_data_all[x] = (None,)
            
        if is_link(input_data) and (not input_info or not input_info.get("rawLink", False)):
            print(f"Input '{x}' is a link")
            input_unique_id = input_data[0]
            output_index = input_data[1]
            print(f"Link points to node {input_unique_id} output {output_index}")
            
            if outputs is None:
                print("No outputs cache provided, marking as missing")
                mark_missing()
                continue # This might be a lazily-evaluated input
                





                
            # # Check if input node is on a remote worker
            # input_node = dynprompt.get_node(input_unique_id)
            # input_class = nodes.NODE_CLASS_MAPPINGS[input_node["class_type"]]
            
            # if hasattr(input_class, 'EXECUTION_TARGET') and input_class.EXECUTION_TARGET != 'local':
            #     try:
            #         # Fetch from remote worker
            #         remote_output = await get_remote_node_output(
            #             dynprompt.server,  # Need to pass server instance to dynprompt
            #             input_unique_id, 
            #             int(input_class.EXECUTION_TARGET)
            #         )
            #         if output_index >= len(remote_output):
            #             print(f"Remote output index {output_index} out of bounds")
            #             mark_missing()
            #             continue
            #         obj = remote_output[output_index]
            #         input_data_all[x] = obj
            #         continue
            #     except Exception as e:
            #         logging.error(f"Failed to get remote input: {e}")
            #         mark_missing()
            #         continue






                
            cached_output = outputs.get(input_unique_id)
            if cached_output is None:
                print(f"No cached output found for node {input_unique_id}")
                mark_missing()
                continue
                
            if output_index >= len(cached_output):
                print(f"Output index {output_index} out of bounds")
                mark_missing()
                continue
                
            obj = cached_output[output_index]
            print(f"Retrieved cached output: {type(obj)}")
            input_data_all[x] = obj
        elif input_category is not None:
            print(f"Input '{x}' is a direct value")
            input_data_all[x] = [input_data]

    print("\nProcessing hidden inputs:")
    if "hidden" in valid_inputs:
        h = valid_inputs["hidden"]
        for x in h:
            print(f"\nProcessing hidden input '{x}': {h[x]}")
            if h[x] == "PROMPT":
                input_data_all[x] = [dynprompt.get_original_prompt() if dynprompt is not None else {}]
                print("Added original prompt")
            if h[x] == "DYNPROMPT":
                input_data_all[x] = [dynprompt]
                print("Added dynamic prompt object")
            if h[x] == "EXTRA_PNGINFO":
                input_data_all[x] = [extra_data.get('extra_pnginfo', None)]
                print("Added extra PNG info")
            if h[x] == "UNIQUE_ID":
                input_data_all[x] = [unique_id]
                print("Added unique ID")
                
    # print(f"\nFinal input data: {input_data_all}")
    print(f"Missing keys: {missing_keys}")
    print(">"*50)
    return input_data_all, missing_keys

map_node_over_list = None #Don't hook this please

def _map_node_over_list(obj, input_data_all, func, allow_interrupt=False, execution_block_cb=None, pre_execute_cb=None):
    # check if node wants the lists
    input_is_list = getattr(obj, "INPUT_IS_LIST", False)

    if len(input_data_all) == 0:
        max_len_input = 0
    else:
        max_len_input = max(len(x) for x in input_data_all.values())

    # get a slice of inputs, repeat last input when list isn't long enough
    def slice_dict(d, i):
        return {k: v[i if len(v) > i else -1] for k, v in d.items()}

    results = []
    def process_inputs(inputs, index=None, input_is_list=False):
        if allow_interrupt:
            nodes.before_node_execution()
        execution_block = None
        for k, v in inputs.items():
            if input_is_list:
                for e in v:
                    if isinstance(e, ExecutionBlocker):
                        v = e
                        break
            if isinstance(v, ExecutionBlocker):
                execution_block = execution_block_cb(v) if execution_block_cb else v
                break
        if execution_block is None:
            if pre_execute_cb is not None and index is not None:
                pre_execute_cb(index)
            results.append(getattr(obj, func)(**inputs))
        else:
            results.append(execution_block)

    if input_is_list:
        process_inputs(input_data_all, 0, input_is_list=input_is_list)
    elif max_len_input == 0:
        process_inputs({})
    else:
        for i in range(max_len_input):
            input_dict = slice_dict(input_data_all, i)
            process_inputs(input_dict, i)
    return results

def merge_result_data(results, obj):
    # check which outputs need concatenating
    output = []
    output_is_list = [False] * len(results[0])
    if hasattr(obj, "OUTPUT_IS_LIST"):
        output_is_list = obj.OUTPUT_IS_LIST

    # merge node execution results
    for i, is_list in zip(range(len(results[0])), output_is_list):
        if is_list:
            value = []
            for o in results:
                if isinstance(o[i], ExecutionBlocker):
                    value.append(o[i])
                else:
                    value.extend(o[i])
            output.append(value)
        else:
            output.append([o[i] for o in results])
    return output

def get_output_data(obj, input_data_all, execution_block_cb=None, pre_execute_cb=None):
    results = []
    uis = []
    subgraph_results = []
    return_values = _map_node_over_list(obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)
    has_subgraph = False
    for i in range(len(return_values)):
        r = return_values[i]
        if isinstance(r, dict):
            if 'ui' in r:
                uis.append(r['ui'])
            if 'expand' in r:
                # Perform an expansion, but do not append results
                has_subgraph = True
                new_graph = r['expand']
                result = r.get("result", None)
                if isinstance(result, ExecutionBlocker):
                    result = tuple([result] * len(obj.RETURN_TYPES))
                subgraph_results.append((new_graph, result))
            elif 'result' in r:
                result = r.get("result", None)
                if isinstance(result, ExecutionBlocker):
                    result = tuple([result] * len(obj.RETURN_TYPES))
                results.append(result)
                subgraph_results.append((None, result))
        else:
            if isinstance(r, ExecutionBlocker):
                r = tuple([r] * len(obj.RETURN_TYPES))
            results.append(r)
            subgraph_results.append((None, r))

    if has_subgraph:
        output = subgraph_results
    elif len(results) > 0:
        output = merge_result_data(results, obj)
    else:
        output = []
    ui = dict()
    if len(uis) > 0:
        ui = {k: [y for x in uis for y in x[k]] for k in uis[0].keys()}
    return output, ui, has_subgraph

def format_value(x):
    if x is None:
        return None
    elif isinstance(x, (int, float, bool, str)):
        return x
    else:
        return str(x)

def execute(server, dynprompt, caches, current_item, extra_data, executed, prompt_id, execution_list, pending_subgraph_results):
    unique_id = current_item
    real_node_id = dynprompt.get_real_node_id(unique_id)
    display_node_id = dynprompt.get_display_node_id(unique_id)
    parent_node_id = dynprompt.get_parent_node_id(unique_id)
    inputs = dynprompt.get_node(unique_id)['inputs']
    class_type = dynprompt.get_node(unique_id)['class_type']
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]

    # # Check if this node should be executed on a remote worker
    # if hasattr(class_def, 'EXECUTION_TARGET') and class_def.EXECUTION_TARGET != 'local':
    #     print("execution remotely on ::", class_def.EXECUTION_TARGET)
    #     try:
    #         # Get worker for target
    #         target = class_def.EXECUTION_TARGET
    #         worker = None
    #         for worker_id, w in server.distributed_manager.worker_instances.items():
    #             if w.port == int(target):
    #                 worker = w
    #                 break
                    
    #         if not worker:
    #             raise Exception(f"No worker found for target {target}")
                
    #         # Execute node on remote worker
    #         node_data = {
    #             "id": unique_id,
    #             "class_type": class_type,
    #             "inputs": inputs
    #         }
            
    #         result = worker.execute_node(node_data)
    #         if result.get("status") == "success":
    #             output_data = result["output"]
    #             caches.outputs.set(unique_id, output_data)
    #             executed.add(unique_id)
    #             return (ExecutionResult.SUCCESS, None, None)
    #         else:
    #             error = {
    #                 "node_id": real_node_id,
    #                 "exception_message": result.get("error", "Unknown error"),
    #                 "exception_type": "RemoteExecutionError",
    #                 "traceback": [],
    #                 "current_inputs": {}
    #             }
    #             return (ExecutionResult.FAILURE, error, Exception(result.get("error")))
                
    #     except Exception as ex:
    #         error = {
    #             "node_id": real_node_id,
    #             "exception_message": str(ex),
    #             "exception_type": "RemoteExecutionError", 
    #             "traceback": traceback.format_exc(),
    #             "current_inputs": {}
    #         }
    #         return (ExecutionResult.FAILURE, error, ex)

    # Continue with local execution
    print(f"\n=== Starting local execution for node {unique_id} ===")
    
    if caches.outputs.get(unique_id) is not None:
        print(f"Node {unique_id} found in cache")
        if server.client_id is not None:
            print(f"Sending cached output to client {server.client_id}")
            cached_output = caches.ui.get(unique_id) or {}
            server.send_sync("executed", { "node": unique_id, "display_node": display_node_id, "output": cached_output.get("output",None), "prompt_id": prompt_id }, server.client_id)
        return (ExecutionResult.SUCCESS, None, None)

    print(f"Node {unique_id} not in cache, executing...")
    input_data_all = None
    try:
        if unique_id in pending_subgraph_results:
            print(f"Found pending subgraph results for node {unique_id}")
            cached_results = pending_subgraph_results[unique_id]
            resolved_outputs = []
            for is_subgraph, result in cached_results:
                print(f"Processing result - is_subgraph: {is_subgraph}")
                if not is_subgraph:
                    resolved_outputs.append(result)
                else:
                    print("Resolving subgraph outputs...")
                    resolved_output = []
                    for r in result:
                        if is_link(r):
                            source_node, source_output = r[0], r[1]
                            print(f"Resolving link from node {source_node} output {source_output}")
                            node_output = caches.outputs.get(source_node)[source_output]
                            for o in node_output:
                                resolved_output.append(o)

                        else:
                            resolved_output.append(r)
                    resolved_outputs.append(tuple(resolved_output))
            print("Merging resolved outputs...")
            output_data = merge_result_data(resolved_outputs, class_def)
            output_ui = []
            has_subgraph = False
        else:
            print("Getting input data...")
            input_data_all, missing_keys = get_input_data(inputs, class_def, unique_id, caches.outputs, dynprompt, extra_data)
            print(f"Input data: {input_data_all}")
            print(f"Missing keys: {missing_keys}")
            
            if server.client_id is not None:
                print(f"Notifying client {server.client_id} of execution")
                server.last_node_id = display_node_id
                server.send_sync("executing", { "node": unique_id, "display_node": display_node_id, "prompt_id": prompt_id }, server.client_id)

            print("Getting/creating node object...")
            obj = caches.objects.get(unique_id)
            if obj is None:
                print("Creating new node object")
                obj = class_def()
                caches.objects.set(unique_id, obj)

            if hasattr(obj, "check_lazy_status"):
                print("Checking lazy status...")
                required_inputs = _map_node_over_list(obj, input_data_all, "check_lazy_status", allow_interrupt=True)
                required_inputs = set(sum([r for r in required_inputs if isinstance(r,list)], []))
                required_inputs = [x for x in required_inputs if isinstance(x,str) and (
                    x not in input_data_all or x in missing_keys
                )]
                if len(required_inputs) > 0:
                    print(f"Node has pending required inputs: {required_inputs}")
                    for i in required_inputs:
                        execution_list.make_input_strong_link(unique_id, i)
                    return (ExecutionResult.PENDING, None, None)

            def execution_block_cb(block):
                if block.message is not None:
                    print(f"Execution blocked: {block.message}")
                    mes = {
                        "prompt_id": prompt_id,
                        "node_id": unique_id,
                        "node_type": class_type,
                        "executed": list(executed),

                        "exception_message": f"Execution Blocked: {block.message}",
                        "exception_type": "ExecutionBlocked",
                        "traceback": [],
                        "current_inputs": [],
                        "current_outputs": [],
                    }
                    server.send_sync("execution_error", mes, server.client_id)
                    return ExecutionBlocker(None)
                else:
                    return block

            def pre_execute_cb(call_index):
                print(f"Pre-execute callback for index {call_index}")
                GraphBuilder.set_default_prefix(unique_id, call_index, 0)

            print("Getting output data...")
            output_data, output_ui, has_subgraph = get_output_data(obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)
            print(f"Got output data, has_subgraph: {has_subgraph}")

        if len(output_ui) > 0:
            print("Setting UI cache...")
            caches.ui.set(unique_id, {
                "meta": {
                    "node_id": unique_id,
                    "display_node": display_node_id,
                    "parent_node": parent_node_id,
                    "real_node_id": real_node_id,
                },
                "output": output_ui
            })
            if server.client_id is not None:
                print(f"Sending UI output to client {server.client_id}")
                server.send_sync("executed", { "node": unique_id, "display_node": display_node_id, "output": output_ui, "prompt_id": prompt_id }, server.client_id)

        if has_subgraph:
            print("Processing subgraph...")
            cached_outputs = []
            new_node_ids = []
            new_output_ids = []
            new_output_links = []
            for i in range(len(output_data)):
                new_graph, node_outputs = output_data[i]
                if new_graph is None:
                    cached_outputs.append((False, node_outputs))
                else:
                    print(f"Processing new graph {i}...")
                    # Check for conflicts
                    for node_id in new_graph.keys():
                        if dynprompt.has_node(node_id):
                            raise DuplicateNodeError(f"Attempt to add duplicate node {node_id}. Ensure node ids are unique and deterministic or use graph_utils.GraphBuilder.")
                    for node_id, node_info in new_graph.items():
                        print(f"Adding node {node_id} to graph")
                        new_node_ids.append(node_id)
                        display_id = node_info.get("override_display_id", unique_id)
                        dynprompt.add_ephemeral_node(node_id, node_info, unique_id, display_id)
                        # Figure out if the newly created node is an output node
                        class_type = node_info["class_type"]
                        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
                        if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
                            print(f"Node {node_id} is an output node")
                            new_output_ids.append(node_id)
                    for i in range(len(node_outputs)):
                        if is_link(node_outputs[i]):
                            from_node_id, from_socket = node_outputs[i][0], node_outputs[i][1]
                            print(f"Adding output link from {from_node_id}:{from_socket}")
                            new_output_links.append((from_node_id, from_socket))
                    cached_outputs.append((True, node_outputs))
            new_node_ids = set(new_node_ids)
            print("Updating caches for new nodes...")
            for cache in caches.all:
                cache.ensure_subcache_for(unique_id, new_node_ids).clean_unused()
            for node_id in new_output_ids:
                print(f"Adding output node {node_id} to execution list")
                execution_list.add_node(node_id)
            for link in new_output_links:
                print(f"Adding strong link {link[0]}:{link[1]} -> {unique_id}")
                execution_list.add_strong_link(link[0], link[1], unique_id)
            pending_subgraph_results[unique_id] = cached_outputs
            return (ExecutionResult.PENDING, None, None)

        print("Setting output cache...")
        caches.outputs.set(unique_id, output_data)

    except comfy.model_management.InterruptProcessingException as iex:
        logging.info("Processing interrupted")

        # skip formatting inputs/outputs
        error_details = {
            "node_id": real_node_id,
        }

        return (ExecutionResult.FAILURE, error_details, iex)

    except Exception as ex:
        print(f"Exception occurred: {ex}")
        typ, _, tb = sys.exc_info()
        exception_type = full_type_name(typ)
        input_data_formatted = {}
        if input_data_all is not None:
            input_data_formatted = {}
            for name, inputs in input_data_all.items():
                input_data_formatted[name] = [format_value(x) for x in inputs]

        logging.error(f"!!! Exception during processing !!! {ex}")
        logging.error(traceback.format_exc())

        error_details = {
            "node_id": real_node_id,
            "exception_message": str(ex),
            "exception_type": exception_type,
            "traceback": traceback.format_tb(tb),
            "current_inputs": input_data_formatted
        }
        if isinstance(ex, comfy.model_management.OOM_EXCEPTION):
            logging.error("Got an OOM, unloading all loaded models.")
            comfy.model_management.unload_all_models()

        return (ExecutionResult.FAILURE, error_details, ex)

    print(f"Adding node {unique_id} to executed set")
    executed.add(unique_id)

    print(f"=== Completed execution for node {unique_id} ===\n")
    return (ExecutionResult.SUCCESS, None, None)

class PromptExecutor:
    def __init__(self, server, lru_size=None):
        self.lru_size = lru_size
        self.server = server
        self.reset()

    def reset(self):
        logging.info("Resetting PromptExecutor state")
        self.caches = CacheSet(self.lru_size)
        self.status_messages = []
        self.success = True

    def add_message(self, event, data: dict, broadcast: bool):
        data = {
            **data,
            "timestamp": int(time.time() * 1000),
        }
        logging.debug(f"Adding message - Event: {event}, Data: {data}, Broadcast: {broadcast}")
        self.status_messages.append((event, data))
        if self.server.client_id is not None or broadcast:
            self.server.send_sync(event, data, self.server.client_id)

    def handle_execution_error(self, prompt_id, prompt, current_outputs, executed, error, ex):
        node_id = error["node_id"]
        class_type = prompt[node_id]["class_type"]
        logging.error(f"Handling execution error for node {node_id} ({class_type})")

        # First, send back the status to the frontend depending
        # on the exception type
        if isinstance(ex, comfy.model_management.InterruptProcessingException):
            logging.info("Processing was interrupted by user")
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),
            }
            self.add_message("execution_interrupted", mes, broadcast=True)
        else:
            logging.error(f"Execution error: {error['exception_message']}")
            logging.error(f"Traceback: {''.join(error['traceback'])}")
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),
                "exception_message": error["exception_message"],
                "exception_type": error["exception_type"],
                "traceback": error["traceback"],
                "current_inputs": error["current_inputs"],
                "current_outputs": list(current_outputs),
            }
            self.add_message("execution_error", mes, broadcast=False)

    # takes entire prompt/workflow
    def execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):




        logging.info(f"Starting execution of prompt {prompt_id}")
        nodes.interrupt_processing(False)


        # if self.caches and self.caches.outputs and self.caches.outputs.dynprompt:
        #     print("very initial:")
        #     cached_nodes = []
        #     for node_id in prompt:
        #         if self.caches.outputs.get(node_id) is not None:
        #             cached_nodes.append(node_id)
        #     print("all cached nodes(initial)::", cached_nodes)


        print("execute workflow in execution.py::")
        print("prompt::", prompt)
        print("")
        print("prompt_id::", prompt_id)
        print("")
        print("extra_data::", extra_data)
        print("")
        print("execute_outputs::", execute_outputs)
        print("")


        if "client_id" in extra_data:
            self.server.client_id = extra_data["client_id"]
            print(f"Using client_id: {extra_data['client_id']}")
        else:
            self.server.client_id = None

        self.status_messages = []
        self.add_message("execution_start", { "prompt_id": prompt_id}, broadcast=False)

        with torch.inference_mode():
            # # Check if this is a remote execution
            # is_remote_execution = len(prompt) == 1 and any(node.get("remote_execution", False) for node in prompt.values())
            # if is_remote_execution:
            #     # For remote execution, create placeholder nodes for any missing inputs
            #     node = list(prompt.values())[0]  # Get the single node
            #     for input_name, input_value in node["inputs"].items():
            #         if isinstance(input_value, list) and len(input_value) == 2:
            #             input_node_id = input_value[0]
            #             if input_node_id not in prompt:
            #                 # Create a placeholder node
            #                 prompt[input_node_id] = {
            #                     "class_type": "PlaceholderNode",
            #                     "inputs": {},
            #                     "is_placeholder": True
            #                 }

            logging.debug("Initializing dynamic prompt and caches")
            dynamic_prompt = DynamicPrompt(prompt)
            


            
            is_changed_cache = IsChangedCache(dynamic_prompt, self.caches.outputs)
            
            # if self.caches:
            #     print("outputs cache before block::", self.caches.outputs.all_node_ids())
            # else:
            #     print("outputs cache before block:: not yet initialized")
            
            # # Disable this for now (GUO1)
            # print("dynamic prompt:::", dynamic_prompt)
            for cache in self.caches.all:
                # cache.set_prompt(dynamic_prompt, prompt.keys(), is_changed_cache)
                cache.set_prompt(dynamic_prompt, ["21", "19"], is_changed_cache)
                # cache.clean_unused()

            print("outputs cache after block::", self.caches.outputs.all_node_ids())
        
            print("cache dump::")
            print(self.caches.recursive_debug_dump())
            print("done with cache dump")

            if execute_outputs[0] == "19":
                print("21--", self.caches.outputs.get("21"))

            cached_nodes = []
            for node_id in prompt:
                if self.caches.outputs.get(node_id) is not None:
                    cached_nodes.append(node_id)

            print("all cached nodes::", cached_nodes)

            logging.info(f"Found {len(cached_nodes)} cached nodes")
            comfy.model_management.cleanup_models_gc()
            self.add_message("execution_cached",
                          { "nodes": cached_nodes, "prompt_id": prompt_id},
                          broadcast=False)
            pending_subgraph_results = {}
            executed = set()
            execution_list = ExecutionList(dynamic_prompt, self.caches.outputs)

            print(f"execute outputs::", execute_outputs)
            # logging.info(f"ExecutionList::", execution_list)

            current_outputs = self.caches.outputs.all_node_ids()

            for one_cached_output in current_outputs:
                print(f"cached for {one_cached_output} = {self.caches.outputs.get(one_cached_output)}")

            # for node_id in list(execute_outputs):
            #     print("node from execution list (initial):::", node_id)
            #     execution_list.add_node(node_id)
            #     logging.debug(f"Added node {node_id} to execution list")


            print("execution list (preflight)::", execution_list.__dict__)

            # Short Circuit
            result, error, ex = execute(self.server, dynamic_prompt, self.caches, node_id, extra_data, executed, prompt_id, execution_list, pending_subgraph_results)
            self.add_message("execution_success", { "prompt_id": prompt_id }, broadcast=False)
            print("done yalla")

            # while not execution_list.is_empty():
            #     node_id, error, ex = execution_list.stage_node_execution()
            #     print("while entry::", node_id, '*', error, '*', ex)
            #     if error is not None:
            #         logging.error(f"Error staging node {node_id} for execution")
            #         self.handle_execution_error(prompt_id, dynamic_prompt.original_prompt, current_outputs, executed, error, ex)
            #         break

            #     print(f"Executing node {node_id}")
                
            #     # Check if this node should be executed remotely
            #     node = dynamic_prompt.get_node(node_id)
            #     class_type = node["class_type"]
            #     class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
                
            #     # if hasattr(class_def, 'EXECUTION_TARGET') and class_def.EXECUTION_TARGET != 'local':
            #     if int(node_id) >= 10:  # This is temporary, should use EXECUTION_TARGET
            #         target_port = 8288  # This is temporary, should use EXECUTION_TARGET
                    
            #         # Skip remote execution if we're already on the target port
            #         if self.server.port == target_port:
            #             print(f"Already on target port {target_port}, executing locally")
            #             result, error, ex = execute(self.server, dynamic_prompt, self.caches, node_id, extra_data, executed, prompt_id, execution_list, pending_subgraph_results)
            #             self.success = result != ExecutionResult.FAILURE
            #             if result == ExecutionResult.FAILURE:
            #                 self.handle_execution_error(prompt_id, dynamic_prompt.original_prompt, current_outputs, executed, error, ex)
            #                 break
            #             elif result == ExecutionResult.PENDING:
            #                 execution_list.unstage_node_execution()
            #             else:
            #                 execution_list.complete_node_execution()
            #             continue
                    
            #         try:
            #             # Find worker for target port
            #             print(f"Looking for worker with port {target_port} (current port is {self.server.port})")
            #             worker = None
            #             for worker_id, w in self.server.distributed_manager.worker_instances.items():
            #                 print(f"Checking worker {worker_id} with port {w.port}")
            #                 if w.port == target_port:
            #                     worker = w
            #                     print(f"Found matching worker: {worker_id}")
            #                     break
                                
            #             if not worker:
            #                 raise Exception(f"No worker found for target port {target_port}")
                            
            #             # Execute node on remote worker
            #             import requests
            #             url = f"{worker.base_url}/distributed/execute_node"
            #             node_data = {
            #                 "id": node_id,
            #                 "class_type": class_type,
            #                 "inputs": node["inputs"]
            #             }
                        
            #             print(f"Sending execution request to {url}")
            #             print(f"Node data: {node_data}")
                        
            #             response = requests.post(url, json={"node_data": node_data})
            #             print(f"Got response with status code: {response.status_code}")
                        
            #             if response.status_code != 200:
            #                 raise Exception(f"Remote execution failed with status {response.status_code}")
                            
            #             result = response.json()
            #             print(f"Response data: {result}")
                        
            #             if result["status"] != "queued":
            #                 raise Exception("Remote execution not queued")

            #             print("Waiting 10 seconds for remote execution to complete...")
            #             import time
            #             time.sleep(10)
                            
            #             # Check if execution is complete
            #             remote_prompt_id = result["prompt_id"]
            #             status_url = f"{worker.base_url}/distributed/node_status/{remote_prompt_id}"
            #             print(f"Checking execution status at {status_url}")
                        
            #             status_response = requests.get(status_url)
            #             print(f"Status response code: {status_response.status_code}")
                        
            #             if status_response.status_code == 200:
            #                 status_result = status_response.json()
            #                 print(f"Status result: {status_result}")
                            
            #                 if status_result["status"] == "completed":
            #                     print("Remote execution completed, fetching output...")
            #                     # Store output in cache
            #                     output_url = f"{worker.base_url}/distributed/node_output/{node_id}"
            #                     output_response = requests.get(output_url)
            #                     print(f"Output response code: {output_response.status_code}")
                                
            #                     if output_response.status_code == 200:
            #                         output_result = output_response.json()
            #                         print(f"Output result: {output_result}")
                                    
            #                         if output_result["status"] == "success":
            #                             print("Successfully got output, deserializing...")
            #                             # Deserialize and store output
            #                             output_data = [deserialize_output(x) for x in output_result["output_data"]]
            #                             print(f"Deserialized output: {output_data}")
                                        
            #                             self.caches.outputs.set(node_id, output_data)
            #                             executed.add(node_id)
            #                             execution_list.complete_node_execution()
            #                             print(f"Node {node_id} remote execution completed successfully")
            #                             continue
                                        
            #             print(f"Remote execution still pending for node {node_id}, will retry later")
            #             execution_list.unstage_node_execution()
            #             continue
                            
            #         except Exception as ex:
            #             print(f"Remote execution failed for node {node_id}: {str(ex)}")
            #             print(f"Traceback: {traceback.format_exc()}")
            #             error = {
            #                 "node_id": node_id,
            #                 "exception_message": str(ex),
            #                 "exception_type": "RemoteExecutionError", 
            #                 "traceback": traceback.format_exc(),
            #                 "current_inputs": {}
            #             }
            #             self.handle_execution_error(prompt_id, dynamic_prompt.original_prompt, current_outputs, executed, error, ex)
            #             break

            #     # Continue with local execution if not remote
            #     result, error, ex = execute(self.server, dynamic_prompt, self.caches, node_id, extra_data, executed, prompt_id, execution_list, pending_subgraph_results)
                
                
            #     self.success = result != ExecutionResult.FAILURE
            #     if result == ExecutionResult.FAILURE:
            #         logging.error(f"Node {node_id} execution failed")
            #         self.handle_execution_error(prompt_id, dynamic_prompt.original_prompt, current_outputs, executed, error, ex)
            #         break
            #     elif result == ExecutionResult.PENDING:
            #         logging.debug(f"Node {node_id} execution pending")
            #         execution_list.unstage_node_execution()
            #     else: # result == ExecutionResult.SUCCESS:
            #         logging.debug(f"Node {node_id} execution completed successfully")
            #         execution_list.complete_node_execution()
            # else:
            #     # Only execute when the while-loop ends without break
            #     logging.info(f"Prompt {prompt_id} execution completed successfully")
            #     self.add_message("execution_success", { "prompt_id": prompt_id }, broadcast=False)

            logging.debug("Collecting UI outputs and meta information")
            ui_outputs = {}
            meta_outputs = {}
            all_node_ids = self.caches.ui.all_node_ids()
            for node_id in all_node_ids:
                ui_info = self.caches.ui.get(node_id)
                if ui_info is not None:
                    ui_outputs[node_id] = ui_info["output"]
                    meta_outputs[node_id] = ui_info["meta"]
            self.history_result = {
                "outputs": ui_outputs,
                "meta": meta_outputs,
            }
            self.server.last_node_id = None
            # if comfy.model_management.DISABLE_SMART_MEMORY:
            #     logging.info("Smart memory disabled, unloading all models")
            #     comfy.model_management.unload_all_models()


def validate_inputs(prompt, item, validated):
    unique_id = item
    if unique_id in validated:
        return validated[unique_id]

    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    obj_class = nodes.NODE_CLASS_MAPPINGS[class_type]

    class_inputs = obj_class.INPUT_TYPES()
    valid_inputs = set(class_inputs.get('required',{})).union(set(class_inputs.get('optional',{})))

    errors = []
    valid = True

    validate_function_inputs = []
    validate_has_kwargs = False
    if hasattr(obj_class, "VALIDATE_INPUTS"):
        argspec = inspect.getfullargspec(obj_class.VALIDATE_INPUTS)
        validate_function_inputs = argspec.args
        validate_has_kwargs = argspec.varkw is not None
    received_types = {}

    for x in valid_inputs:
        type_input, input_category, extra_info = get_input_info(obj_class, x, class_inputs)
        assert extra_info is not None
        if x not in inputs:
            if input_category == "required":
                error = {
                    "type": "required_input_missing",
                    "message": "Required input is missing",
                    "details": f"{x}",
                    "extra_info": {
                        "input_name": x
                    }
                }
                errors.append(error)
            continue

        val = inputs[x]
        info = (type_input, extra_info)
        if isinstance(val, list):
            if len(val) != 2:
                error = {
                    "type": "bad_linked_input",
                    "message": "Bad linked input, must be a length-2 list of [node_id, slot_index]",
                    "details": f"{x}",
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_value": val
                    }
                }
                errors.append(error)
                continue

            o_id = val[0]
            o_class_type = prompt[o_id]['class_type']
            r = nodes.NODE_CLASS_MAPPINGS[o_class_type].RETURN_TYPES
            received_type = r[val[1]]
            received_types[x] = received_type
            if 'input_types' not in validate_function_inputs and not validate_node_input(received_type, type_input):
                details = f"{x}, received_type({received_type}) mismatch input_type({type_input})"
                error = {
                    "type": "return_type_mismatch",
                    "message": "Return type mismatch between linked nodes",
                    "details": details,
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_type": received_type,
                        "linked_node": val
                    }
                }
                errors.append(error)
                continue
            try:
                r = validate_inputs(prompt, o_id, validated)
                if r[0] is False:
                    # `r` will be set in `validated[o_id]` already
                    valid = False
                    continue
            except Exception as ex:
                typ, _, tb = sys.exc_info()
                valid = False
                exception_type = full_type_name(typ)
                reasons = [{
                    "type": "exception_during_inner_validation",
                    "message": "Exception when validating inner node",
                    "details": str(ex),
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "exception_message": str(ex),
                        "exception_type": exception_type,
                        "traceback": traceback.format_tb(tb),
                        "linked_node": val
                    }
                }]
                validated[o_id] = (False, reasons, o_id)
                continue
        else:
            try:
                if type_input == "INT":
                    val = int(val)
                    inputs[x] = val
                if type_input == "FLOAT":
                    val = float(val)
                    inputs[x] = val
                if type_input == "STRING":
                    val = str(val)
                    inputs[x] = val
                if type_input == "BOOLEAN":
                    val = bool(val)
                    inputs[x] = val
            except Exception as ex:
                error = {
                    "type": "invalid_input_type",
                    "message": f"Failed to convert an input value to a {type_input} value",
                    "details": f"{x}, {val}, {ex}",
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_value": val,
                        "exception_message": str(ex)
                    }
                }
                errors.append(error)
                continue

            if x not in validate_function_inputs and not validate_has_kwargs:
                if "min" in extra_info and val < extra_info["min"]:
                    error = {
                        "type": "value_smaller_than_min",
                        "message": "Value {} smaller than min of {}".format(val, extra_info["min"]),
                        "details": f"{x}",
                        "extra_info": {
                            "input_name": x,
                            "input_config": info,
                            "received_value": val,
                        }
                    }
                    errors.append(error)
                    continue
                if "max" in extra_info and val > extra_info["max"]:
                    error = {
                        "type": "value_bigger_than_max",
                        "message": "Value {} bigger than max of {}".format(val, extra_info["max"]),
                        "details": f"{x}",
                        "extra_info": {
                            "input_name": x,
                            "input_config": info,
                            "received_value": val,
                        }
                    }
                    errors.append(error)
                    continue

                if isinstance(type_input, list):
                    if val not in type_input:
                        input_config = info
                        list_info = ""

                        # Don't send back gigantic lists like if they're lots of
                        # scanned model filepaths
                        if len(type_input) > 20:
                            list_info = f"(list of length {len(type_input)})"
                            input_config = None
                        else:
                            list_info = str(type_input)

                        error = {
                            "type": "value_not_in_list",
                            "message": "Value not in list",
                            "details": f"{x}: '{val}' not in {list_info}",
                            "extra_info": {
                                "input_name": x,
                                "input_config": input_config,
                                "received_value": val,
                            }
                        }
                        errors.append(error)
                        continue

    if len(validate_function_inputs) > 0 or validate_has_kwargs:
        input_data_all, _ = get_input_data(inputs, obj_class, unique_id)
        input_filtered = {}
        for x in input_data_all:
            if x in validate_function_inputs or validate_has_kwargs:
                input_filtered[x] = input_data_all[x]
        if 'input_types' in validate_function_inputs:
            input_filtered['input_types'] = [received_types]

        #ret = obj_class.VALIDATE_INPUTS(**input_filtered)
        ret = _map_node_over_list(obj_class, input_filtered, "VALIDATE_INPUTS")
        for x in input_filtered:
            for i, r in enumerate(ret):
                if r is not True and not isinstance(r, ExecutionBlocker):
                    details = f"{x}"
                    if r is not False:
                        details += f" - {str(r)}"

                    error = {
                        "type": "custom_validation_failed",
                        "message": "Custom validation failed for node",
                        "details": details,
                        "extra_info": {
                            "input_name": x,
                        }
                    }
                    errors.append(error)
                    continue

    if len(errors) > 0 or valid is not True:
        ret = (False, errors, unique_id)
    else:
        ret = (True, [], unique_id)

    validated[unique_id] = ret
    return ret

def full_type_name(klass):
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__
    return module + '.' + klass.__qualname__

def validate_prompt(prompt):
    outputs = set()
    for x in prompt:
        if 'class_type' not in prompt[x]:
            error = {
                "type": "invalid_prompt",
                "message": "Cannot execute because a node is missing the class_type property.",
                "details": f"Node ID '#{x}'",
                "extra_info": {}
            }
            return (False, error, [], [])

        class_type = prompt[x]['class_type']
        class_ = nodes.NODE_CLASS_MAPPINGS.get(class_type, None)
        if class_ is None:
            error = {
                "type": "invalid_prompt",
                "message": f"Cannot execute because node {class_type} does not exist.",
                "details": f"Node ID '#{x}'",
                "extra_info": {}
            }
            return (False, error, [], [])

        if hasattr(class_, 'OUTPUT_NODE') and class_.OUTPUT_NODE is True:
            outputs.add(x)

    if len(outputs) == 0:
        error = {
            "type": "prompt_no_outputs",
            "message": "Prompt has no outputs",
            "details": "",
            "extra_info": {}
        }
        return (False, error, [], [])

    good_outputs = set()
    errors = []
    node_errors = {}
    validated = {}
    for o in outputs:
        valid = False
        reasons = []
        try:
            m = validate_inputs(prompt, o, validated)
            valid = m[0]
            reasons = m[1]
        except Exception as ex:
            typ, _, tb = sys.exc_info()
            valid = False
            exception_type = full_type_name(typ)
            reasons = [{
                "type": "exception_during_validation",
                "message": "Exception when validating node",
                "details": str(ex),
                "extra_info": {
                    "exception_type": exception_type,
                    "traceback": traceback.format_tb(tb)
                }
            }]
            validated[o] = (False, reasons, o)

        if valid is True:
            good_outputs.add(o)
        else:
            logging.error(f"Failed to validate prompt for output {o}:")
            if len(reasons) > 0:
                logging.error("* (prompt):")
                for reason in reasons:
                    logging.error(f"  - {reason['message']}: {reason['details']}")
            errors += [(o, reasons)]
            for node_id, result in validated.items():
                valid = result[0]
                reasons = result[1]
                # If a node upstream has errors, the nodes downstream will also
                # be reported as invalid, but there will be no errors attached.
                # So don't return those nodes as having errors in the response.
                if valid is not True and len(reasons) > 0:
                    if node_id not in node_errors:
                        class_type = prompt[node_id]['class_type']
                        node_errors[node_id] = {
                            "errors": reasons,
                            "dependent_outputs": [],
                            "class_type": class_type
                        }
                        logging.error(f"* {class_type} {node_id}:")
                        for reason in reasons:
                            logging.error(f"  - {reason['message']}: {reason['details']}")
                    node_errors[node_id]["dependent_outputs"].append(o)
            logging.error("Output will be ignored")

    if len(good_outputs) == 0:
        errors_list = []
        for o, errors in errors:
            for error in errors:
                errors_list.append(f"{error['message']}: {error['details']}")
        errors_list = "\n".join(errors_list)

        error = {
            "type": "prompt_outputs_failed_validation",
            "message": "Prompt outputs failed validation",
            "details": errors_list,
            "extra_info": {}
        }

        return (False, error, list(good_outputs), node_errors)

    return (True, None, list(good_outputs), node_errors)

MAXIMUM_HISTORY_SIZE = 10000

class PromptQueue:
    def __init__(self, server):
        self.server = server
        self.mutex = threading.RLock()
        self.not_empty = threading.Condition(self.mutex)
        self.task_counter = 0
        self.queue = []
        self.currently_running = {}
        self.history = {}
        self.flags = {}
        server.prompt_queue = self

    def put(self, item):
        with self.mutex:
            heapq.heappush(self.queue, item)
            self.server.queue_updated()
            self.not_empty.notify()

    def get(self, timeout=None):
        with self.not_empty:
            while len(self.queue) == 0:
                self.not_empty.wait(timeout=timeout)
                if timeout is not None and len(self.queue) == 0:
                    return None
            item = heapq.heappop(self.queue)
            i = self.task_counter
            self.currently_running[i] = copy.deepcopy(item)
            self.task_counter += 1
            self.server.queue_updated()
            return (item, i)

    class ExecutionStatus(NamedTuple):
        status_str: Literal['success', 'error']
        completed: bool
        messages: List[str]

    def task_done(self, item_id, history_result,
                  status: Optional['PromptQueue.ExecutionStatus']):
        with self.mutex:
            prompt = self.currently_running.pop(item_id)
            if len(self.history) > MAXIMUM_HISTORY_SIZE:
                self.history.pop(next(iter(self.history)))

            status_dict: Optional[dict] = None
            if status is not None:
                status_dict = copy.deepcopy(status._asdict())

            self.history[prompt[1]] = {
                "prompt": prompt,
                "outputs": {},
                'status': status_dict,
            }
            self.history[prompt[1]].update(history_result)
            self.server.queue_updated()

    def get_current_queue(self):
        with self.mutex:
            out = []
            for x in self.currently_running.values():
                out += [x]
            return (out, copy.deepcopy(self.queue))

    def get_tasks_remaining(self):
        with self.mutex:
            return len(self.queue) + len(self.currently_running)

    def wipe_queue(self):
        with self.mutex:
            self.queue = []
            self.server.queue_updated()

    def delete_queue_item(self, function):
        with self.mutex:
            for x in range(len(self.queue)):
                if function(self.queue[x]):
                    if len(self.queue) == 1:
                        self.wipe_queue()
                    else:
                        self.queue.pop(x)
                        heapq.heapify(self.queue)
                    self.server.queue_updated()
                    return True
        return False

    def get_history(self, prompt_id=None, max_items=None, offset=-1):
        with self.mutex:
            if prompt_id is None:
                out = {}
                i = 0
                if offset < 0 and max_items is not None:
                    offset = len(self.history) - max_items
                for k in self.history:
                    if i >= offset:
                        out[k] = self.history[k]
                        if max_items is not None and len(out) >= max_items:
                            break
                    i += 1
                return out
            elif prompt_id in self.history:
                return {prompt_id: copy.deepcopy(self.history[prompt_id])}
            else:
                return {}

    def wipe_history(self):
        with self.mutex:
            self.history = {}

    def delete_history_item(self, id_to_delete):
        with self.mutex:
            self.history.pop(id_to_delete, None)

    def set_flag(self, name, data):
        with self.mutex:
            self.flags[name] = data
            self.not_empty.notify()

    def get_flags(self, reset=True):
        with self.mutex:
            if reset:
                ret = self.flags
                self.flags = {}
                return ret
            else:
                return self.flags.copy()
