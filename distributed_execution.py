import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Set, Tuple
import urllib.request
import urllib.parse
from comfy_execution.graph import ExecutionList, DynamicPrompt
import nodes

class DistributedExecutionManager:
    def __init__(self, main_instance_port: int = 8188):
        self.main_instance_port = main_instance_port
        self.worker_instances: Dict[str, WorkerInstance] = {}
        self.task_assignments: Dict[str, Dict[str, str]] = {}  # prompt_id -> {node_id -> worker_id}
        self.available_workers: Set[str] = set()
        self.node_outputs: Dict[str, Dict[str, any]] = {}  # prompt_id -> {node_id -> output}
        
    def add_worker(self, host: str, port: int) -> str:
        worker_id = str(uuid.uuid4())
        worker = WorkerInstance(host, port, worker_id)
        self.worker_instances[worker_id] = worker
        self.available_workers.add(worker_id)
        logging.info(f"Added worker {worker_id} at {host}:{port}")
        return worker_id
        
    def remove_worker(self, worker_id: str):
        if worker_id in self.worker_instances:
            del self.worker_instances[worker_id]
            self.available_workers.discard(worker_id)
            logging.info(f"Removed worker {worker_id}")
            
    def get_available_worker(self) -> Optional[str]:
        if not self.available_workers:
            return None
        return next(iter(self.available_workers))
        
    def mark_worker_busy(self, worker_id: str):
        self.available_workers.discard(worker_id)
        
    def mark_worker_available(self, worker_id: str):
        if worker_id in self.worker_instances:
            self.available_workers.add(worker_id)

    def _create_subprompt(self, prompt: dict, node_id: str) -> dict:
        """Create a subprompt containing only the specified node and its direct dependencies."""
        node = prompt[node_id]
        subprompt = {node_id: node}
        
        # Add input dependencies
        for input_name, input_value in node["inputs"].items():
            if isinstance(input_value, list) and len(input_value) == 2:
                dep_node_id = input_value[0]
                if dep_node_id in prompt:
                    subprompt[dep_node_id] = prompt[dep_node_id]
                    
        return subprompt

    async def _execute_node(self, host: str, port: int, prompt: dict, node_id: str, client_id: str, prompt_id: str) -> Tuple[str, any]:
        """Execute a single node on a worker instance."""
        server_address = f"{host}:{port}"
        
        # Create subprompt for this node
        subprompt = self._create_subprompt(prompt, node_id)
        
        # Queue the subprompt
        p = {"prompt": subprompt, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        
        sub_prompt_id = result['prompt_id']
        
        # Connect to websocket to monitor execution
        async with websockets.connect(f"ws://{server_address}/ws?clientId={client_id}") as ws:
            # Monitor execution
            while True:
                out = await ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['node'] is None and data['prompt_id'] == sub_prompt_id:
                            break  # Execution complete
                
        # Get execution history
        with urllib.request.urlopen(f"http://{server_address}/history/{sub_prompt_id}") as response:
            history = json.loads(response.read())
            
        # Extract node output from history
        if sub_prompt_id in history and "outputs" in history[sub_prompt_id]:
            node_output = history[sub_prompt_id]["outputs"].get(node_id)
            return node_id, node_output
        return node_id, None

    async def execute_workflow(self, prompt: dict, client_id: str) -> dict:
        """Execute a workflow by distributing nodes across available workers."""
        prompt_id = str(uuid.uuid4())
        self.task_assignments[prompt_id] = {}
        self.node_outputs[prompt_id] = {}
        
        # Create execution list to get topological order
        dynprompt = DynamicPrompt(prompt)
        execution_list = ExecutionList(dynprompt, {})
        
        # Execute nodes in topological order
        while True:
            node_id, error, _ = execution_list.stage_node_execution()
            if error:
                raise Exception(f"Error in workflow: {error}")
            if node_id is None:
                break  # No more nodes to execute
                
            # Get available worker
            worker_id = self.get_available_worker()
            if not worker_id:
                # If no worker available, execute on main instance
                worker_id = "main"
                host, port = "localhost", self.main_instance_port
            else:
                worker = self.worker_instances[worker_id]
                host, port = worker.host, worker.port
                self.mark_worker_busy(worker_id)
                
            try:
                # Execute node
                self.task_assignments[prompt_id][node_id] = worker_id
                node_id, output = await self._execute_node(host, port, prompt, node_id, client_id, prompt_id)
                if output:
                    self.node_outputs[prompt_id][node_id] = output
                
                if worker_id != "main":
                    self.mark_worker_available(worker_id)
                    
            except Exception as e:
                logging.error(f"Error executing node {node_id} on worker {worker_id}: {e}")
                if worker_id != "main":
                    self.mark_worker_available(worker_id)
                # Fallback to main instance
                node_id, output = await self._execute_node("localhost", self.main_instance_port, prompt, node_id, client_id, prompt_id)
                if output:
                    self.node_outputs[prompt_id][node_id] = output
                    
            execution_list.complete_node_execution()
            
        # Combine results
        result = {
            "prompt": {
                "prompt_id": prompt_id,
                "prompt": prompt
            },
            "outputs": self.node_outputs[prompt_id]
        }
        
        # Cleanup
        del self.task_assignments[prompt_id]
        del self.node_outputs[prompt_id]
        
        return result

class WorkerInstance:
    def __init__(self, host: str, port: int, worker_id: str):
        self.host = host
        self.port = port
        self.worker_id = worker_id
        self.status = "available" 