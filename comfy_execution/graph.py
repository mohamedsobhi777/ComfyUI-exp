# Import required modules
import nodes
import logging

from comfy_execution.graph_utils import is_link

# Custom exceptions for graph-related errors
class DependencyCycleError(Exception):
    pass

class NodeInputError(Exception):
    pass

class NodeNotFoundError(Exception):
    pass

class DynamicPrompt:
    """
    Represents a dynamic prompt that can be modified during execution.
    Manages both original nodes from user input and ephemeral nodes created during execution.
    """
    def __init__(self, original_prompt):
        # The original prompt provided by the user
        self.original_prompt = original_prompt
        # Any extra pieces of the graph created during execution
        self.ephemeral_prompt = {}
        # Maps ephemeral nodes to their parent nodes
        self.ephemeral_parents = {}
        # Maps ephemeral nodes to their display nodes
        self.ephemeral_display = {}
        logging.info(f"Created DynamicPrompt with {len(original_prompt)} original nodes")

    def get_node(self, node_id):
        """Get a node by ID from either ephemeral or original prompts"""
        if node_id in self.ephemeral_prompt:
            return self.ephemeral_prompt[node_id]
        if node_id in self.original_prompt:
            return self.original_prompt[node_id]
        raise NodeNotFoundError(f"Node {node_id} not found")

    def has_node(self, node_id):
        """Check if a node exists in either ephemeral or original prompts"""
        return node_id in self.original_prompt or node_id in self.ephemeral_prompt
    
    def add_ephemeral_node(self, node_id, node_info, parent_id, display_id):
        """Add a new ephemeral node with its parent and display information"""
        self.ephemeral_prompt[node_id] = node_info
        self.ephemeral_parents[node_id] = parent_id
        self.ephemeral_display[node_id] = display_id
        logging.info(f"Added ephemeral node {node_id} with parent {parent_id} and display {display_id}")

    def get_real_node_id(self, node_id):
        """Get the original parent node ID by traversing the parent chain"""
        while node_id in self.ephemeral_parents:
            node_id = self.ephemeral_parents[node_id]
        return node_id

    def get_parent_node_id(self, node_id):
        """Get immediate parent node ID if it exists"""
        return self.ephemeral_parents.get(node_id, None)

    def get_display_node_id(self, node_id):
        """Get the display node ID by traversing the display chain"""
        while node_id in self.ephemeral_display:
            node_id = self.ephemeral_display[node_id]
        return node_id

    def all_node_ids(self):
        """Get set of all node IDs from both original and ephemeral prompts"""
        return set(self.original_prompt.keys()).union(set(self.ephemeral_prompt.keys()))

    def get_original_prompt(self):
        """Get the original prompt"""
        return self.original_prompt

def get_input_info(class_def, input_name, valid_inputs=None):
    """
    Get input type information for a node class.
    Returns tuple of (input_type, input_category, extra_info)
    """
    valid_inputs = valid_inputs or class_def.INPUT_TYPES()
    input_info = None
    input_category = None
    if "required" in valid_inputs and input_name in valid_inputs["required"]:
        input_category = "required"
        input_info = valid_inputs["required"][input_name]
    elif "optional" in valid_inputs and input_name in valid_inputs["optional"]:
        input_category = "optional"
        input_info = valid_inputs["optional"][input_name]
    elif "hidden" in valid_inputs and input_name in valid_inputs["hidden"]:
        input_category = "hidden"
        input_info = valid_inputs["hidden"][input_name]
    if input_info is None:
        return None, None, None
    input_type = input_info[0]
    if len(input_info) > 1:
        extra_info = input_info[1]
    else:
        extra_info = {}
    return input_type, input_category, extra_info

class TopologicalSort:
    """
    Implements topological sorting of nodes in the execution graph.
    Tracks dependencies between nodes and manages execution order.
    """
    def __init__(self, dynprompt):
        self.dynprompt = dynprompt
        self.pendingNodes = {}
        self.blockCount = {} # Number of nodes this node is directly blocked by
        self.blocking = {} # Which nodes are blocked by this node
        logging.info("Initialized TopologicalSort")

    def get_input_info(self, unique_id, input_name):
        """Get input information for a specific node and input"""
        class_type = self.dynprompt.get_node(unique_id)["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        return get_input_info(class_def, input_name)

    def make_input_strong_link(self, to_node_id, to_input):
        """Create a strong dependency link from an input to its source node"""
        inputs = self.dynprompt.get_node(to_node_id)["inputs"]
        if to_input not in inputs:
            raise NodeInputError(f"Node {to_node_id} says it needs input {to_input}, but there is no input to that node at all")
        value = inputs[to_input]
        if not is_link(value):
            raise NodeInputError(f"Node {to_node_id} says it needs input {to_input}, but that value is a constant")
        from_node_id, from_socket = value
        self.add_strong_link(from_node_id, from_socket, to_node_id)

    def add_strong_link(self, from_node_id, from_socket, to_node_id):
        """Add a strong dependency link between nodes"""
        if not self.is_cached(from_node_id):
            self.add_node(from_node_id)
            if to_node_id not in self.blocking[from_node_id]:
                self.blocking[from_node_id][to_node_id] = {}
                self.blockCount[to_node_id] += 1
            self.blocking[from_node_id][to_node_id][from_socket] = True
            logging.info(f"Added strong link from {from_node_id}:{from_socket} to {to_node_id}")

    def add_node(self, node_unique_id, include_lazy=False, subgraph_nodes=None):
        """Add a node and its dependencies to the execution graph"""
        node_ids = [node_unique_id]
        links = []
        logging.info(f"Adding node {node_unique_id} to execution graph")

        while len(node_ids) > 0:
            unique_id = node_ids.pop()
            if unique_id in self.pendingNodes:
                continue

            self.pendingNodes[unique_id] = True
            self.blockCount[unique_id] = 0
            self.blocking[unique_id] = {}

            inputs = self.dynprompt.get_node(unique_id)["inputs"]
            for input_name in inputs:
                value = inputs[input_name]
                if is_link(value):
                    from_node_id, from_socket = value
                    if subgraph_nodes is not None and from_node_id not in subgraph_nodes:
                        continue
                    input_type, input_category, input_info = self.get_input_info(unique_id, input_name)
                    is_lazy = input_info is not None and "lazy" in input_info and input_info["lazy"]
                    if (include_lazy or not is_lazy) and not self.is_cached(from_node_id):
                        node_ids.append(from_node_id)
                        links.append((from_node_id, from_socket, unique_id))

        for link in links:
            self.add_strong_link(*link)

    def is_cached(self, node_id):
        """Check if a node's output is cached"""
        return False

    def get_ready_nodes(self):
        """Get list of nodes ready for execution (no blocking dependencies)"""
        ready_nodes = [node_id for node_id in self.pendingNodes if self.blockCount[node_id] == 0]
        logging.info(f"Found {len(ready_nodes)} nodes ready for execution")
        return ready_nodes

    def pop_node(self, unique_id):
        """Remove a completed node from the execution graph"""
        del self.pendingNodes[unique_id]
        for blocked_node_id in self.blocking[unique_id]:
            self.blockCount[blocked_node_id] -= 1
        del self.blocking[unique_id]
        logging.info(f"Removed node {unique_id} from execution graph")

    def is_empty(self):
        """Check if execution graph is empty"""
        return len(self.pendingNodes) == 0

class ExecutionList(TopologicalSort):
    """
    ExecutionList implements a topological dissolve of the graph. After a node is staged for execution,
    it can still be returned to the graph after having further dependencies added.
    """
    def __init__(self, dynprompt, output_cache):
        super().__init__(dynprompt)
        self.output_cache = output_cache
        self.staged_node_id = None
        logging.info("Initialized ExecutionList")

    def is_cached(self, node_id):
        """Check if node output exists in cache"""
        is_cached = self.output_cache.get(node_id) is not None
        if is_cached:
            logging.info(f"Node {node_id} found in cache")
        return is_cached

    def stage_node_execution(self):
        """Stage next node for execution, handling dependency cycles"""
        assert self.staged_node_id is None
        if self.is_empty():
            logging.info("No nodes left to execute")
            return None, None, None
        available = self.get_ready_nodes()
        if len(available) == 0:
            cycled_nodes = self.get_nodes_in_cycle()
            # Because cycles composed entirely of static nodes are caught during initial validation,
            # we will 'blame' the first node in the cycle that is not a static node.
            blamed_node = cycled_nodes[0]
            for node_id in cycled_nodes:
                display_node_id = self.dynprompt.get_display_node_id(node_id)
                if display_node_id != node_id:
                    blamed_node = display_node_id
                    break
            ex = DependencyCycleError("Dependency cycle detected")
            error_details = {
                "node_id": blamed_node,
                "exception_message": str(ex),
                "exception_type": "graph.DependencyCycleError",
                "traceback": [],
                "current_inputs": []
            }
            logging.error(f"Dependency cycle detected, blamed node: {blamed_node}")
            return None, error_details, ex

        self.staged_node_id = self.ux_friendly_pick_node(available)
        logging.info(f"Staged node {self.staged_node_id} for execution")
        return self.staged_node_id, None, None

    def ux_friendly_pick_node(self, node_list):
        # If an output node is available, do that first.
        # Technically this has no effect on the overall length of execution, but it feels better as a user
        # for a PreviewImage to display a result as soon as it can
        # Some other heuristics could probably be used here to improve the UX further.
        def is_output(node_id):
            class_type = self.dynprompt.get_node(node_id)["class_type"]
            class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
            if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
                return True
            return False

        # First priority: direct output nodes
        for node_id in node_list:
            if is_output(node_id):
                logging.info(f"Selected output node {node_id} for execution")
                return node_id

        #This should handle the VAEDecode -> preview case
        for node_id in node_list:
            for blocked_node_id in self.blocking[node_id]:
                if is_output(blocked_node_id):
                    logging.info(f"Selected node {node_id} that blocks output node {blocked_node_id}")
                    return node_id

        #This should handle the VAELoader -> VAEDecode -> preview case
        for node_id in node_list:
            for blocked_node_id in self.blocking[node_id]:
                for blocked_node_id1 in self.blocking[blocked_node_id]:
                    if is_output(blocked_node_id1):
                        logging.info(f"Selected node {node_id} that indirectly blocks output node {blocked_node_id1}")
                        return node_id

        #TODO: this function should be improved
        logging.info(f"Selected first available node {node_list[0]}")
        return node_list[0]

    def unstage_node_execution(self):
        """Cancel staging of current node"""
        assert self.staged_node_id is not None
        logging.info(f"Unstaging node {self.staged_node_id}")
        self.staged_node_id = None

    def complete_node_execution(self):
        """Mark staged node as complete and remove from graph"""
        node_id = self.staged_node_id
        self.pop_node(node_id)
        self.staged_node_id = None
        logging.info(f"Completed execution of node {node_id}")

    def get_nodes_in_cycle(self):
        # We'll dissolve the graph in reverse topological order to leave only the nodes in the cycle.
        # We're skipping some of the performance optimizations from the original TopologicalSort to keep
        # the code simple (and because having a cycle in the first place is a catastrophic error)
        blocked_by = { node_id: {} for node_id in self.pendingNodes }
        for from_node_id in self.blocking:
            for to_node_id in self.blocking[from_node_id]:
                if True in self.blocking[from_node_id][to_node_id].values():
                    blocked_by[to_node_id][from_node_id] = True
        
        # Remove nodes with no blockers until only cycle remains
        to_remove = [node_id for node_id in blocked_by if len(blocked_by[node_id]) == 0]
        while len(to_remove) > 0:
            for node_id in to_remove:
                for to_node_id in blocked_by:
                    if node_id in blocked_by[to_node_id]:
                        del blocked_by[to_node_id][node_id]
                del blocked_by[node_id]
            to_remove = [node_id for node_id in blocked_by if len(blocked_by[node_id]) == 0]
        cycle_nodes = list(blocked_by.keys())
        logging.info(f"Found cycle containing nodes: {cycle_nodes}")
        return cycle_nodes

class ExecutionBlocker:
    """
    Return this from a node and any users will be blocked with the given error message.
    If the message is None, execution will be blocked silently instead.
    Generally, you should avoid using this functionality unless absolutely necessary. Whenever it's
    possible, a lazy input will be more efficient and have a better user experience.
    This functionality is useful in two cases:
    1. You want to conditionally prevent an output node from executing. (Particularly a built-in node
       like SaveImage. For your own output nodes, I would recommend just adding a BOOL input and using
       lazy evaluation to let it conditionally disable itself.)
    2. You have a node with multiple possible outputs, some of which are invalid and should not be used.
       (I would recommend not making nodes like this in the future -- instead, make multiple nodes with
       different outputs. Unfortunately, there are several popular existing nodes using this pattern.)
       like SaveImage. For your own output nodes, I would recommend just adding a BOOL input and using
       lazy evaluation to let it conditionally disable itself.)
    """
    def __init__(self, message):
        self.message = message
        if message:
            logging.info(f"Created ExecutionBlocker with message: {message}")
        else:
            logging.info("Created silent ExecutionBlocker")

