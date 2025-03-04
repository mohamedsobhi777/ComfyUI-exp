# Import required modules and set up initial configuration
import comfy.options
comfy.options.enable_args_parsing()  # Enable command line argument parsing for ComfyUI

# Standard library imports
import os
import time
import logging
import itertools
import folder_paths
import importlib.util
import utils.extra_config
from comfy.cli_args import args  # Import command line arguments
from app.logger import setup_logger

if __name__ == "__main__":
    #NOTE: These do not do anything on core ComfyUI which should already have no communication with the internet, they are for custom nodes.
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'

# Set up logging based on command line arguments
setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)

def apply_custom_paths():
    """Configure custom paths for models and other resources"""
    # Load extra model paths from config files
    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        utils.extra_config.load_extra_path_config(extra_model_paths_config_path)

    # Load additional model path configs specified via command line
    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            utils.extra_config.load_extra_path_config(config_path)

    # --output-directory, --input-directory, --user-directory
    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logging.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    # These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))
    folder_paths.add_model_folder_path("diffusion_models",
                                       os.path.join(folder_paths.get_output_directory(), "diffusion_models"))
    folder_paths.add_model_folder_path("loras", os.path.join(folder_paths.get_output_directory(), "loras"))

    # Set input and user directories if specified
    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if args.user_directory:
        user_dir = os.path.abspath(args.user_directory)
        logging.info(f"Setting user directory to: {user_dir}")
        folder_paths.set_user_directory(user_dir)


def execute_prestartup_script():
    """Execute prestartup scripts for custom nodes"""
    def execute_script(script_path):
        """Helper function to execute a single prestartup script"""
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            logging.error(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    # Skip if custom nodes are disabled
    if args.disable_all_custom_nodes:
        return

    # Find and execute prestartup scripts in custom node directories
    node_paths = folder_paths.get_folder_paths("custom_nodes")
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        node_prestartup_times = []

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue

            script_path = os.path.join(module_path, "prestartup_script.py")
            if os.path.exists(script_path):
                time_before = time.perf_counter()
                success = execute_script(script_path)
                node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))

    # Log execution times for prestartup scripts
    if len(node_prestartup_times) > 0:
        logging.info("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (PRESTARTUP FAILED)"
            logging.info("{:6.1f} seconds{}: {}".format(n[0], import_message, n[1]))
        logging.info("")

# Apply custom paths and execute prestartup scripts
apply_custom_paths()
execute_prestartup_script()


# Import required modules for main functionality
import asyncio
import shutil
import threading
import gc

# Filter out specific xformers warning on Windows
if os.name == "nt":
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

if __name__ == "__main__":
    # Configure CUDA device if specified
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ['HIP_VISIBLE_DEVICES'] = str(args.cuda_device)
        logging.info("Set cuda device to: {}".format(args.cuda_device))

    # Configure OneAPI device if specified
    if args.oneapi_device_selector is not None:
        os.environ['ONEAPI_DEVICE_SELECTOR'] = args.oneapi_device_selector
        logging.info("Set oneapi device selector to: {}".format(args.oneapi_device_selector))

    # Enable deterministic mode if requested
    if args.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    import cuda_malloc

# Fix for Windows standalone builds
if args.windows_standalone_build:
    try:
        from fix_torch import fix_pytorch_libomp
        fix_pytorch_libomp()
    except:
        pass

# Import ComfyUI specific modules
import comfy.utils
import execution
import server
from server import BinaryEventTypes
import nodes
import comfy.model_management

def cuda_malloc_warning():
    """Check and warn about potential CUDA malloc issues"""
    device = comfy.model_management.get_torch_device()
    device_name = comfy.model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logging.warning("\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")


def prompt_worker(q, server_instance):
    """Main worker function that processes prompts from the queue"""
    current_time: float = 0.0
    e = execution.PromptExecutor(server_instance, lru_size=args.cache_lru)
    # Store executor reference in server
    server_instance.executor = e # GUO1
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0

    while True:
        # Calculate timeout for garbage collection
        timeout = 1000.0
        if need_gc:
            timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

        # Get item from queue
        queue_item = q.get(timeout=timeout)
        if queue_item is not None:
            # Process the queue item
            item, item_id = queue_item
            execution_start_time = time.perf_counter()
            prompt_id = item[1]
            server_instance.last_prompt_id = prompt_id
            print("worker is executing something::", item_id, ", prompt_id::", prompt_id)
            print("gotten item::[0]", item[0])
            print("gotten item::[1]", item[1])
            print("gotten item::[2]", item[2])
            print("gotten item::[3]", item[3])
            print("gotten item::[4]", item[4])

            # print("but this is the pre-cache::", e.caches.recursive_debug_dump())

            # Execute the prompt
            e.execute(item[2], prompt_id, item[3], item[4])
            need_gc = True
            # Mark task as done and update status
            q.task_done(item_id,
                        e.history_result,
                        status=execution.PromptQueue.ExecutionStatus(
                            status_str='success' if e.success else 'error',
                            completed=e.success,
                            messages=e.status_messages))
            print("server_instance_client_id::", server_instance.client_id)
            if server_instance.client_id is not None:
                print("sending executing to client")
                print("prompt_id::", prompt_id)
                server_instance.send_sync("executing", {"node": None, "prompt_id": prompt_id}, server_instance.client_id)

            # Log execution time
            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time
            logging.info("Prompt executed in {:.2f} seconds".format(execution_time))

        # Handle memory management flags
        flags = q.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            comfy.model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if free_memory:
            e.reset()
            need_gc = True
            last_gc_collect = 0

        # Perform garbage collection if needed
        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:
                gc.collect()
                comfy.model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False


async def run(server_instance, address='', port=8188, verbose=True, call_on_start=None):
    """Start the server on specified addresses and ports"""
    addresses = []
    for addr in address.split(","):
        addresses.append((addr, port))
    await asyncio.gather(
        server_instance.start_multi_address(addresses, call_on_start, verbose), server_instance.publish_loop()
    )


def hijack_progress(server_instance):
    """Set up progress reporting hook for the server"""
    def hook(value, total, preview_image):
        comfy.model_management.throw_exception_if_processing_interrupted()
        progress = {"value": value, "max": total, "prompt_id": server_instance.last_prompt_id, "node": server_instance.last_node_id}
        print(f"progress:: {progress}, max:: {total}, prompt_id:: {server_instance.last_prompt_id}, node:: {server_instance.last_node_id}")
        server_instance.send_sync("progress", progress, server_instance.client_id)
        if preview_image is not None:
            server_instance.send_sync(BinaryEventTypes.UNENCODED_PREVIEW_IMAGE, preview_image, server_instance.client_id)

    comfy.utils.set_progress_bar_global_hook(hook)


def cleanup_temp():
    """Clean up temporary directory"""
    temp_dir = folder_paths.get_temp_directory()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def start_comfyui(asyncio_loop=None):
    """
    Initialize and start the ComfyUI server
    
    Args:
        asyncio_loop: Optional existing asyncio event loop
        
    Returns:
        Tuple of (event_loop, server_instance, start_all function)
    """
    
    # if args.temp_directory:
    #     temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
    #     logging.info(f"Setting temp directory to: {temp_dir}")
    #     folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    # Set up event loop
    if not asyncio_loop:
        asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(asyncio_loop)
    prompt_server = server.PromptServer(asyncio_loop)
    q = execution.PromptQueue(prompt_server)

    # Initialize nodes
    nodes.init_extra_nodes(init_custom_nodes=not args.disable_all_custom_nodes)

    # Check for CUDA malloc issues
    cuda_malloc_warning()

    # Set up server routes and progress reporting
    prompt_server.add_routes()
    hijack_progress(prompt_server)

    # Start prompt worker thread
    threading.Thread(target=prompt_worker, daemon=True, args=(q, prompt_server,)).start()

    # Create temp directory
    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    
    # Set up auto-launch if enabled
    call_on_start = None
    if args.auto_launch:
        def startup_server(scheme, address, port):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0':
                address = '127.0.0.1'
            if ':' in address:
                address = "[{}]".format(address)
            webbrowser.open(f"{scheme}://{address}:{port}")
        call_on_start = startup_server

    # Define async startup function
    async def start_all():
        await prompt_server.setup()
        await run(prompt_server, address=args.listen, port=args.port, verbose=not args.dont_print_server, call_on_start=call_on_start)

    # Returning these so that other code can integrate with the ComfyUI loop and server
    return asyncio_loop, prompt_server, start_all


if __name__ == "__main__":
    # Start ComfyUI when running directly
    event_loop, _, start_all_func = start_comfyui()
    try:
        event_loop.run_until_complete(start_all_func())
    except KeyboardInterrupt:
        logging.info("\nStopped server")

    cleanup_temp()
