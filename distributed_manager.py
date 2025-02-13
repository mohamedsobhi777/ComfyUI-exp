import argparse
import json
import logging
import requests
import subprocess
import sys
import time
from typing import List, Optional
import urllib.parse

# Configure more detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DistributedManager:
    def __init__(self, main_port: int = 8188):
        self.main_port = main_port
        self.main_url = f"http://localhost:{main_port}"
        self.worker_processes: List[subprocess.Popen] = []
        self.worker_ports: List[int] = []
        logger.info("D1. Initialized Distributed Manager")
        logger.info(f"D2. Main instance URL: {self.main_url}")
        
    def check_main_instance(self) -> bool:
        """Check if the main ComfyUI instance is running."""
        try:
            logger.info("D3. Checking main instance status")
            response = requests.get(self.main_url)
            if response.status_code == 200:
                logger.info("D4. Main instance is running")
                return True
            else:
                logger.error(f"D4x. Main instance returned error status: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logger.error("D4x. Main instance is not running")
            return False
            
    def get_registered_workers(self) -> List[dict]:
        """Get list of currently registered workers from main instance."""
        try:
            logger.info("D5. Fetching registered workers")
            response = requests.get(f"{self.main_url}/guo/distributed/workers")
            if response.status_code == 200:
                workers = response.json()
                logger.info(f"D6. Found {len(workers)} registered workers")
                for worker in workers:
                    logger.info(f"D7. Worker: {worker['worker_id']} on port {worker['port']}")
                return workers
            else:
                logger.error(f"D6x. Failed to get workers: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"D6x. Error getting workers: {e}")
            return []
        
    def start_worker(self, port: int) -> Optional[subprocess.Popen]:
        """Start a new ComfyUI worker instance on the specified port."""
        logger.info(f"D8. Starting new worker on port {port}")
        
        if not self.check_main_instance():
            logger.error("D9x. Cannot start worker - main instance not running")
            return None
            
        try:
            # Check port availability
            try:
                logger.info(f"D9. Checking if port {port} is available")
                response = requests.get(f"http://localhost:{port}")
                logger.error(f"D10x. Port {port} is already in use")
                return None
            except requests.exceptions.ConnectionError:
                logger.info(f"D10. Port {port} is available")
                
            # Start ComfyUI instance
            logger.info(f"D11. Launching worker process on port {port}")
            process = subprocess.Popen(
                ["python3", "main.py", "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for startup
            logger.info(f"D12. Waiting for worker to start")
            max_retries = 30
            for i in range(max_retries):
                try:
                    response = requests.get(f"http://localhost:{port}")
                    if response.status_code == 200:
                        logger.info(f"D13. Worker is running on port {port}")
                        break
                except requests.exceptions.ConnectionError:
                    if i == max_retries - 1:
                        logger.error(f"D13x. Worker failed to start after {max_retries} attempts")
                        return None
                    time.sleep(1)
                    logger.debug(f"D12.{i+1} Waiting for worker startup...")
                    
            # Register worker
            logger.info(f"D14. Registering worker with main instance")
            response = requests.post(
                f"{self.main_url}/guo/distributed/add_worker",
                json={"host": "localhost", "port": port}
            )
            
            if response.status_code == 200:
                worker_id = response.json()["worker_id"]
                self.worker_processes.append(process)
                self.worker_ports.append(port)
                logger.info(f"D15. Worker registered successfully:")
                logger.info(f"D16. - Worker ID: {worker_id}")
                logger.info(f"D17. - Port: {port}")
                
                self.get_registered_workers()
                return process
            else:
                logger.error(f"D15x. Failed to register worker: {response.text}")
                process.kill()
                return None
                
        except Exception as e:
            logger.error(f"D18x. Error starting worker: {e}")
            return None
            
    def stop_worker(self, port: int):
        """Stop a specific worker instance."""
        logger.info(f"Stopping worker on port {port}")
        try:
            idx = self.worker_ports.index(port)
            process = self.worker_processes[idx]
            
            # Try to gracefully remove the worker from the main instance
            try:
                # Get worker info to find its ID
                workers = self.get_registered_workers()
                worker_id = None
                for worker in workers:
                    if worker["port"] == port:
                        worker_id = worker["worker_id"]
                        break
                        
                if worker_id:
                    logger.info(f"Deregistering worker {worker_id} from main instance")
                    response = requests.post(
                        f"{self.main_url}/guo/distributed/remove_worker",
                        json={"worker_id": worker_id}
                    )
                    if response.status_code == 200:
                        logger.info(f"Successfully deregistered worker {worker_id}")
                    else:
                        logger.warning(f"Failed to deregister worker: {response.text}")
                else:
                    logger.warning(f"Could not find worker ID for port {port}")
                    
            except Exception as e:
                logger.warning(f"Failed to deregister worker on port {port}: {e}")
                
            # Kill the process
            logger.info(f"Terminating worker process on port {port}")
            process.kill()
            process.wait()
            
            # Remove from our tracking lists
            self.worker_processes.pop(idx)
            self.worker_ports.pop(idx)
            
            logger.info(f"Successfully stopped worker on port {port}")
            
            # Show current workers
            self.get_registered_workers()
            
        except ValueError:
            logger.error(f"No worker found on port {port}")
        except Exception as e:
            logger.error(f"Error stopping worker on port {port}: {e}")
            
    def stop_all(self):
        """Stop all worker instances."""
        logger.info("Stopping all workers")
        ports = self.worker_ports.copy()  # Create copy since we'll modify during iteration
        for port in ports:
            self.stop_worker(port)
            
def main():
    parser = argparse.ArgumentParser(description="Manage distributed ComfyUI instances")
    parser.add_argument("--main-port", type=int, default=8188, help="Port of the main ComfyUI instance")
    parser.add_argument("--worker-ports", type=str, default="8288", help="Comma-separated list of ports for worker instances")
    
    args = parser.parse_args()
    
    manager = DistributedManager(args.main_port)
    
    try:
        # First check if main instance is running
        if not manager.check_main_instance():
            logger.error("Main ComfyUI instance must be running first")
            return
            
        # Start workers on specified ports
        worker_ports = [int(p.strip()) for p in args.worker_ports.split(",")]
        logger.info(f"Starting workers on ports: {worker_ports}")
        
        for port in worker_ports:
            manager.start_worker(port)
            
        # Keep the script running and periodically check worker status
        logger.info("Press Ctrl+C to stop all instances")
        while True:
            time.sleep(10)  # Check every 10 seconds
            logger.info("Current worker status:")
            manager.get_registered_workers()
            
    except KeyboardInterrupt:
        logger.info("Stopping all instances...")
        manager.stop_all()
        
if __name__ == "__main__":
    main() 