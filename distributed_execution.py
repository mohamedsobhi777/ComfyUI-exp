import asyncio
import aiohttp
import logging
import uuid
from typing import Dict, Optional

class WorkerInstance:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.status = "disconnected"
        self.ws_client: Optional[aiohttp.ClientWebSocketResponse] = None
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"

    async def connect(self):
        """Establish WebSocket connection with worker"""
        try:
            session = aiohttp.ClientSession()
            self.ws_client = await session.ws_connect(self.ws_url)
            self.status = "connected"
            logging.info(f"Connected to worker at {self.ws_url}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to worker: {e}")
            self.status = "error"
            return False

    async def disconnect(self):
        """Close WebSocket connection"""
        if self.ws_client:
            await self.ws_client.close()
            self.status = "disconnected"

    async def execute_node(self, node_data):
        """Execute a single node on this worker"""
        if not self.ws_client:
            raise RuntimeError("Worker not connected")
        
        try:
            # Send node execution request
            await self.ws_client.send_json({
                "type": "execute_node",
                "data": node_data
            })
            
            # Wait for response
            response = await self.ws_client.receive_json()
            return response
        except Exception as e:
            logging.error(f"Error executing node on worker: {e}")
            raise

class DistributedExecutionManager:
    def __init__(self):
        self.worker_instances: Dict[str, WorkerInstance] = {}

    def add_worker(self, host: str, port: int) -> str:
        """Add a new worker instance"""
        worker_id = str(uuid.uuid4())
        self.worker_instances[worker_id] = WorkerInstance(host, port)
        logging.info(f"Added worker {worker_id} at {host}:{port}")
        return worker_id

    def remove_worker(self, worker_id: str):
        """Remove a worker instance"""
        if worker_id in self.worker_instances:
            worker = self.worker_instances[worker_id]
            asyncio.create_task(worker.disconnect())
            del self.worker_instances[worker_id]
            logging.info(f"Removed worker {worker_id}")

    async def connect_worker(self, worker_id: str) -> bool:
        """Connect to a worker instance"""
        if worker_id not in self.worker_instances:
            return False
        return await self.worker_instances[worker_id].connect()

    async def execute_workflow(self, prompt: dict, client_id: str) -> dict:
        """Execute a workflow across distributed workers"""
        # TODO: Implement node distribution logic
        # For now, this is a placeholder that will need to be expanded
        return {
            "status": "success",
            "prompt": {
                "prompt_id": str(uuid.uuid4())
            }
        }

    def get_worker_status(self, worker_id: str) -> str:
        """Get status of a worker instance"""
        if worker_id in self.worker_instances:
            return self.worker_instances[worker_id].status
        return "not_found" 