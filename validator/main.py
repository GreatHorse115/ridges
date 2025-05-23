
# Python built in imports 
from pathlib import Path
import sys 
import time
from typing import Optional

# External package imports
from fiber.validator import client as validator
from fiber.chain.interface import get_substrate
from fiber.chain.models import Node
from fiber.chain.chain_utils import load_hotkey_keypair
from loguru import logger
from dotenv import load_dotenv
import httpx
import asyncio
import random
import os

# Internal package imports
from validator.db.operations import DatabaseManager
from validator.challenge.challenge_types import File, FilePair, EmbeddedFile, GeneratedProblemStatement, ChallengeTask
from validator.config import (
    NETUID, SUBTENSOR_NETWORK, SUBTENSOR_ADDRESS,
    WALLET_NAME, HOTKEY_NAME,
    CHALLENGE_TIMEOUT, DB_PATH,
    MAX_MINERS
)
from validator.evaluation.evaluation import CodeGenValidator
from validator.evaluation.evaluation_loop import run_evaluation_loop

project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

# Load environment variables
validator_dir = Path(__file__).parent
env_path = validator_dir / ".env"
load_dotenv(env_path)

async def process_challenge_results():
    """Process challenge results without blocking."""
    pass

async def construct_server_address(node: Node) -> str:
    """Construct server address for a node.
    
    For local development:
    - Nodes register as 0.0.0.1 on the chain (since 127.0.0.1 is not allowed)
    - But we connect to them via 127.0.0.1 locally
    """
    if node.ip == "0.0.0.1":
        # For local development, connect via localhost
        return f"http://127.0.0.1:{node.port}"
    return f"http://{node.ip}:{node.port}"

async def check_miner_availability(
    node: Node, 
    client: httpx.AsyncClient, 
    db_manager: None, 
    hotkey: str
) -> None: 
    """Check if a miner is available and log the result."""
    server_address = construct_server_address(node)
    start_time = time.time()
    
    try:
        headers = {"validator-hotkey": hotkey}
        response = await client.get(f"{server_address}/availability", headers=headers, timeout=5.0)
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        is_available = response.json().get("available", False)
        
        # Log availability check
        db_manager.log_availability_check(
            node_id=node.node_id,
            hotkey=node.hotkey,
            is_available=is_available,
            response_time_ms=response_time
        )
        
        return is_available
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        # Log failed check
        db_manager.log_availability_check(
            node_id=node.node_id,
            hotkey=node.hotkey,
            is_available=False,
            response_time_ms=response_time,
            error=str(e)
        )
        
        logger.warning(f"Failed to check availability for node {node.node_id}: {str(e)}")
        return False

async def get_available_nodes(
    nodes: list[Node],
    client: httpx.AsyncClient,
    db_manager: DatabaseManager,
    hotkey: str
) -> list[Node]:
    """Check availability of all nodes and return available ones up to MAX_MINERS."""
    
    logger.info(f"Checking availability for all {len(nodes)} nodes")
    availability_tasks = [
        check_miner_availability(node, client, db_manager, hotkey)
        for node in nodes
    ]

    availability_results = await asyncio.gather(*availability_tasks)

    # Filter available nodes
    available_nodes = [
        node for node, is_available in zip(nodes, availability_results)
        if is_available
    ]

    total_available = len(available_nodes)
    logger.info(f"Found {total_available} available nodes out of {len(nodes)} total nodes")

    # If we have more available nodes than MAX_MINERS, randomly select MAX_MINERS
    selected_nodes = available_nodes
    if total_available > MAX_MINERS:
        logger.info(f"Randomly selecting {MAX_MINERS} nodes from {total_available} available nodes")
        selected_nodes = random.sample(available_nodes, MAX_MINERS)
    else:
        logger.info(f"Using all {total_available} available nodes (less than MAX_MINERS={MAX_MINERS})")
    
    # Log selected nodes
    for node in selected_nodes:
        logger.info(f"Selected node {node.node_id} (hotkey: {node.hotkey})")
    
    return selected_nodes

async def weights_update_loop(db_manager: None) -> None:
    pass

async def weights_update_loop(db_manager: DatabaseManager) -> None:
    pass

async def periodic_cleanup(db_manager: DatabaseManager, interval_hours: int = 24):
    pass 

async def get_next_challenge_with_retry(hotkey: str, max_retries: int = 2, initial_delay: float = 5.0) -> Optional[dict]:
    pass 

async def main():
    """Main validator loop."""
    
    # Load env vars and make sure they are present
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Load validator hotkey
    try:
        hotkey = load_hotkey_keypair(WALLET_NAME, HOTKEY_NAME)
    except Exception as e:
        logger.error(f"Failed to load keys: {str(e)}")
        return
    
     # Initialize database manager and validator
    logger.info(f"Initializing database manager with path: {DB_PATH}")
    db_manager = DatabaseManager(DB_PATH)

    # Initialize validator instances
    validator = CodeGenValidator(openai_api_key=OPENAI_API_KEY, validator_hotkey=hotkey.ss58_address)
    
    # Creates a susbtrate instance
    substrate = get_substrate(
        subtensor_network=SUBTENSOR_NETWORK,
        subtensor_address=SUBTENSOR_ADDRESS
    )

    # Initialize HTTP client with long timeout
    async with httpx.AsyncClient(timeout=CHALLENGE_TIMEOUT.total_seconds()) as client:
        active_challenge_tasks = []  # Track active challenges

        # Start evaluation loop as a separate task
        logger.info("Starting evaluation loop task...")
        evaluation_task = asyncio.create_task(
            run_evaluation_loop(
                db_path=DB_PATH,
                openai_api_key=OPENAI_API_KEY,
                validator_hotkey=hotkey.ss58_address,
                batch_size=10,
                sleep_interval=120
            )
        )
        evaluation_task.add_done_callback(
            lambda t: logger.error(f"Evaluation task ended unexpectedly: {t.exception()}")
            if t.exception() else None
        )
    # Task 1: Eval loop on active challenges?
        # Track active challenges and creates an eval loop
    # Task 2: Update weights loop
    # Task 3: Periodic cleanup task 

    # runs the main iteration loop within async client
        # Checks if background tasks (eval, weights, cleanup) failed 
        # Removes completed challenges from the active_challenge_tasks list

        # Gets nodes that are active and are less than their stake max
        # Does not issue a challenge if there arent enough miners available

        # Then checks num available nodes, and runs the same min check, sleeping if available < MIN

        # If there are enough, generate and send a challenge -> basically some checks and then it sends a signed request to their internal server

        # Once a challenge is constructred, it logs the current background task status, then generates the challenge object and sends it to miners
        # It extends the active_challenge_tasks with this

        # it then processes and grades any completed challenges

        # Theres some helpers like breaking out of this into the finally on keyboardinterrupt, and logging errors + restarting loop

        # Finally statement cancels eval, weights, cleanup tasks, runs them all and then closes
    
    # After finihsingt he main loop theres also a db manager cleanup 

# Lastly the if name runs it all and then exists out on keyboard interrupt