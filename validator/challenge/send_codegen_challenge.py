'''
Ridges has multiple challenge types. The first is one shot code gen.
We are currently built on Rayon Labs' Fiber + use our synthetic code generation system to send tasks to miners
'''

import httpx
from fiber import Keypair
from fiber.logging_utils import get_logger

from validator.challenge.challenge_types import GeneratedCodegenProblem, HyrdatedGeneratedCodegenProblem
from validator.challenge.create_codegen_challenge import dehydrate_codegen_problem
from validator.utils.async_utils import AsyncBarrier
from validator.db.operations import DatabaseManager
from validator.config import CHALLENGE_TIMEOUT

logger = get_logger(__name__)

async def send_challenge(
    challenge: HyrdatedGeneratedCodegenProblem,
    server_address: str,
    hotkey: str,
    keypair: Keypair,
    node_id: int,
    barrier: AsyncBarrier,
    db_manager: DatabaseManager = None,
    client: httpx.AsyncClient = None,
    timeout: float = CHALLENGE_TIMEOUT.total_seconds()  # Use config timeout in seconds
) -> httpx.Response:
    """Send a challenge to a miner node using fiber 2.0.0 protocol."""
    payload = dehydrate_codegen_problem(challenge).to_dict()

    logger.info(f"Preparing to send challenge to node {node_id}")
    logger.info(f"  Server address: {server_address}")
    logger.info(f"  Hotkey: {hotkey}")
    logger.info(f"  Challenge ID: {challenge.problem_uuid}")

    remaining_barriers = 2
    response = None 

    try:
        # First, store the challenge in the challenges table
        if db_manager:
            logger.debug(f"Storing challenge {challenge.problem_uuid} in database")
            db_manager.store_challenge(
                challenge_id=challenge.problem_uuid,
                challenge_type=str(challenge.type),  # Convert enum to string
                video_url=challenge.video_url,
                task_name="soccer"
            )
        
         # Record the assignment
        if db_manager:
            logger.debug(f"Recording challenge assignment in database")
            db_manager.assign_challenge(challenge.challenge_id, hotkey, node_id)

        # Create client if not provided
        should_close_client = False
        if client is None:
            logger.debug("Creating new HTTP client")
            client = httpx.AsyncClient(timeout=timeout)
            should_close_client = True
        
        # except Exception as e:
        #     if remaining_barriers:
        #         await barrier.wait()
        #         remaining_barriers -= 1

        #         logger.error(f"Response error: {str(e)}")
        #         logger.error(f"Response status code: {response.status_code}")
        #         logger.error(f"Response headers: {response.headers}")
        #         error_msg = f"Failed to send challenge {challenge.problem_uuid} to {hotkey} (node {node_id}): {str(e)}"
        #         logger.error(error_msg)
        #         logger.error("Full error traceback:", exc_info=True)
        #         raise ValueError(error_msg)
    
    except Exception as e:
        if remaining_barriers: 
            await barrier.wait()
            remaining_barriers -= 1
        if remaining_barriers: 
            await barrier.wait()
            remaining_barriers -= 1
        error_msg = f"Failed to send challenge {challenge.problem_uuid} to {hotkey} (node {node_id}): {str(e)}"
        logger.error(error_msg)
        logger.error("Full error traceback:", exc_info=True)
        raise ValueError(error_msg)
        

    