from datetime import timedelta
from pathlib import Path
import os

# Network configuration
# Consider defaulting to testnet uid?
NETUID = int(os.getenv("NETUID", "62"))
SUBTENSOR_NETWORK = os.getenv("SUBTENSOR_NETWORK", "test")
SUBTENSOR_ADDRESS = os.getenv("SUBTENSOR_ADDRESS", "127.0.0.1:9944")

# Validator configuration
HOTKEY_NAME = os.getenv("HOTKEY_NAME", "default")
WALLET_NAME = os.getenv("WALLET_NAME", "default")
MIN_STAKE_THRESHOLD = float(os.getenv("MIN_STAKE_THRESHOLD", "2"))
VALIDATOR_PORT = int(os.getenv("VALIDATOR_PORT", "8000"))
VALIDATOR_HOST = os.getenv("VALIDATOR_HOST", "0.0.0.0")

# Default configuration values
MIN_MINERS = 1
MAX_MINERS = 25

# Additional settings needed for operation
CHALLENGE_INTERVAL = timedelta(minutes=1)
CHALLENGE_TIMEOUT = timedelta(minutes=4)

WEIGHTS_INTERVAL = timedelta(minutes=30)
VALIDATION_DELAY = timedelta(minutes=5)

DB_PATH = Path("validator.db")

VERSION_KEY = 2

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")

# Log initial configuration
import logging
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

logger.info("Validator Configuration:")
logger.info(f"Network: {SUBTENSOR_NETWORK}")
logger.info(f"Netuid: {NETUID}")
logger.info(f"Min miners: {MIN_MINERS}")
logger.info(f"Max miners: {MAX_MINERS}")
logger.info(f"Min stake threshold: {MIN_STAKE_THRESHOLD}")
logger.info(f"Challenge interval: {CHALLENGE_INTERVAL}")
logger.info(f"Challenge timeout: {CHALLENGE_TIMEOUT}")
logger.info(f"Weights interval: {WEIGHTS_INTERVAL}")
logger.info(f"DB path: {DB_PATH}")
logger.info(f"Log level: {LOG_LEVEL}")
logger.info(f"Validation delay: {VALIDATION_DELAY}")