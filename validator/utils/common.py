from cryptography.fernet import Fernet
import asyncio
from pathlib import Path

# Get the absolute path to .env
validator_dir = Path(__file__).parents[1]
env_path = validator_dir / ".env"

