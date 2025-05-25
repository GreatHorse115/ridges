from pathlib import Path

async def run_evaluation_loop(
    db_path: Path,
    openai_api_key: str,
    validator_hotkey: str,
    batch_size: int = 10,
    sleep_interval: int = 60
) -> None:
    pass