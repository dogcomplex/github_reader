from dataclasses import dataclass
from pathlib import Path
import aiohttp

@dataclass
class ProcessingContext:
    """Context for metadata processing."""
    session: aiohttp.ClientSession
    level: int
    base_dir: Path
    llm_url: str
    embedding_url: str
    overwrite: bool = False 