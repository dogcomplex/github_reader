from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, List

class MetadataScope(Enum):
    SINGLE = "single"
    GROUP = "group"

@dataclass
class MetadataConfig:
    name: str
    scope: MetadataScope
    file_extension: str = ".md"
    min_level: int = 1
    max_level: Optional[int] = None
    processor: Optional[Callable] = None
    system_prompt_template: Optional[str] = None
    user_prompt_template: Optional[str] = None
    output_directory: str = ""
    group_size: Optional[int] = None
    split_strategy: str = "default"
    enable: bool = True
    override: bool = False
    dependencies: List[str] = field(default_factory=list)
    max_chars: int = 30000
    cache_strategy: str = "file"
    input_type: str = "text"
    output_type: str = "text"
    parallel: bool = True
    retry_count: int = 3
    # Add new chunking parameters
    chunk_size: Optional[int] = None
    chunk_overlap: int = 1000
    merge_strategy: str = "concat"
    
    def get_output_path(
        self,
        doc_info: 'DocumentInfo',
        level: int,
        base_dir: Path
    ) -> Path:
        """Generate output path for this metadata type."""
        level_dir = base_dir / f"level_{level}"
        type_dir = level_dir / (self.output_directory or self.name.lower())
        type_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{doc_info.priority:06d}_{doc_info.name}"
        if level > 1:
            filename += f"_{level}"
        filename += f"_{self.name}{self.file_extension}"
        
        return type_dir / filename

@dataclass
class DocumentInfo:
    priority: int
    name: str
    content: str 