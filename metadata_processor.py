from typing import Dict, List, Optional, Any, Callable
import aiohttp
import logging
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from tqdm.asyncio import tqdm
import asyncio
import re
from sklearn.cluster import KMeans
from tenacity import retry, stop_after_attempt, wait_exponential
import json
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException

# Import from metadata_config instead of having circular imports
from metadata_config import MetadataConfig, MetadataScope, DEFAULT_METADATA_CONFIGS
from group_summaries import DocumentInfo
from processing_context import ProcessingContext

@dataclass
class ProcessingContext:
    """Context for metadata processing."""
    session: aiohttp.ClientSession
    level: int
    base_dir: Path
    llm_url: str
    embedding_url: str
    overwrite: bool = False

class MetadataProcessor:
    def __init__(
        self,
        configs: Optional[Dict[str, MetadataConfig]] = None,
        custom_processors: Optional[Dict[str, Callable]] = None
    ):
        # Use DEFAULT_METADATA_CONFIGS if no configs provided
        self.configs = configs if configs is not None else DEFAULT_METADATA_CONFIGS
        
        # Apply any custom processors
        if custom_processors:
            for name, processor in custom_processors.items():
                if name in self.configs:
                    self.configs[name].processor = processor

    def _sort_by_dependencies(self, configs: Dict[str, MetadataConfig]) -> List[MetadataConfig]:
        """Sort configs by dependencies."""
        # Create dependency graph
        graph = {name: set(config.dependencies) for name, config in configs.items()}
        
        # Topological sort
        sorted_names = []
        visited = set()
        temp_visited = set()
        
        def visit(name: str):
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected: {name}")
            if name not in visited:
                temp_visited.add(name)
                for dep in graph[name]:
                    visit(dep)
                temp_visited.remove(name)
                visited.add(name)
                sorted_names.append(name)
                
        for name in graph:
            if name not in visited:
                visit(name)
                
        # Return configs in dependency order
        return [configs[name] for name in sorted_names]

    async def process_metadata(
        self,
        documents: Dict[str, DocumentInfo],
        context: ProcessingContext
    ) -> Dict[str, Dict[str, Any]]:
        """Process metadata types one document at a time."""
        results = {}
        ordered_configs = self._sort_by_dependencies(self.configs)
        
        # Process one document at a time
        for doc_id, doc in documents.items():
            for config in ordered_configs:
                if not config.enable:
                    continue
                    
                output_path = config.get_output_path(doc, context.level, context.base_dir)
                if not context.overwrite and output_path.exists():
                    # Skip if already processed
                    continue
                    
                try:
                    result = await config.processor(doc, context, config)
                    if result:
                        if config.name not in results:
                            results[config.name] = {}
                        results[config.name][doc_id] = result
                except Exception as e:
                    logging.error(f"Error processing {config.name} for {doc_id}: {str(e)}")
                    
        return results

    def _create_chunks(
        self, 
        content: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """Split content into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(content):
            end = start + chunk_size
            if end > len(content):
                end = len(content)
            chunks.append(content[start:end])
            start = end - overlap
        return chunks

    def _merge_chunk_results(
        self,
        results: List[Any],
        strategy: str
    ) -> Any:
        """Merge results from multiple chunks based on strategy."""
        if not results:
            return None
            
        if strategy == "concat":
            if isinstance(results[0], str):
                return " ".join(results)
            elif isinstance(results[0], list):
                merged = []
                for result in results:
                    merged.extend(result)
                return merged
                
        elif strategy == "unique":
            if isinstance(results[0], list):
                merged = set()
                for result in results:
                    merged.update(result)
                return list(merged)
                
        return results[0]  # Default to first result

    def _check_dependencies(
        self,
        config: MetadataConfig,
        results: Dict[str, Any]
    ) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in results for dep in config.dependencies)
    
    def _should_process_level(self, config: MetadataConfig, level: int) -> bool:
        """Check if metadata should be processed for this level."""
        if level < config.min_level:
            return False
        if config.max_level and level > config.max_level:
            return False
        return True
    
    async def _process_documents(
        self,
        documents: Dict[str, DocumentInfo],
        config: MetadataConfig,
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Process metadata for individual documents."""
        results = {}
        
        # Create tasks for all documents
        tasks = []
        for doc_name, doc in documents.items():
            task = self._process_single_document(doc_name, doc, config, context)
            tasks.append(task)
        
        # Process tasks with nested progress bar
        with tqdm(
            total=len(documents),
            desc=f"Processing {config.name}",
            position=1,
            leave=False,
            dynamic_ncols=True  # Adapt to terminal width
        ) as doc_pbar:
            for completed_task in asyncio.as_completed(tasks):
                doc_name, result = await completed_task
                if result is not None:  # Skip failed documents
                    results[doc_name] = result
                doc_pbar.update(1)
                
        return results
    
    async def _process_single_document(
        self,
        doc_name: str,
        doc: DocumentInfo,
        config: MetadataConfig,
        context: ProcessingContext
    ) -> tuple[str, Any]:
        """Process a single document for the given metadata type."""
        output_path = config.get_output_path(doc, context.level, context.base_dir)
        
        # Skip if file exists and not overwriting
        if not context.overwrite and output_path.exists():
            return doc_name, self._load_metadata(output_path)
        
        try:
            # Use custom processor if available
            if config.name in self.custom_processors:
                result = await self.custom_processors[config.name](
                    doc, context, config
                )
            else:
                result = await self._default_llm_processor(
                    doc, context, config
                )
            
            self._save_metadata(result, output_path)
            return doc_name, result
            
        except Exception as e:
            logging.error(f"Error processing {config.name} for {doc_name}: {str(e)}")
            return doc_name, None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _default_llm_processor(
        self,
        doc: DocumentInfo,
        context: ProcessingContext,
        config: MetadataConfig
    ) -> Any:
        """Default processor using LLM API with content splitting."""
        content = doc.content
        
        # Pre-emptively split if content too large
        if len(content) > self.max_chars:
            logging.info(
                f"Content too large ({len(content)} chars) for {config.name} processing of '{doc.name}', splitting"
            )
            splits = await self._split_and_process(
                doc, content, context, config
            )
            
            # Handle results based on metadata type
            if config.name == "TAGS":
                # Combine and deduplicate tags
                all_tags = []
                for split_result in splits:
                    all_tags.extend(tag.strip() for tag in split_result.split(','))
                return ', '.join(sorted(set(all_tags)))
            
            elif config.name == "SUMMARY":
                # For summaries, join splits and create a final summary
                combined = "\n\n---\n\n".join(splits)
                return await self._default_llm_processor(
                    DocumentInfo(doc.priority, doc.name, combined),
                    context,
                    config
                )
            
            elif config.name == "REPORT":
                # For reports, just join with separators
                return "\n\n---\n\n".join(splits)
            
            else:
                # Default behavior: join with separators
                return "\n\n---\n\n".join(splits)

        system_message = {
            "role": "system",
            "content": config.system_prompt_template.format(level=context.level)
        }
        
        user_content = config.user_prompt_template or "Process this content:\n\n{content}"
        user_message = {
            "role": "user",
            "content": user_content.format(content=content)
        }
        
        # Match error handling from summarize.py
        try:
            async with context.session.post(
                context.llm_url,
                json={
                    "messages": [system_message, user_message],
                    "model": "local-model"
                }
            ) as response:
                if response.status != 200:
                    raise Exception(f"API error {response.status}: {await response.text()}")
                result = await response.json()
                return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            logging.error(f"API call failed: {str(e)}")
            raise
    
    def _load_metadata(self, path: Path) -> Any:
        """Load metadata from file."""
        if path.suffix == '.npy':
            return np.load(path)
        return path.read_text(encoding='utf-8')
    
    def _save_metadata(self, data: Any, path: Path):
        """Save metadata to file."""
        if isinstance(data, np.ndarray):
            np.save(path, data)
        else:
            path.write_text(str(data), encoding='utf-8') 
    
    async def _get_embedding(
        self,
        text: str,
        context: ProcessingContext
    ) -> Optional[np.ndarray]:
        """Get embedding with translation support."""
        try:
            # Try to find translation first
            doc_name = text[:100]  # Use start of text to find doc
            translation_path = context.base_dir / f"level_{context.level}" / "translations" / f"{doc_name}_TRANSLATION.md"
            
            if translation_path.exists():
                text = translation_path.read_text()
            
            async with context.session.post(
                context.embedding_url,
                json={
                    "input": text,
                    "model": "text-embedding-bge-m3"
                }
            ) as response:
                if response.status != 200:
                    logging.error(f"Embedding API error {response.status}: {await response.text()}")
                    return None
                
                result = await response.json()
                if "data" not in result or not result["data"]:
                    logging.error(f"Invalid embedding response format: {result}")
                    return None
                
                return np.array(result["data"][0]["embedding"])
            
        except Exception as e:
            logging.error(f"Error getting embedding: {str(e)}")
            return None

    async def _process_single(
        self,
        doc_id: str,
        doc: DocumentInfo,
        config: MetadataConfig,
        context: ProcessingContext
    ) -> Any:
        """Process a single document."""
        # For CHUNK type, return the chunk mapping
        if config.name == "CHUNK":
            return await config.processor(doc, context, config)
            
        # For other types, check if we should use chunked content
        if "CHUNK" in self.configs and config.name != "CHUNK":
            chunk_config = self.configs["CHUNK"]
            chunk_path = chunk_config.get_output_path(doc, context.level, context.base_dir).parent
            chunks = sorted(chunk_path.glob(f"{doc.name}_*of*_CHUNK.md"))
            
            if chunks:
                # Process each chunk and merge results
                results = []
                for chunk_path in chunks:
                    chunk_content = chunk_path.read_text()
                    chunk_doc = DocumentInfo(doc.priority, doc.name, chunk_content)
                    result = await config.processor(chunk_doc, context, config)
                    results.append(result)
                    
                # Merge results based on type
                if isinstance(results[0], str):
                    return " ".join(results)
                elif isinstance(results[0], list):
                    merged = []
                    for r in results:
                        merged.extend(r)
                    return merged
                    
        # Fall back to processing full document
        return await config.processor(doc, context, config)