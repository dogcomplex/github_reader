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
        custom_processors: Optional[Dict[str, Callable]] = None,
        context_size: int = 30000,
        chars_per_token: float = 3.5,
        safety_margin: float = 0.8
    ):
        self.configs = configs or DEFAULT_METADATA_CONFIGS.copy()
        self.custom_processors = custom_processors or {}
        self.max_chars = int(context_size * chars_per_token * safety_margin)
        
    async def process_metadata(
        self,
        documents: Dict[str, DocumentInfo],
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Process all metadata types for the current level."""
        results = {}
        
        # Get configs for this level
        single_configs = [
            config for config in self.configs.values()
            if config.scope == MetadataScope.SINGLE 
            and self._should_process_level(config, context.level)
        ]
        
        group_configs = [
            config for config in self.configs.values()
            if config.scope == MetadataScope.GROUP 
            and self._should_process_level(config, context.level)
        ]

        # Create main progress bar for metadata types
        with tqdm(
            total=len(single_configs) + len(group_configs),
            desc="Processing metadata types",
            position=0,
            leave=True
        ) as metadata_pbar:
            # Process single-document metadata
            for config in single_configs:
                metadata_pbar.set_description(f"Processing {config.name}")
                results[config.name] = await self._process_single_metadata(
                    config, documents, context
                )
                metadata_pbar.update(1)

            # Process group metadata
            for config in group_configs:
                metadata_pbar.set_description(f"Processing {config.name}")
                results[config.name] = await self._process_group_metadata(
                    config, documents, context
                )
                metadata_pbar.update(1)
                
        return results
    
    def _should_process_level(self, config: MetadataConfig, level: int) -> bool:
        """Check if metadata should be processed for this level."""
        if level < config.min_level:
            return False
        if config.max_level and level > config.max_level:
            return False
        return True
    
    async def _process_single_metadata(
        self,
        config: MetadataConfig,
        documents: Dict[str, DocumentInfo],
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
    
    async def _process_group_metadata(
        self,
        config: MetadataConfig,
        documents: Dict[str, DocumentInfo],
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Process metadata that operates on groups of documents."""
        results = {}
        
        # Skip if not enough documents for grouping
        if len(documents) <= 1:
            return results
            
        # For GROUP type, perform clustering and create groups
        if config.name == "GROUP":
            # For GROUP type, we need embeddings first
            if config.requires_embedding:
                embeddings = {}
                embedding_config = self.configs.get("EMBEDDING")
                if not embedding_config:
                    logging.error("EMBEDDING config required but not found")
                    return results
                    
                # Get embeddings for all documents
                for doc_name, doc in documents.items():
                    embedding_path = embedding_config.get_output_path(
                        doc, context.level, context.base_dir
                    )
                    
                    # Match caching behavior from group_summaries.py
                    if not context.overwrite and embedding_path.exists():
                        embedding = np.load(embedding_path)
                    else:
                        try:
                            # Match embedding generation from SummaryGrouper
                            cleaned_text = re.sub(r'#+ ', '', doc.content)
                            cleaned_text = re.sub(r'[^\w\s.,!?-]', ' ', cleaned_text)
                            cleaned_text = ' '.join(cleaned_text.split())
                            
                            embedding = await self._get_embedding(cleaned_text, context)
                            if embedding is None:
                                continue
                            
                            np.save(embedding_path, embedding)
                        except Exception as e:
                            logging.error(f"Error getting embedding for {doc_name}: {str(e)}")
                            continue
                            
                    embeddings[doc_name] = embedding
                    
                # Group documents using same clustering logic as SummaryGrouper
                n_clusters = max(2, len(embeddings) // config.group_size)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                
                doc_names = list(embeddings.keys())
                X = np.array([embeddings[name] for name in doc_names])
                labels = kmeans.fit_predict(X)
                
                # Create groups with consistent naming
                for doc_name, label in zip(doc_names, labels):
                    # Extract priority and full name from doc_name
                    priority = int(doc_name[:6])
                    name = doc_name[7:].split(f"_{context.level}_")[0] if context.level > 1 else doc_name[7:].split("_SUMMARY")[0]
                    
                    group_name = f"group_{label}"
                    if group_name not in results:
                        results[group_name] = {}
                    results[group_name][doc_name] = documents[doc_name]
                    
                # Save group contents with same structure as before
                for group_name, group_docs in results.items():
                    group_dir = config.get_output_path(
                        DocumentInfo(0, group_name, ""),
                        context.level,
                        context.base_dir
                    ).parent
                    group_dir.mkdir(parents=True, exist_ok=True)
                    
                    for doc_name, doc in group_docs.items():
                        doc_path = group_dir / f"{doc_name}{config.file_extension}"
                        doc_path.write_text(doc.content, encoding='utf-8')
            
            return results
            
        # For GROUP_SUMMARY type, create summaries for existing groups
        elif config.name == "GROUP_SUMMARY":
            group_config = self.configs.get("GROUP")
            if not group_config:
                logging.error("GROUP config required but not found")
                return results
                
            # Get the group directory
            group_base_dir = context.base_dir / f"level_{context.level}" / group_config.output_directory
            if not group_base_dir.exists():
                logging.error(f"No group directory found at {group_base_dir}")
                return results
                
            # Process each group directory
            for group_dir in group_base_dir.iterdir():
                if not group_dir.is_dir():
                    continue
                    
                # Combine all documents in the group
                group_docs = []
                total_priority = 0
                for doc_path in group_dir.glob(f"*{group_config.file_extension}"):
                    content = doc_path.read_text(encoding='utf-8')
                    # Extract priority from filename
                    priority = int(doc_path.stem[:6])
                    total_priority += priority
                    group_docs.append(content)
                
                if not group_docs:
                    continue
                    
                # Create combined content for the group
                combined_content = "\n\n---\n\n".join(group_docs)
                
                # Create a DocumentInfo for the group
                group_doc = DocumentInfo(
                    priority=total_priority // len(group_docs),  # Average priority
                    name=group_dir.name,
                    content=combined_content
                )
                
                # Generate summary for the group
                try:
                    summary = await self._default_llm_processor(
                        group_doc, context, config
                    )
                    
                    # Save the summary
                    output_path = config.get_output_path(
                        group_doc, context.level, context.base_dir
                    )
                    self._save_metadata(summary, output_path)
                    
                    # Store in results
                    results[f"{group_doc.priority:06d}_{group_doc.name}"] = summary
                    
                except Exception as e:
                    logging.error(f"Error processing group summary for {group_dir.name}: {str(e)}")
                    continue
        
        return results
    
    async def _split_and_process(
        self,
        doc: DocumentInfo,
        content: str,
        context: ProcessingContext,
        config: MetadataConfig
    ) -> List[str]:
        """Split content and process each chunk with better size handling."""
        # Use smaller chunks for better reliability
        max_chunk_size = self.max_chars // 2
        
        # Split into paragraphs first
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            if current_size + para_size > max_chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
                
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        results = []
        for i, chunk in enumerate(chunks, 1):
            try:
                result = await self._default_llm_processor(
                    DocumentInfo(doc.priority, f"{doc.name}_part{i}", chunk),
                    context,
                    config
                )
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing chunk {i}: {str(e)}")
                continue
                
        return results
    
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