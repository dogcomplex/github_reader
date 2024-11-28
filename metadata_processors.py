import logging
import json
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
from typing import Optional, Dict, Any
import numpy as np
from sklearn.cluster import KMeans
from models import DocumentInfo, MetadataConfig
from processing_context import ProcessingContext
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path
import re
from math import ceil

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def _call_llm_api(
    doc: DocumentInfo,
    context: ProcessingContext,
    config: MetadataConfig
) -> str:
    """Helper function for LLM API calls."""
    async with context.session.post(
        context.llm_url,
        json={
            "messages": [
                {
                    "role": "system",
                    "content": config.system_prompt_template
                },
                {
                    "role": "user",
                    "content": f"Process this content:\n\n{doc.content}"
                }
            ],
            "model": "local-model"
        }
    ) as response:
        if response.status != 200:
            raise Exception(f"API error {response.status}: {await response.text()}")
        result = await response.json()
        return result['choices'][0]['message']['content'].strip()

async def process_summary(
    doc: DocumentInfo,
    context: ProcessingContext,
    config: MetadataConfig
) -> str:
    """Process document summary using LLM."""
    # Check for existing split summary
    split_path = config.get_output_path(doc, context.level, context.base_dir).parent.parent / "split_summaries" / f"{doc.name}_SPLIT_SUMMARY.md"
    
    if split_path.exists():
        # Use split summary instead of original content
        content = split_path.read_text()
    else:
        content = doc.content
        
    return await _call_llm_api(
        DocumentInfo(doc.priority, doc.name, content),
        context,
        config
    )

async def process_tags(
    doc: DocumentInfo,
    context: ProcessingContext,
    config: MetadataConfig
) -> str:
    """Process document tags using LLM."""
    # Similar pattern for tags
    split_path = config.get_output_path(doc, context.level, context.base_dir).parent.parent / "split_summaries" / f"{doc.name}_SPLIT_SUMMARY.md"
    
    if split_path.exists():
        content = split_path.read_text()
    else:
        content = doc.content
        
    return await _call_llm_api(
        DocumentInfo(doc.priority, doc.name, content),
        context,
        config
    )

async def process_report(
    doc: DocumentInfo,
    context: ProcessingContext,
    config: MetadataConfig
) -> str:
    """Process detailed document report using LLM."""
    return await _call_llm_api(doc, context, config)

async def process_embedding(
    doc: DocumentInfo,
    context: ProcessingContext,
    config: MetadataConfig
) -> Optional[np.ndarray]:
    """Generate document embedding."""
    async with context.session.post(
        context.embedding_url,
        json={
            "input": doc.content,
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

async def process_groups(
    documents: Dict[str, DocumentInfo],
    context: ProcessingContext, 
    config: MetadataConfig
) -> Dict[str, list]:
    """Process groups in batches to limit memory usage."""
    groups = {}
    batch_size = 100  # Process embeddings in batches
    
    # Process embeddings in batches
    doc_items = list(documents.items())
    for i in range(0, len(doc_items), batch_size):
        batch = dict(doc_items[i:i + batch_size])
        
        # Load batch embeddings
        batch_embeddings = {}
        for doc_name, doc in batch.items():
            embedding_path = config.get_output_path(
                doc, context.level, context.base_dir
            ).parent.parent / "embeddings" / f"{doc.name}_EMBEDDING.npy"
            
            if embedding_path.exists():
                batch_embeddings[doc_name] = np.load(embedding_path)
        
        if batch_embeddings:
            # Process batch
            X = np.stack([batch_embeddings[name] for name in batch_embeddings])
            n_clusters = max(2, min(len(batch_embeddings) // config.group_size, len(batch_embeddings) - 1))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Update groups
            for doc_name, label in zip(batch_embeddings.keys(), labels):
                group_name = f"group_{label}"
                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append(doc_name)
                
            # Save batch results
            groups_path = context.base_dir / f"level_{context.level}" / "groups" / "groups.json"
            groups_path.parent.mkdir(parents=True, exist_ok=True)
            groups_path.write_text(json.dumps(groups))
            
    return groups

async def process_group_summary(
    documents: Dict[str, DocumentInfo],
    context: ProcessingContext,
    config: MetadataConfig
) -> Dict[str, str]:
    """Generate summaries for document groups."""
    # Load groups
    groups_path = context.base_dir / f"level_{context.level}" / "groups" / "groups.json"
    if not groups_path.exists():
        return {}
        
    groups = json.loads(groups_path.read_text())
    summaries = {}
    
    for group_name, doc_names in groups.items():
        # Combine document contents
        group_docs = [documents[name] for name in doc_names if name in documents]
        combined_content = "\n\n---\n\n".join(doc.content for doc in group_docs)
        
        # Generate group summary
        summary = await process_summary(
            DocumentInfo(0, group_name, combined_content),
            context,
            config
        )
        summaries[group_name] = summary
        
    return summaries

async def detect_language(
    doc: DocumentInfo,
    context: ProcessingContext,
    config: MetadataConfig
) -> str:
    """Detect document language."""
    try:
        # Remove markdown/HTML for better detection
        text = BeautifulSoup(doc.content, "html.parser").get_text()
        lang = detect(text)
        return json.dumps({
            "language": lang,
            "is_english": lang == "en"
        })
    except LangDetectException:
        return json.dumps({
            "language": "unknown",
            "is_english": False
        })

async def translate_if_needed(
    doc: DocumentInfo,
    context: ProcessingContext,
    config: MetadataConfig
) -> Optional[str]:
    """Translate document if not in English."""
    # Check language from cache if available
    lang_path = config.get_output_path(doc, context.level, context.base_dir).parent.parent / "language" / f"{doc.name}_LANGUAGE.json"
    
    try:
        if lang_path.exists():
            lang_info = json.loads(lang_path.read_text())
            if lang_info.get("is_english", False):
                return "ALREADY_ENGLISH"
    except Exception as e:
        logging.warning(f"Error reading language cache: {e}")
    
    # If no cache or error, translate using LLM
    return await _call_llm_api(doc, context, config)

async def process_split_summary(
    doc: DocumentInfo,
    context: ProcessingContext,
    config: MetadataConfig
) -> Optional[str]:
    """Split and summarize large documents."""
    # Check if splitting needed
    if len(doc.content) <= config.max_chars:
        return None  # Skip if document is small enough
        
    logging.info(
        f"Content too large ({len(doc.content)} chars) for processing '{doc.name}', splitting"
    )
    
    # Split content into chunks
    chunks = []
    current_chunk = []
    current_size = 0
    
    # Simple paragraph-based splitting
    paragraphs = doc.content.split('\n\n')
    
    for para in paragraphs:
        if current_size + len(para) > config.max_chars:
            if current_chunk:  # Process current chunk
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
        current_chunk.append(para)
        current_size += len(para)
        
    if current_chunk:  # Add final chunk
        chunks.append('\n\n'.join(current_chunk))
    
    # Process each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        chunk_doc = DocumentInfo(
            priority=doc.priority,
            name=f"{doc.name}_chunk_{i+1}",
            content=chunk
        )
        summary = await _call_llm_api(chunk_doc, context, config)
        if summary:
            summaries.append(summary)
    
    if not summaries:
        return None
        
    # Combine summaries
    combined = "\n\n---\n\n".join(summaries)
    
    # Create final summary of summaries if needed
    if len(combined) > config.max_chars:
        return await _call_llm_api(
            DocumentInfo(doc.priority, doc.name, combined),
            context,
            config
        )
    
    return combined 

async def process_chunks(
    doc: DocumentInfo,
    context: ProcessingContext,
    config: MetadataConfig
) -> Dict[str, str]:
    """Split large documents into chunks with overlap."""
    content = doc.content
    max_chars = config.max_chars
    overlap = config.chunk_overlap

    # If content is small enough, return as single chunk
    if len(content) <= max_chars:
        output_path = config.get_output_path(doc, context.level, context.base_dir)
        output_path = output_path.parent / f"{doc.name}_1of1_CHUNK.md"
        if context.overwrite or not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding='utf-8')
        return {f"{doc.name}_1of1_CHUNK": content}

    # Process chunks one at a time to minimize memory usage
    chunk_results = {}
    start = 0
    chunk_num = 1
    total_chunks = ceil((len(content) - overlap) / (max_chars - overlap))
    
    while start < len(content):
        # Find a good break point near max_chars
        end = start + max_chars
        if end < len(content):
            # Try to break at paragraph or sentence
            break_point = _find_break_point(content[start:end + overlap])
            end = start + break_point
        else:
            end = len(content)
            
        # Process and write this chunk
        chunk = content[start:end]
        chunk_name = f"{doc.name}_{chunk_num}of{total_chunks}_CHUNK"
        output_path = config.get_output_path(doc, context.level, context.base_dir)
        output_path = output_path.parent / f"{chunk_name}.md"
        
        if context.overwrite or not output_path.exists():
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(chunk, encoding='utf-8')
                chunk_results[chunk_name] = chunk
            except UnicodeEncodeError as e:
                logging.error(f"Encoding error in {doc.name} chunk {chunk_num}: {str(e)}")
        
        # Move to next chunk
        start = end - overlap
        chunk_num += 1

    return chunk_results

def _find_break_point(text: str) -> int:
    """Find a natural break point in text (paragraph or sentence)."""
    # Try to break at paragraph first
    paragraphs = re.split(r'\n\s*\n', text)
    if len(paragraphs) > 1:
        return len(paragraphs[0]) + 2  # Include the newlines
        
    # Try to break at sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 1:
        return len(sentences[0]) + 1  # Include the space
        
    # If no good break point, break at word boundary
    words = text.split()
    if len(words) > 1:
        return len(' '.join(words[:-1])) + 1
        
    # Last resort: break at max_chars
    return len(text) 