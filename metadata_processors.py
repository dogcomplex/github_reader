import logging
import json
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
from typing import Optional
from models import DocumentInfo, MetadataConfig
from processing_context import ProcessingContext

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
    result = await context.llm_processor(doc, context, config)
    
    # Don't save if already English
    if result == "ALREADY_ENGLISH":
        return None
        
    return result 