from typing import Dict
from models import MetadataConfig, MetadataScope
from metadata_processors import detect_language, translate_if_needed

DEFAULT_METADATA_CONFIGS: Dict[str, MetadataConfig] = {
    "SUMMARY": MetadataConfig(
        name="SUMMARY",
        scope=MetadataScope.SINGLE,
        system_prompt_template=(
            "You are creating a level {level} summary. "
            "Create a concise, single-paragraph summary that captures key themes "
            "and relationships in plain English."
        ),
        output_directory="summaries"
    ),
    
    
    
    "TAGS": MetadataConfig(
        name="TAGS",
        scope=MetadataScope.SINGLE,
        system_prompt_template=(
            "You are summarizing the contents of this document into a list of keyword tags of length 5-20 main tags."
            "Return only a comma-separated list of relevant English keywords that capture "
            "the key concepts, technologies, and themes present in the content. "
            "Do not include explanations or other text.  Translate concepts to English if necessary."
        ),
        output_directory="tags"
    ),
    
    "REPORT": MetadataConfig(
        name="REPORT",
        scope=MetadataScope.SINGLE,
        min_level=2,  # Only generate reports from level 2+
        system_prompt_template=(
            "You are creating a level {level} detailed report. "
            "Provide comprehensive high-level notes about the content in 1-5 paragraphs. "
            "Focus on key findings, relationships, and important details."
        ),
        output_directory="reports"
    ),
    
    "EMBEDDING": MetadataConfig(
        name="EMBEDDING",
        scope=MetadataScope.SINGLE,
        file_extension=".npy",
        requires_embedding=True,
        output_directory="embeddings"
    ),
    
    "GROUP": MetadataConfig(
        name="GROUP",
        scope=MetadataScope.GROUP,
        group_size=10,
        requires_embedding=True,
        output_directory="groups"
    ),
    
    "GROUP_SUMMARY": MetadataConfig(
        name="GROUP_SUMMARY",
        scope=MetadataScope.GROUP,
        system_prompt_template=(
            "You are creating a general summary of multiple documents. "
            "Create a concise summary that captures key themes and relationships "
            "across all documents in this group in plain English. Focus on "
            "common themes and relationships between the documents."
        ),
        output_directory="summaries",
        requires_embedding=False  # Don't need embeddings for summarization
    ),
    
    "LANGUAGE": MetadataConfig(
        name="LANGUAGE",
        scope=MetadataScope.SINGLE,
        file_extension=".json",  # JSON is good for structured data
        processor=detect_language,  # We'll implement this
        output_directory="language"
    ),
    
    "TRANSLATION": MetadataConfig(
        name="TRANSLATION", 
        scope=MetadataScope.SINGLE,
        system_prompt_template=(
            "You are a translator. Translate the following text to English, "
            "preserving any markdown formatting. If the text is already in English, "
            "respond with 'ALREADY_ENGLISH'. Maintain any code blocks, links, or "
            "other markdown elements in their original form."
        ),
        output_directory="translations",
        # Only process docs that need translation
        processor=translate_if_needed  # We'll implement this
    ),
} 