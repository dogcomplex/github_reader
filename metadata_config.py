from typing import Dict, List
from models import MetadataConfig, MetadataScope
from metadata_processors import (
    detect_language, 
    translate_if_needed,
    process_summary,
    process_tags,
    process_report,
    process_embedding,
    process_groups,
    process_group_summary,
    process_split_summary,
    process_chunks
)

DEFAULT_METADATA_CONFIGS: Dict[str, MetadataConfig] = {
    "CHUNK": MetadataConfig(
        name="CHUNK",
        scope=MetadataScope.SINGLE,
        output_directory="chunks",
        processor=process_chunks,
        max_chars=30000,
        chunk_size=30000,
        chunk_overlap=1000,
        enable=True,
        override=False,
        dependencies=[],  # This should run first
        merge_strategy="concat"
    ),
    "SUMMARY": MetadataConfig(
        name="SUMMARY",
        scope=MetadataScope.SINGLE,
        system_prompt_template=(
            "Create a concise, single-paragraph summary that captures key themes "
            "and relationships in plain English."
        ),
        output_directory="summaries",
        processor=process_summary,
        enable=True,
        override=False,
        dependencies=["CHUNK"]
    ),
    
    "TAGS": MetadataConfig(
        name="TAGS",
        scope=MetadataScope.SINGLE,
        system_prompt_template=(
            "Return only a comma-separated list of relevant English keywords that capture "
            "the key concepts, technologies, and themes present in the content."
        ),
        output_directory="tags",
        processor=process_tags,
        enable=True,
        override=False,
        dependencies=["CHUNK"]
    ),
    
    "REPORT": MetadataConfig(
        name="REPORT",
        scope=MetadataScope.SINGLE,
        min_level=2,
        system_prompt_template=(
            "Provide comprehensive high-level notes about the content in 1-5 paragraphs. "
            "Focus on key findings, relationships, and important details."
        ),
        output_directory="reports",
        processor=process_report,
        enable=True,
        override=False,
        dependencies=["SUMMARY"]
    ),
    
    "EMBEDDING": MetadataConfig(
        name="EMBEDDING",
        scope=MetadataScope.SINGLE,
        file_extension=".npy",
        output_directory="embeddings",
        processor=process_embedding,
        enable=True,
        override=False,
        dependencies=[]
    ),
    
    "GROUP": MetadataConfig(
        name="GROUP",
        scope=MetadataScope.GROUP,
        group_size=10,
        output_directory="groups",
        processor=process_groups,
        enable=True,
        override=False,
        dependencies=["EMBEDDING"]
    ),
    
    "GROUP_SUMMARY": MetadataConfig(
        name="GROUP_SUMMARY",
        scope=MetadataScope.GROUP,
        system_prompt_template=(
            "Create a concise summary that captures key themes and relationships "
            "across all documents in this group in plain English."
        ),
        output_directory="summaries",
        processor=process_group_summary,
        enable=True,
        override=False,
        dependencies=["GROUP"]
    ),
    
    "LANGUAGE": MetadataConfig(
        name="LANGUAGE",
        scope=MetadataScope.SINGLE,
        file_extension=".json",
        processor=detect_language,
        output_directory="language",
        enable=True,
        override=False,
        dependencies=[]
    ),
    
    "TRANSLATION": MetadataConfig(
        name="TRANSLATION", 
        scope=MetadataScope.SINGLE,
        system_prompt_template=(
            "Translate the following text to English, preserving any markdown formatting."
        ),
        output_directory="translations",
        processor=translate_if_needed,
        enable=True,
        override=False,
        dependencies=["LANGUAGE"]
    ),
    "SPLIT_SUMMARY": MetadataConfig(
        name="SPLIT_SUMMARY",
        scope=MetadataScope.SINGLE,
        system_prompt_template=(
            "Create a concise summary of this content section, preserving key details "
            "and relationships. The summary will be combined with others for a final summary."
        ),
        output_directory="split_summaries",
        processor=process_split_summary,
        max_chars=30000,  # Default max chars for context
        enable=True,
        override=False,
        dependencies=["CHUNK"]
    ),
} 