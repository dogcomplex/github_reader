import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import Dict, List, Callable, Optional

from models import DocumentInfo, MetadataConfig, MetadataScope
from metadata_config import DEFAULT_METADATA_CONFIGS
from metadata_processor import MetadataProcessor
from processing_context import ProcessingContext
from tqdm.asyncio import tqdm

class HierarchicalSummarizer:
    def __init__(
        self,
        base_dir: Path = Path("summaries"),
        readmes_dir: Path = Path("readmes/202411"),
        metadata_configs: Optional[Dict[str, MetadataConfig]] = None,
        custom_processors: Optional[Dict[str, Callable]] = None,
        overwrite: bool = False
    ):
        self.base_dir = base_dir
        self.readmes_dir = readmes_dir
        self.overwrite = overwrite
        self.metadata_processor = MetadataProcessor(
            configs=metadata_configs,
            custom_processors=custom_processors
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def run(self):
        """Main processing loop."""
        async with aiohttp.ClientSession() as session:
            documents = self._load_initial_documents()
            level = 1

            with tqdm(desc="Processing levels") as level_pbar:
                while documents:
                    logging.info(f"Processing level {level}")
                    documents = await self.process_level(session, documents, level)
                    level += 1
                    level_pbar.update(1)

    def _load_initial_documents(self) -> Dict[str, DocumentInfo]:
        """Load initial README documents."""
        documents = {}
        for path in self.readmes_dir.glob("*_README.md"):
            try:
                # Extract priority (first 6 digits)
                priority = int(path.stem[:6])
                
                # Extract name (everything between priority_ and _README)
                name = path.stem[7:]  # Skip priority and underscore
                if name.endswith("_README"):
                    name = name[:-7]  # Remove _README suffix
                
                # Add UTF-8 encoding
                content = path.read_text(encoding='utf-8')
                documents[path.stem] = DocumentInfo(priority, name, content)
            except Exception as e:
                logging.error(f"Error loading {path}: {str(e)}")
        return documents

    def _prepare_next_level(
        self,
        summaries: Dict[str, str],
        level: int
    ) -> Dict[str, DocumentInfo]:
        """Prepare documents for next level from summaries."""
        if not summaries:
            return {}
            
        documents = {}
        for doc_name, summary in summaries.items():
            # Extract priority from start of filename (always 6 digits + underscore)
            priority = int(doc_name[:6])
            
            # Extract name by removing priority prefix and type suffix
            name_part = doc_name[7:]  # Skip priority and underscore
            
            # Remove level and type markers from end if present
            if f"_{level}_" in name_part:
                name = name_part.split(f"_{level}_")[0]
            else:
                # For level 1 or if no level marker
                name = name_part.split("_SUMMARY")[0]
            
            documents[doc_name] = DocumentInfo(
                priority=priority,
                name=name,
                content=summary
            )
        
        return documents

    async def process_level(
        self,
        session: aiohttp.ClientSession,
        documents: Dict[str, DocumentInfo],
        level: int
    ) -> Dict[str, DocumentInfo]:
        """Process one level of the hierarchy."""
        logging.info(f"Processing level {level} with {len(documents)} documents")
        
        context = ProcessingContext(
            session=session,
            level=level,
            base_dir=self.base_dir,
            llm_url="http://localhost:1234/v1/chat/completions",
            embedding_url="http://localhost:1234/v1/embeddings",
            overwrite=self.overwrite
        )
        
        # Process all metadata types for this level
        results = await self.metadata_processor.process_metadata(documents, context)
        
        # If we have a single document, we're done
        if len(documents) == 1 and level > 1:
            logging.info("Created final summary!")
            return {}
        
        # First check for group summaries
        if "GROUP" in results and "GROUP_SUMMARY" in results:
            return self._prepare_next_level(results["GROUP_SUMMARY"], level)
        # Fall back to individual summaries if no groups
        elif "SUMMARY" in results:
            return self._prepare_next_level(results["SUMMARY"], level)
        
        logging.warning("No summaries generated for next level")
        return {}

async def main():
    logging.basicConfig(level=logging.INFO)
    summarizer = HierarchicalSummarizer(overwrite=False)
    await summarizer.run()

if __name__ == "__main__":
    asyncio.run(main())