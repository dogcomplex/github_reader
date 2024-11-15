import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import Dict, List
from summarize import Summarizer
from group_summaries import SummaryGrouper, DocumentInfo
from tqdm.asyncio import tqdm_asyncio

class HierarchicalSummarizer:
    def __init__(
        self,
        base_dir: Path = Path("summaries"),
        readmes_dir: Path = Path("readmes/202411")
    ):
        self.base_dir = base_dir
        self.readmes_dir = readmes_dir
        self.summarizer = Summarizer()
        self.grouper = SummaryGrouper()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_level_dirs(self, level: int) -> tuple[Path, Path, Path]:
        """Get directories for summaries, embeddings, and groups at given level."""
        level_dir = self.base_dir / f"level_{level}"
        summaries_dir = level_dir / "summaries"
        embeddings_dir = level_dir / "embeddings"
        groups_dir = level_dir / "groups"
        
        # Create all directories
        for dir_path in [summaries_dir, embeddings_dir, groups_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return summaries_dir, embeddings_dir, groups_dir

    def load_documents(self, directory: Path) -> Dict[str, DocumentInfo]:
        """Load documents from directory maintaining priority information."""
        documents = {}
        for file_path in directory.glob("*_README.md"):
            priority, name = self.grouper.parse_filename(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            documents[file_path.stem] = DocumentInfo(
                priority=priority,
                name=name,
                content=content
            )
        return documents

    async def create_initial_summaries(
        self,
        session: aiohttp.ClientSession,
        documents: Dict[str, DocumentInfo]
    ) -> Dict[str, DocumentInfo]:
        """Create first-level summaries for individual documents."""
        logging.info(f"Creating initial summaries for {len(documents)} documents")
        summaries_dir, _, _ = self.get_level_dirs(1)
        
        new_documents = {}
        async for filename, doc in tqdm_asyncio(documents.items()):
            summary_path = summaries_dir / self.grouper.get_summary_path(doc, 1)
            
            # Skip if summary already exists
            if summary_path.exists():
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = f.read()
                new_documents[summary_path.stem] = DocumentInfo(
                    priority=doc.priority,
                    name=doc.name,
                    content=summary
                )
                continue
                
            try:
                summary = await self.summarizer.create_summary(
                    session,
                    doc.content,
                    level=1
                )
                
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                new_documents[summary_path.stem] = DocumentInfo(
                    priority=doc.priority,
                    name=doc.name,
                    content=summary
                )
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
                continue
                
        return new_documents

    async def process_level(
        self,
        session: aiohttp.ClientSession,
        documents: Dict[str, DocumentInfo],
        level: int
    ) -> Dict[str, DocumentInfo]:
        """Process one level of the hierarchy."""
        logging.info(f"Processing level {level} with {len(documents)} documents")
        
        summaries_dir, embeddings_dir, groups_dir = self.get_level_dirs(level)
        
        if len(documents) <= self.grouper.max_summaries_per_group:
            total_priority = sum(doc.priority for doc in documents.values())
            final_path = summaries_dir / f"{total_priority:06d}_final_{level}_SUMMARY.md"
            
            # Skip if final summary exists
            if final_path.exists():
                with open(final_path, 'r', encoding='utf-8') as f:
                    final_summary = f.read()
                return {"final": DocumentInfo(total_priority, "final", final_summary)}
            
            final_summary = await self.summarizer.create_group_summary(
                session,
                [doc.content for doc in documents.values()],
                "final",
                level
            )
            
            with open(final_path, 'w', encoding='utf-8') as f:
                f.write(final_summary)
            
            return {"final": DocumentInfo(total_priority, "final", final_summary)}
        
        # Group documents
        groups = self.grouper.group_summaries(documents, level, embeddings_dir)
        
        # Process each group
        new_documents = {}
        for group_name, group_docs in groups.items():
            group_dir = groups_dir / group_name
            group_dir.mkdir(exist_ok=True)
            
            total_priority = sum(doc.priority for doc in group_docs.values())
            summary_path = group_dir / f"{total_priority:06d}_{group_name}_SUMMARY.md"
            
            # Skip if group summary exists
            if summary_path.exists():
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = f.read()
                new_documents[group_name] = DocumentInfo(
                    priority=total_priority,
                    name=group_name,
                    content=summary
                )
                continue
            
            summary = await self.summarizer.create_group_summary(
                session,
                [doc.content for doc in group_docs.values()],
                group_name,
                level
            )
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            new_documents[group_name] = DocumentInfo(
                priority=total_priority,
                name=group_name,
                content=summary
            )
        
        return new_documents

    async def run(self):
        """Run the complete hierarchical summarization process."""
        # Load initial documents
        initial_documents = self.load_documents(self.readmes_dir)
        
        async with aiohttp.ClientSession() as session:
            try:
                # First level: Create individual summaries
                current_documents = await self.create_initial_summaries(
                    session,
                    initial_documents
                )
                
                # Subsequent levels: Group and summarize
                current_level = 2
                while True:
                    current_documents = await self.process_level(
                        session,
                        current_documents,
                        current_level
                    )
                    
                    if len(current_documents) == 1:
                        logging.info("Created final summary!")
                        break
                        
                    current_level += 1
                    
            except Exception as e:
                logging.error(f"Error during processing: {e}")

async def main():
    logging.basicConfig(level=logging.INFO)
    summarizer = HierarchicalSummarizer()
    await summarizer.run()

if __name__ == "__main__":
    asyncio.run(main())