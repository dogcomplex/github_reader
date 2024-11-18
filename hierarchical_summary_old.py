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
        readmes_dir: Path = Path("readmes/202411"),
        min_level_for_reports: int = 2,
        overwrite: bool = False
    ):
        self.base_dir = base_dir
        self.readmes_dir = readmes_dir
        self.min_level_for_reports = min_level_for_reports
        self.overwrite = overwrite
        self.summarizer = Summarizer()
        self.grouper = SummaryGrouper()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_level_dirs(self, level: int) -> tuple[Path, Path, Path, Path, Path]:
        """Get directories for summaries, embeddings, groups, tags, and reports at given level."""
        level_dir = self.base_dir / f"level_{level}"
        summaries_dir = level_dir / "summaries"
        embeddings_dir = level_dir / "embeddings"
        groups_dir = level_dir / "groups"
        tags_dir = level_dir / "tags"
        reports_dir = level_dir / "reports"
        
        # Create all directories
        for dir_path in [summaries_dir, embeddings_dir, groups_dir, tags_dir, reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return summaries_dir, embeddings_dir, groups_dir, tags_dir, reports_dir

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
        """Create first-level summaries and tags for individual documents."""
        logging.info(f"Creating initial summaries and tags for {len(documents)} documents")
        summaries_dir, _, _, tags_dir, _ = self.get_level_dirs(1)
        
        new_documents = {}
        async for filename, doc in tqdm_asyncio(documents.items()):
            # Process summary
            summary_path = summaries_dir / self.grouper.get_summary_path(doc, 1)
            if not self.overwrite and summary_path.exists():
                summary = summary_path.read_text(encoding='utf-8')
            else:
                try:
                    # Try direct summarization first
                    summary = await self.summarizer.create_summary(session, doc.content, level=1)
                    summary_path.write_text(summary, encoding='utf-8')
                except Exception as e:
                    if "context length" in str(e).lower():
                        logging.info(f"Splitting {filename} due to length...")
                        split_summaries = await self.summarizer.split_and_summarize(
                            session, doc.content, doc, level=1
                        )
                        
                        for split_name, split_summary in split_summaries:
                            split_path = summaries_dir / f"{split_name}_SUMMARY.md"
                            if not self.overwrite and split_path.exists():
                                summary = split_path.read_text(encoding='utf-8')
                            else:
                                split_path.write_text(split_summary, encoding='utf-8')
                                summary = split_summary
                            new_documents[split_path.stem] = DocumentInfo(
                                priority=doc.priority,
                                name=split_name,
                                content=summary
                            )
                        continue
                    else:
                        logging.error(f"Error processing {filename}: {str(e)}")
                        continue

            new_documents[summary_path.stem] = DocumentInfo(
                priority=doc.priority,
                name=doc.name,
                content=summary
            )
                    
            # Process tags (for original document only)
            tags_path = tags_dir / self.grouper.get_metadata_path(doc, 1, "TAGS")
            if not tags_path.exists() or self.overwrite:
                try:
                    tags = await self.summarizer.create_tags(session, doc.content, level=1)
                    tags_path.write_text(tags, encoding='utf-8')
                except Exception as e:
                    logging.error(f"Error creating tags for {filename}: {str(e)}")
                
        return new_documents

    async def process_level(
        self,
        session: aiohttp.ClientSession,
        documents: Dict[str, DocumentInfo],
        level: int
    ) -> Dict[str, DocumentInfo]:
        """Process one level of the hierarchy."""
        logging.info(f"Processing level {level} with {len(documents)} documents")
        
        summaries_dir, embeddings_dir, groups_dir, tags_dir, reports_dir = self.get_level_dirs(level)
        
        # Process metadata for each document independently
        for doc_name, doc in documents.items():
            # Generate tags if missing or overwrite is True
            tags_path = tags_dir / self.grouper.get_metadata_path(doc, level, "TAGS")
            if not tags_path.exists() or self.overwrite:
                try:
                    tags = await self.summarizer.create_tags(session, doc.content, level)
                    tags_path.write_text(tags, encoding='utf-8')
                except Exception as e:
                    logging.error(f"Error creating tags for {doc_name}: {str(e)}")
            
            # Generate reports for level 2+ if missing or overwrite is True
            if level >= self.min_level_for_reports:
                report_path = reports_dir / self.grouper.get_metadata_path(doc, level, "REPORT")
                if not report_path.exists() or self.overwrite:
                    try:
                        report = await self.summarizer.create_report(session, doc.content, level)
                        report_path.write_text(report, encoding='utf-8')
                    except Exception as e:
                        logging.error(f"Error creating report for {doc_name}: {str(e)}")

        # Process summaries
        if len(documents) <= self.grouper.max_summaries_per_group:
            total_priority = sum(doc.priority for doc in documents.values())
            final_path = summaries_dir / f"{total_priority:06d}_final_{level}_SUMMARY.md"
            
            if not self.overwrite and final_path.exists():
                final_summary = final_path.read_text(encoding='utf-8')
            else:
                final_summary = await self.summarizer.create_group_summary(
                    session,
                    [doc.content for doc in documents.values()],
                    "final",
                    level
                )
                final_path.write_text(final_summary, encoding='utf-8')
            
            return {"final": DocumentInfo(total_priority, "final", final_summary)}
        
        # Group documents and process each group independently
        groups = self.grouper.group_summaries(documents, level, embeddings_dir)
        sorted_groups = sorted(
            groups.items(),
            key=lambda x: sum(doc.priority for doc in x[1].values()),
            reverse=True
        )
        
        new_documents = {}
        for group_name, group_docs in sorted_groups:
            # Create group directory
            group_dir = groups_dir / group_name
            group_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy previous level's summaries
            for doc_name, doc in group_docs.items():
                source_path = summaries_dir.parent.parent / f"level_{level-1}" / "summaries" / f"{doc_name}.md"
                if source_path.exists():
                    target_path = group_dir / f"{doc_name}.md"
                    content = source_path.read_text(encoding='utf-8')
                    target_path.write_text(content, encoding='utf-8')
            
            # Process group summary
            total_priority = sum(doc.priority for doc in group_docs.values())
            summary_path = summaries_dir / f"{total_priority:06d}_{group_name}_SUMMARY.md"
            
            if not self.overwrite and summary_path.exists():
                summary = summary_path.read_text(encoding='utf-8')
            else:
                summary = await self.summarizer.create_group_summary(
                    session,
                    [doc.content for doc in group_docs.values()],
                    group_name,
                    level
                )
                summary_path.write_text(summary, encoding='utf-8')
            
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
    summarizer = HierarchicalSummarizer(overwrite=False)  # Set overwrite=False by default
    await summarizer.run()

if __name__ == "__main__":
    asyncio.run(main())