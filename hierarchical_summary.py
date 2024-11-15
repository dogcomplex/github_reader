import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import Dict, List
from summarize import Summarizer
from group_summaries import SummaryGrouper

class HierarchicalSummarizer:
    def __init__(self, output_dir: Path = Path("hierarchical_summaries")):
        self.output_dir = output_dir
        self.summarizer = Summarizer()
        self.grouper = SummaryGrouper()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def process_level(
        self,
        session: aiohttp.ClientSession,
        summaries: Dict[str, str],
        level: int
    ) -> Dict[str, str]:
        """Process one level of the hierarchy."""
        logging.info(f"Processing level {level} with {len(summaries)} summaries")
        
        # Create level directory
        level_dir = self.output_dir / f"level_{level}"
        level_dir.mkdir(exist_ok=True)
        
        # If few enough summaries, create final summary
        if len(summaries) <= self.grouper.max_summaries_per_group:
            final_summary = await self.summarizer.create_group_summary(
                session,
                list(summaries.values()),
                "final",
                level
            )
            
            with open(level_dir / "final_SUMMARY.md", 'w') as f:
                f.write(final_summary)
            
            return {"final": final_summary}
        
        # Group summaries
        groups = self.grouper.group_summaries(summaries)
        
        # Create new summaries for each group
        new_summaries = await self.summarizer.process_groups(session, groups, level)
        
        # Save level summaries
        for name, summary in new_summaries.items():
            with open(level_dir / f"{name}_SUMMARY.md", 'w') as f:
                f.write(summary)
        
        return new_summaries

    async def run(self, initial_summaries: Dict[str, str]):
        """Run the complete hierarchical summarization process."""
        async with aiohttp.ClientSession() as session:
            current_level = 1
            current_summaries = initial_summaries
            
            while True:
                try:
                    current_summaries = await self.process_level(
                        session,
                        current_summaries,
                        current_level
                    )
                    
                    if len(current_summaries) == 1:
                        logging.info("Created final summary!")
                        break
                        
                    current_level += 1
                    
                except Exception as e:
                    logging.error(f"Error at level {current_level}: {e}")
                    break

async def main():
    logging.basicConfig(level=logging.INFO)
    
    # Load initial summaries
    initial_summaries = {}
    clusters_dir = Path("grouped_summaries/clusters")
    for summary_file in clusters_dir.glob("**/*_SUMMARY.md"):
        with open(summary_file, 'r') as f:
            initial_summaries[summary_file.stem] = f.read()
    
    summarizer = HierarchicalSummarizer()
    await summarizer.run(initial_summaries)

if __name__ == "__main__":
    asyncio.run(main())