import asyncio
import aiohttp
import json
from typing import List, Dict
import logging

class Summarizer:
    def __init__(self, llm_url: str = "http://localhost:1234/v1/chat/completions"):
        self.llm_url = llm_url
        
    async def create_summary(
        self,
        session: aiohttp.ClientSession,
        content: str,
        level: int = 0
    ) -> str:
        """Create a single summary from content."""
        system_message = {
            "role": "system",
            "content": (
                f"You are creating a level {level} summary that combines related content. "
                "Create a concise, single-paragraph summary that captures key themes and relationships in plain English."
            )
        }
        
        prompt = {
            "messages": [
                system_message,
                {"role": "user", "content": f"Please summarize this content:\n\n{content}"}
            ],
            "model": "local-model"
        }

        try:
            async with session.post(self.llm_url, json=prompt) as response:
                if response.status != 200:
                    raise Exception(f"API error {response.status}: {await response.text()}")
                result = await response.json()
                return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            logging.error(f"Summary creation failed: {str(e)}")
            raise

    async def create_group_summary(
        self,
        session: aiohttp.ClientSession,
        summaries: List[str],
        group_name: str,
        level: int
    ) -> str:
        """Create a summary for a group of related summaries."""
        combined_text = f"Group {group_name} contains these related summaries:\n\n"
        combined_text += "\n\n---\n\n".join(summaries)
        return await self.create_summary(session, combined_text, level)

    async def process_groups(
        self,
        session: aiohttp.ClientSession,
        groups: Dict[str, List[str]],
        level: int
    ) -> Dict[str, str]:
        """Process all groups into new summaries."""
        new_summaries = {}
        for group_name, texts in groups.items():
            summary = await self.create_group_summary(session, texts, group_name, level)
            new_summaries[group_name] = summary
        return new_summaries
