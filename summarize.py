import asyncio
import aiohttp
import json
from typing import List, Dict, Tuple
import logging
from group_summaries import DocumentInfo

class Summarizer:
    def __init__(
        self,
        llm_url: str = "http://localhost:1234/v1/chat/completions",
        context_size: int = 40000,  # Default context size in tokens
        chars_per_token: float = 3.5,  # Rough estimate of chars per token
        safety_margin: float = 0.8  # Use only 80% of max context for safety
    ):
        self.llm_url = llm_url
        self.context_size = context_size
        self.chars_per_token = chars_per_token
        self.max_chars = int(context_size * chars_per_token * safety_margin)
        
    def estimate_splits_needed(self, content: str) -> int:
        """Estimate number of splits needed based on content length."""
        content_length = len(content)
        # Add 20% buffer for safety
        return max(2, int((content_length / (self.max_chars * 0.8)) + 1))

    async def split_and_summarize(
        self,
        session: aiohttp.ClientSession,
        content: str,
        doc_info: DocumentInfo,
        level: int,
        num_splits: int = None
    ) -> List[Tuple[str, str]]:
        """Split content and create summaries, returning list of (name, summary)."""
        if num_splits is None:
            num_splits = self.estimate_splits_needed(content)
            logging.info(f"Estimated {num_splits} splits needed for {len(content)} chars")
        
        # Split content into roughly equal chunks
        words = content.split()
        chunk_size = len(words) // num_splits
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            try:
                # Pre-emptively split if chunk still too large
                if len(chunk) > self.max_chars:
                    logging.info(f"Chunk {i} still too large ({len(chunk)} chars), splitting further")
                    sub_splits = await self.split_and_summarize(
                        session,
                        chunk,
                        doc_info,
                        level,
                        num_splits + 1
                    )
                    summaries.extend(sub_splits)
                    continue
                
                summary = await self.create_summary(session, chunk, level)
                split_name = f"{doc_info.priority:06d}_{doc_info.name}_{i}of{num_splits}"
                summaries.append((split_name, summary))
            except Exception as e:
                if "context length" in str(e).lower() and num_splits < 8:
                    # If we still hit context length error, try splitting again
                    logging.warning(f"Context length error despite pre-split, trying {num_splits + 1} splits")
                    return await self.split_and_summarize(
                        session,
                        chunk,
                        doc_info,
                        level,
                        num_splits + 1
                    )
                raise
        
        return summaries

    async def create_summary(
        self,
        session: aiohttp.ClientSession,
        content: str,
        level: int = 0
    ) -> str:
        """Create a single summary from content."""
        # Pre-emptively split if content too large
        if len(content) > self.max_chars:
            logging.info(f"Content too large ({len(content)} chars), splitting before summary")
            doc_info = DocumentInfo(0, "temp", content)  # Temporary DocInfo for splitting
            splits = await self.split_and_summarize(session, content, doc_info, level)
            # Combine split summaries
            combined = "\n\n---\n\n".join(summary for _, summary in splits)
            return await self.create_summary(session, combined, level)

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

        return await self._make_api_call(session, prompt)

    async def create_report(
        self,
        session: aiohttp.ClientSession,
        content: str,
        level: int = 0
    ) -> str:
        """Create a detailed report from content."""
        # Pre-emptively split if content too large
        if len(content) > self.max_chars:
            logging.info(f"Content too large for report ({len(content)} chars), splitting")
            doc_info = DocumentInfo(0, "temp", content)
            splits = await self.split_and_summarize(session, content, doc_info, level)
            # Create reports for each split and combine
            reports = []
            for _, split_content in splits:
                report = await self.create_report(session, split_content, level)
                reports.append(report)
            return "\n\n---\n\n".join(reports)

        system_message = {
            "role": "system",
            "content": (
                f"You are creating a level {level} detailed report. "
                "Provide comprehensive high-level notes about the content in 1-5 paragraphs. "
                "Focus on key findings, relationships, and important details."
            )
        }
        
        prompt = {
            "messages": [
                system_message,
                {"role": "user", "content": f"Create a detailed report for this content:\n\n{content}"}
            ],
            "model": "local-model"
        }
        
        return await self._make_api_call(session, prompt)

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

    async def create_tags(
        self,
        session: aiohttp.ClientSession,
        content: str,
        level: int = 0
    ) -> str:
        """Create keyword tags from content."""
        # Pre-emptively split if content too large
        if len(content) > self.max_chars:
            logging.info(f"Content too large for tags ({len(content)} chars), splitting")
            doc_info = DocumentInfo(0, "temp", content)
            splits = await self.split_and_summarize(session, content, doc_info, level)
            # Get tags for each split and combine
            tag_lists = []
            for _, split_content in splits:
                tags = await self.create_tags(session, split_content, level)
                tag_lists.append(tags)
            # Combine and deduplicate tags
            all_tags = []
            for tags in tag_lists:
                all_tags.extend(tag.strip() for tag in tags.split(','))
            unique_tags = sorted(set(all_tags))
            return ', '.join(unique_tags)

        system_message = {
            "role": "system",
            "content": (
                f"You are creating level {level} keyword tags. "
                "Return only a comma-separated list of relevant English keywords that capture "
                "the key concepts, technologies, and themes present in the content. "
                "Do not include explanations or other text."
            )
        }
        
        prompt = {
            "messages": [
                system_message,
                {"role": "user", "content": f"Generate tags for this content:\n\n{content}"}
            ],
            "model": "local-model"
        }
        
        return await self._make_api_call(session, prompt)

    async def _make_api_call(
        self,
        session: aiohttp.ClientSession,
        prompt: dict
    ) -> str:
        """Make API call to LLM endpoint."""
        try:
            async with session.post(self.llm_url, json=prompt) as response:
                if response.status != 200:
                    raise Exception(f"API error {response.status}: {await response.text()}")
                result = await response.json()
                return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            logging.error(f"API call failed: {str(e)}")
            raise
