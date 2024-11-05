import os
import asyncio
import aiohttp
import pathlib
from typing import List, Tuple
import json

# Configuration
LOCAL_LLM_URL = "http://localhost:1234/v1/chat/completions"
BATCH_SIZE = 1  # Number of concurrent requests
MAX_TOKENS = 10000  # Safe limit below model's 60801 context length

async def create_summary(session: aiohttp.ClientSession, content: str) -> str:
    """Request a summary from the local LLM."""
    # Rough approximation: 1 token ≈ 4 characters for English text
    max_chars = MAX_TOKENS * 4
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n[Content truncated due to length...]"
    
    prompt = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that creates concise, single-paragraph summaries of GitHub repositories based on their README files. Focus only on the repository's main purpose and key features."
            },
            {
                "role": "user",
                "content": f"Please summarize this README in one paragraph (with no preamble or postscript):\n\n{content}"
            }
        ],
        "model": "local-model"  # Adjust based on your local setup
    }

    try:
        async with session.post(LOCAL_LLM_URL, json=prompt) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API returned status {response.status}: {error_text}")
            
            result = await response.json()
            
            # Validate response structure
            if not isinstance(result, dict):
                raise Exception(f"Unexpected response format: {result}")
            
            if 'choices' not in result:
                raise Exception(f"No 'choices' in response: {result}")
            
            if not result['choices'] or 'message' not in result['choices'][0]:
                raise Exception(f"Invalid choices format: {result}")
            
            return result['choices'][0]['message']['content'].strip()
    except aiohttp.ClientError as e:
        raise Exception(f"Network error: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON response: {str(e)}")

async def process_batch(
    session: aiohttp.ClientSession,
    batch: List[Tuple[pathlib.Path, pathlib.Path]]
) -> None:
    """Process a batch of README files concurrently."""
    async def process_single_file(readme_path: pathlib.Path, summary_path: pathlib.Path):
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            summary = await create_summary(session, content)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            print(f"Created summary for {readme_path.name}")
        except Exception as e:
            print(f"Error processing {readme_path.name}: {e}")

    tasks = [
        process_single_file(readme_path, summary_path) 
        for readme_path, summary_path in batch
    ]
    await asyncio.gather(*tasks)

async def main():
    # Create summaries directory with same structure as readmes
    readmes_dir = pathlib.Path("readmes")
    summaries_dir = pathlib.Path("summaries")
    
    # Get all README files (excluding MISSING ones)
    readme_files = []
    for month_dir in readmes_dir.iterdir():
        if month_dir.is_dir():
            summary_month_dir = summaries_dir / month_dir.name
            summary_month_dir.mkdir(parents=True, exist_ok=True)
            
            for readme_path in month_dir.glob("*_README.md"):
                summary_path = summary_month_dir / readme_path.name.replace("_README.md", "_SUMMARY.md")
                if not summary_path.exists():
                    # Extract total_activity from filename (first 6 digits)
                    total_activity = int(readme_path.name[:6])
                    readme_files.append((readme_path, summary_path, total_activity))

    # Sort by total_activity in descending order
    readme_files.sort(key=lambda x: x[2], reverse=True)
    # Remove total_activity from tuples after sorting
    readme_files = [(r, s) for r, s, _ in readme_files]

    print(f"Found {len(readme_files)} README files to summarize")
    
    # Process in batches
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(readme_files), BATCH_SIZE):
            batch = readme_files[i:i + BATCH_SIZE]
            print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(readme_files)-1)//BATCH_SIZE + 1}")
            await process_batch(session, batch)

if __name__ == "__main__":
    asyncio.run(main())
