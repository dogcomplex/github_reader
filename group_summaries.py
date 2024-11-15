import numpy as np
from sklearn.cluster import KMeans
import requests
import logging
from typing import Dict, List, Tuple, NamedTuple
import re
from pathlib import Path

class DocumentInfo(NamedTuple):
    priority: int
    name: str
    content: str

class SummaryGrouper:
    def __init__(
        self,
        api_url: str = 'http://localhost:1234/v1/embeddings',
        model_name: str = 'text-embedding-bge-m3',
        max_summaries_per_group: int = 10
    ):
        self.api_url = api_url
        self.model_name = model_name
        self.max_summaries_per_group = max_summaries_per_group

    def parse_filename(self, filepath: Path) -> Tuple[int, str]:
        """Extract priority and name from filename."""
        filename = filepath.stem  # removes .md extension
        
        # Split only on first underscore to get priority
        parts = filename.split('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid filename format: {filename}")
        
        priority_str = parts[0]
        rest = parts[1]
        
        # Split on last underscore to separate name from README
        name_parts = rest.rsplit('_', 1)
        if len(name_parts) != 2 or name_parts[1] != 'README':
            raise ValueError(f"Invalid filename format: {filename}")
        
        name = name_parts[0]
        
        try:
            priority = int(priority_str)
        except ValueError:
            raise ValueError(f"Invalid priority format in filename: {filename}")
        
        return priority, name

    def generate_group_name(
        self,
        texts: List[str],
        level: int,
        group_idx: int
    ) -> str:
        """Generate a semantic group name based on common themes."""
        # TODO: Could use LLM to generate a descriptive name based on common themes
        # For now, use a simple numbered group
        return f"group_{group_idx}"

    def get_embedding_path(self, doc_info: DocumentInfo, level: int) -> Path:
        """Get path for embedding file."""
        if level == 1:
            return Path(f"{doc_info.priority:06d}_{doc_info.name}_EMBEDDING.npy")
        return Path(f"{doc_info.priority:06d}_{doc_info.name}_{level}_EMBEDDING.npy")

    def get_summary_path(self, doc_info: DocumentInfo, level: int) -> Path:
        """Get path for summary file."""
        if level == 1:
            return Path(f"{doc_info.priority:06d}_{doc_info.name}_SUMMARY.md")
        return Path(f"{doc_info.priority:06d}_{doc_info.name}_{level}_SUMMARY.md")

    def get_metadata_path(self, doc_info: DocumentInfo, level: int, metadata_type: str) -> Path:
        """Get path for metadata file."""
        if level == 1:
            return Path(f"{doc_info.priority:06d}_{doc_info.name}_{metadata_type}.md")
        return Path(f"{doc_info.priority:06d}_{doc_info.name}_{level}_{metadata_type}.md")

    def compute_embeddings(
        self,
        documents: Dict[str, DocumentInfo],
        level: int,
        embeddings_dir: Path
    ) -> np.ndarray:
        """Compute or load cached embeddings for documents."""
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        embeddings = []
        
        for doc in documents.values():
            embedding_path = embeddings_dir / self.get_embedding_path(doc, level)
            
            if embedding_path.exists():
                embedding = np.load(embedding_path)
            else:
                embedding = self.get_embedding(doc.content)
                np.save(embedding_path, embedding)
            
            embeddings.append(embedding)
            
        return np.array(embeddings)

    def group_summaries(
        self,
        documents: Dict[str, DocumentInfo],
        level: int,
        embeddings_dir: Path
    ) -> Dict[str, Dict[str, DocumentInfo]]:
        """Group summaries based on semantic similarity."""
        embeddings = self.compute_embeddings(documents, level, embeddings_dir)
        
        # Determine number of clusters
        n_clusters = max(2, len(embeddings) // self.max_summaries_per_group)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Group by cluster
        groups: Dict[str, Dict[str, DocumentInfo]] = {}
        for (filename, doc), label in zip(documents.items(), labels):
            group_docs = groups.setdefault(label, {})
            group_docs[filename] = doc
            
        # Generate final groups with proper naming
        final_groups: Dict[str, Dict[str, DocumentInfo]] = {}
        for label, group_docs in groups.items():
            total_priority = sum(doc.priority for doc in group_docs.values())
            group_name = self.generate_group_name(
                [doc.content for doc in group_docs.values()],
                level,
                label
            )
            group_key = f"{total_priority:06d}_{group_name}_{level}_GROUP"
            final_groups[group_key] = group_docs
            
        return final_groups

    def clean_text(self, text: str) -> str:
        """Clean text for embedding."""
        text = re.sub(r'#+ ', '', text)
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        return ' '.join(text.split())

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text."""
        cleaned_text = self.clean_text(text)
        
        for attempt in range(3):
            try:
                response = requests.post(
                    self.api_url,
                    headers={'Content-Type': 'application/json'},
                    json={'input': cleaned_text, 'model': self.model_name},
                    timeout=10
                )
                response.raise_for_status()
                return np.array(response.json()['data'][0]['embedding'])
            except Exception as e:
                if attempt == 2:
                    raise
                continue
