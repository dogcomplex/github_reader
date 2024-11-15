import os
import json
import requests
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import shutil
import logging
from tqdm.auto import tqdm
import re
import time

# Directories
SUMMARIES_DIR = './summaries/202411'
EMBEDDINGS_DIR = './summary_embeddings'
OUTPUT_DIR = './grouped_summaries'

# Ensure directories exist
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# API settings
API_URL = 'http://localhost:1234/v1/embeddings'
MODEL_NAME = 'text-embedding-bge-m3'  # Replace with your model name
LENGTH_LIMIT = 8000

# Add to top with other settings
USE_SYMLINKS = True  # Toggle between symlinks (True) and copies (False)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def clean_text_for_embedding(text):
    """Clean and prepare text for embedding"""
    # Remove markdown headers
    text = re.sub(r'#+ ', '', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    # Truncate if needed
    return text[:LENGTH_LIMIT]

def get_embedding(text):
    """Get embedding with better error handling and text preparation"""
    try:
        # Clean and prepare text
        cleaned_text = clean_text_for_embedding(text)
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer your-api-key'
        }
        data = {
            'input': cleaned_text,
            'model': MODEL_NAME
        }
        
        # Add retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(API_URL, headers=headers, json=data, timeout=10)
                response.raise_for_status()
                return response.json()['data'][0]['embedding']
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)  # Wait before retry
                
    except Exception as e:
        logging.error(f"Failed to get embedding: {str(e)}")
        return None

def load_summaries():
    logging.info(f"Loading summaries from {SUMMARIES_DIR}")
    summaries = []
    filenames = []
    
    # Get all summary files and sort in reverse order (highest importance first)
    all_files = [f for f in os.listdir(SUMMARIES_DIR) if f.endswith('_SUMMARY.md')]
    all_files.sort(reverse=True)  # Will sort by importance score since it's prefix
    
    total_files = len(all_files)
    logging.info(f"Found {total_files} summary files")
    
    for filename in tqdm(all_files, total=total_files, desc="Loading summaries"):
        filepath = os.path.join(SUMMARIES_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            if len(content) > LENGTH_LIMIT:
                char_count = len(content)
                word_count = len(content.split())
                logging.warning(f"Skipping {filename}: Summary too long ({char_count} chars, ~{word_count} words)")
                continue
            summaries.append(content)
            filenames.append(filename)
    
    logging.info(f"Loaded {len(summaries)} valid summaries")
    return summaries, filenames

def get_embedding_filename(summary_filename):
    """Convert SUMMARY filename to EMBEDDING filename"""
    return summary_filename.replace('_SUMMARY.md', '_EMBEDDING.npy')

def compute_embeddings(summaries, filenames):
    embeddings = []
    processed_filenames = []
    
    for summary, filename in tqdm(zip(summaries, filenames), total=len(summaries), desc="Computing embeddings"):
        embedding_filename = get_embedding_filename(filename)
        embedding_path = os.path.join(EMBEDDINGS_DIR, embedding_filename)
        
        try:
            if os.path.exists(embedding_path):
                embedding = np.load(embedding_path)
            else:
                embedding = get_embedding(summary)
                if embedding is None:
                    continue
                np.save(embedding_path, embedding)
            
            embeddings.append(embedding)
            processed_filenames.append(filename)
            
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            continue
    
    if not embeddings:
        raise ValueError("No valid embeddings could be computed!")
        
    return np.array(embeddings), processed_filenames

def cluster_embeddings(embeddings, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels

def group_summaries_by_labels(filenames, labels):
    grouped = {}
    for filename, label in zip(filenames, labels):
        grouped.setdefault(label, []).append(filename)
    return grouped

def save_grouped_summaries(grouped, method='clusters', use_symlinks=USE_SYMLINKS):
    """
    Save grouped summaries with method-specific subfolder, checking for existing groups
    """
    base_dir = os.path.join(OUTPUT_DIR, method)
    os.makedirs(base_dir, exist_ok=True)
    
    # Check if groups already exist
    existing_groups = set()
    if os.path.exists(base_dir):
        existing_groups = {d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))}
    
    total_files = sum(len(files) for files in grouped.values())
    link_type = "symlinks" if use_symlinks else "copies"
    
    if existing_groups:
        logging.info(f"Found {len(existing_groups)} existing {method} groups")
        should_continue = input("Do you want to recreate these groups? (y/n): ").lower()
        if should_continue != 'y':
            logging.info("Skipping group creation - using existing groups")
            return
        
    logging.info(f"Saving {total_files} files as {link_type} in {base_dir}")
    
    # Clear existing groups if we're recreating them
    if existing_groups:
        for group in existing_groups:
            shutil.rmtree(os.path.join(base_dir, group))
    
    with tqdm(total=total_files, desc=f"Saving {method} groups") as pbar:
        for label, files in grouped.items():
            group_dir = os.path.join(base_dir, f'group_{label}')
            os.makedirs(group_dir, exist_ok=True)
            for filename in files:
                src = os.path.join(SUMMARIES_DIR, filename)
                dst = os.path.join(group_dir, filename)
                if os.path.exists(dst):
                    os.remove(dst)
                
                if use_symlinks:
                    os.symlink(src, dst)
                else:
                    shutil.copy2(src, dst)
                pbar.update(1)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_clusters(embeddings, labels):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=set(labels))
    plt.title('Clusters of GitHub README Summaries')
    plt.show()


from scipy.spatial.distance import cosine

def compute_similarity_matrix(embeddings):
    num_embeddings = embeddings.shape[0]
    similarity_matrix = np.zeros((num_embeddings, num_embeddings))
    
    total_comparisons = (num_embeddings * (num_embeddings - 1)) // 2
    with tqdm(total=total_comparisons, desc="Computing similarities") as pbar:
        for i in range(num_embeddings):
            for j in range(i+1, num_embeddings):
                similarity = 1 - cosine(embeddings[i], embeddings[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
                pbar.update(1)
    
    return similarity_matrix

def group_by_similarity(similarity_matrix, filenames, threshold=0.8):
    """Group summaries by similarity threshold"""
    logging.info(f"Grouping by similarity (threshold={threshold})")
    groups = {}
    processed = set()
    
    for i in tqdm(range(len(filenames)), desc="Finding similar groups"):
        if i in processed:
            continue
            
        group = {i}
        for j in range(len(filenames)):
            if j != i and similarity_matrix[i][j] >= threshold:
                group.add(j)
                
        if len(group) > 1:
            group_id = len(groups)
            groups[group_id] = [filenames[idx] for idx in group]
            processed.update(group)
    
    logging.info(f"Created {len(groups)} similarity groups")
    return groups

# In main():

def main():
    logging.info("Starting summary grouping process")
    
    summaries, filenames = load_summaries()
    embeddings, filenames = compute_embeddings(summaries, filenames)
    
    if len(embeddings) == 0:
        logging.error("No embeddings to process. Exiting.")
        return
        
    # Cluster-based grouping
    logging.info("Performing cluster-based grouping")
    labels = cluster_embeddings(embeddings, n_clusters=10)
    grouped = group_summaries_by_labels(filenames, labels)
    save_grouped_summaries(grouped, method='clusters', use_symlinks=USE_SYMLINKS)
    
    # Visualization
    logging.info("Generating cluster visualization")
    visualize_clusters(embeddings, labels)
    
    # Similarity-based grouping
    logging.info("Performing similarity-based grouping")
    similarity_matrix = compute_similarity_matrix(embeddings)
    similarity_groups = group_by_similarity(similarity_matrix, filenames, threshold=0.8)
    save_grouped_summaries(similarity_groups, method='similarity', use_symlinks=USE_SYMLINKS)
    
    logging.info("Grouping complete! Results saved in 'grouped_summaries' directory")
    logging.info(f"Created {len(grouped)} cluster groups and {len(similarity_groups)} similarity groups")



if __name__ == '__main__':
    main()
