import os
from dotenv import load_dotenv
import requests
from google.cloud import bigquery
from google.oauth2 import service_account
from base64 import b64decode
import datetime
import pathlib

# Load environment variables
load_dotenv()

# Set up BigQuery and GitHub credentials
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
BIGQUERY_CREDENTIALS = service_account.Credentials.from_service_account_file(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
)

# Initialize BigQuery client
client = bigquery.Client(credentials=BIGQUERY_CREDENTIALS, project=GOOGLE_CLOUD_PROJECT)

# BigQuery query to get all repos with recent changes in the last month
# This query gets all repos with recent changes in the last month
# and orders them by total activity (stars + forks)
# limited to 1000 (5000 is api hourly limit)
QUERY = """
    SELECT repo.name AS repo_name, 
       COUNTIF(type = 'WatchEvent') AS star_count,
       COUNTIF(type = 'ForkEvent') AS fork_count,
       (COUNTIF(type = 'WatchEvent') + COUNTIF(type = 'ForkEvent')) AS total_activity
    FROM `githubarchive.month.202410`
    GROUP BY repo.name
    HAVING total_activity > 1
    ORDER BY total_activity DESC
    LIMIT 1000
"""

def get_recent_repositories():
    query_job = client.query(QUERY)  # Run the query
    results = query_job.result()  # Wait for the results
    # Modified to return both repo name and total activity
    repos = [(row.repo_name, row.total_activity) for row in results]
    return repos

def get_output_directory():
    current_date = datetime.datetime.now()
    year_month = current_date.strftime("%Y%m")
    path = pathlib.Path(f"readmes/{year_month}")
    path.mkdir(parents=True, exist_ok=True)
    return path

def download_readme(repo_full_name, total_activity, output_dir):
    # List of possible README paths to try
    readme_paths = [
        "README.md",
        "readme.md",
        "Readme.md",
        "README.markdown",
        "README.rst",
        "README",
        "readme",
        "docs/README.md",
        ".github/README.md",
    ]
    
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    safe_repo_name = repo_full_name.replace('/', '_')
    
    # Try each possible README path
    for path in readme_paths:
        url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            try:
                readme_content = b64decode(response.json()['content']).decode('utf-8')
                output_file = output_dir / f"{total_activity:06d}_{safe_repo_name}_README.md"
                
                with open(output_file, "w", encoding='utf-8') as file:
                    file.write(readme_content)
                print(f"Downloaded {path} for {repo_full_name}")
                return
            except Exception as e:
                print(f"Error processing {path} for {repo_full_name}: {e}")
                continue
    
    # If no README was found and downloaded, create a MISSING file
    missing_file = output_dir / f"{total_activity:06d}_{safe_repo_name}_MISSING.md"
    with open(missing_file, "w", encoding='utf-8') as file:
        file.write(f"==={repo_full_name} README is missing or not found====")
    print(f"No README found for {repo_full_name}, created MISSING file")

def get_existing_readmes(output_dir):
    """Returns a set of repo names that already have downloaded READMEs or marked as missing."""
    existing = set()
    for file in output_dir.glob('*_*_*.*'):  # Match any readme or missing file
        try:
            # Skip the activity number prefix and _{README|MISSING}.md suffix
            repo_name = '_'.join(file.stem.split('_')[1:-1])
            repo_name = repo_name.replace('_', '/', 1)  # Replace only first underscore
            existing.add(repo_name)
        except Exception as e:
            print(f"Warning: Couldn't parse filename {file}: {e}")
    return existing

def main():
    print("Fetching repositories with recent activity...")
    recent_repos = get_recent_repositories()
    print(f"Found {len(recent_repos)} repositories with recent activity")
    
    output_dir = get_output_directory()
    existing_readmes = get_existing_readmes(output_dir)
    
    # Filter out repos that already have READMEs
    repos_to_download = [(name, activity) for name, activity in recent_repos 
                        if name not in existing_readmes]
    
    print(f"Found {len(existing_readmes)} existing READMEs")
    print(f"Need to download {len(repos_to_download)} new READMEs")
    print(f"Estimated MB to download (at 5kb/file avg): {len(repos_to_download) * 5 / 1000} MB")
    
    print("Downloading README.md files...")
    for repo_name, total_activity in repos_to_download:
        download_readme(repo_name, total_activity, output_dir)

if __name__ == "__main__":
    main()
