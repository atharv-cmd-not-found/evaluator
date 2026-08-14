import json
import re
import urllib.request
from pathlib import Path
from typing import List, Dict, Any


def split_into_sentences(text: str) -> List[str]:
    """Splits multi-sentence student answers into individual sentences for fine-grained evaluation."""
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
    return sentences if sentences else [text]


def load_json_from_url(url: str) -> Dict[str, Any]:
    """Fetches and parses a remote JSON file directly from GitHub."""
    req = urllib.request.Request(
        url, 
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_all_student_files_from_repo(repo_owner: str, repo_name: str, folder_path: str = "input") -> List[Dict[str, str]]:
    """
    Uses GitHub REST API to list all student files in the 'input/' folder on GitHub,
    allowing full automation across all student submissions.
    """
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req) as response:
        files = json.loads(response.read().decode("utf-8"))
        
    student_files = []
    for item in files:
        if item["type"] == "file" and item["name"].endswith(".json"):
            student_files.append({
                "name": item["name"],
                "download_url": item["download_url"]
            })
            
    return student_files


def save_json_file(file_path: Path, data_str: str) -> None:
    """Saves stringified JSON output locally to the output directory."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(data_str)