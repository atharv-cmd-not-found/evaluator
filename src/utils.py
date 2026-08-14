import json
import re
import urllib.request
from pathlib import Path
from typing import List, Dict, Any


def split_into_sentences(text: str) -> List[str]:
    """Splits multi-sentence student answers into individual sentences."""
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
    return sentences if sentences else [text]


def save_json_file(file_path: Path, data: Any) -> None:
    """Saves data as a formatted JSON file locally, creating parent folders if missing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        if isinstance(data, str):
            f.write(data)
        else:
            json.dump(data, f, indent=2)


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Loads a local JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path.resolve()}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def download_and_save_remote_json(url: str, save_path: Path) -> Dict[str, Any]:
    """Fetches a JSON file from GitHub and saves a copy locally into the target folder."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode("utf-8"))
    
    # Save the fetched JSON directly to the specified folder
    save_json_file(save_path, data)
    return data


def fetch_all_student_files_from_repo(repo_owner: str, repo_name: str, folder_path: str = "input") -> List[Dict[str, str]]:
    """Uses GitHub REST API to query all student JSON files available in the input folder."""
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