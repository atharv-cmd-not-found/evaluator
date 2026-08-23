import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Union


def split_into_sentences(text: str) -> List[str]:
    """Splits multi-sentence student answers into individual sentences."""
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
    return sentences if sentences else [text]


def save_json_file(file_path: Union[Path, str], data: Any) -> None:
    """Saves data as a formatted JSON file locally, creating parent folders if missing."""
    target_path = Path(file_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        if isinstance(data, str):
            f.write(data)
        else:
            json.dump(data, f, indent=2)


def load_json_file(file_path: Union[Path, str]) -> Dict[str, Any]:
    """Loads a local JSON file."""
    target_path = Path(file_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Missing required file: {target_path.resolve()}")
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_all_local_student_files(
    folder_path: Union[Path, str] = "input",
    exclude_files: List[str] = None
) -> List[Path]:
    """
    Scans a local directory and returns a list of Paths for all student JSON files,
    excluding static config files like 'rubric.json' and 'model_answers.json'.
    """
    if exclude_files is None:
        exclude_files = ["rubric.json", "model_answers.json"]

    directory = Path(folder_path)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {directory.resolve()}")

    student_files = [
        file_path for file_path in directory.glob("*.json")
        if file_path.name not in exclude_files
    ]
    return sorted(student_files)


def load_all_local_students(
    folder_path: Union[Path, str] = "input",
    exclude_files: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Scans and loads all local student JSON files from the specified folder.
    """
    files = get_all_local_student_files(folder_path, exclude_files)
    students = []
    for file_path in files:
        students.append(load_json_file(file_path))
    return students


# ---------------------------------------------------------
# Remote Repository Utility Functions (Optional / Fallback)
# ---------------------------------------------------------

def download_and_save_remote_json(url: str, save_path: Union[Path, str]) -> Dict[str, Any]:
    """Fetches a JSON file from GitHub and saves a copy locally into the target folder."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode("utf-8"))
    
    save_json_file(save_path, data)
    return data


def fetch_all_student_files_from_repo(
    repo_owner: str, 
    repo_name: str, 
    folder_path: str = "input"
) -> List[Dict[str, str]]:
    """Uses GitHub REST API to query all student JSON files available in the remote input folder."""
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