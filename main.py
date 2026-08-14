import sys
from pathlib import Path
from src.utils import (
    download_and_save_remote_json, 
    fetch_all_student_files_from_repo, 
    save_json_file
)
from src.schemas import StudentEvaluationReport
from src.engine import NLIRubricEvaluator

# Repository Settings
REPO_OWNER = "amolbedade08"
REPO_NAME = "ai-question-mapping"
BRANCH = "main"

GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}"
RUBRIC_URL = f"{GITHUB_RAW_BASE}/model_answers/rubric.json"
MODEL_ANSWER_URL = f"{GITHUB_RAW_BASE}/model_answers/model_answers.json"


def run_automated_pipeline():
    base_path = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    input_dir = base_path / "input"
    output_dir = base_path / "output"

    print("🚀 Starting Automated GitHub Evaluation Pipeline...")
    
    # 1. Fetch & Store Rubric and Model Answers inside input/
    try:
        print(f"Downloading rubric.json to 'input/'...")
        rubrics = download_and_save_remote_json(RUBRIC_URL, input_dir / "rubric.json")

        print(f"Downloading model_answers.json to 'input/'...")
        model_answers = download_and_save_remote_json(MODEL_ANSWER_URL, input_dir / "model_answers.json")
    except Exception as e:
        print(f"❌ Error fetching configuration files from GitHub: {e}")
        sys.exit(1)

    # 2. Fetch list of student JSON files from GitHub 'input/' directory
    try:
        print(f"Listing student submissions from GitHub repository...")
        student_files = fetch_all_student_files_from_repo(REPO_OWNER, REPO_NAME, folder_path="input")
        print(f"Found {len(student_files)} student file(s) on GitHub.")
    except Exception as e:
        print(f"❌ Failed to query GitHub API for student files: {e}")
        sys.exit(1)

    # 3. Initialize Evaluator
    evaluator = NLIRubricEvaluator()

    # 4. Download and process each student file
    for student_file_info in student_files:
        file_name = student_file_info["name"]
        download_url = student_file_info["download_url"]
        local_student_path = input_dir / file_name

        print(f"\n--------------------------------------------------")
        print(f"Downloading and saving student file to 'input/{file_name}'...")
        student_input = download_and_save_remote_json(download_url, local_student_path)

        student_id = student_input.get("student", Path(file_name).stem)
        student_answers = student_input.get("answers", {})

        question_evaluations = {}
        grand_total_awarded = 0.0
        grand_total_max = 0.0

        for q_id, student_ans_text in student_answers.items():
            if q_id in rubrics and q_id in model_answers:
                q_eval = evaluator.evaluate_question(
                    question_id=q_id,
                    question_data=model_answers[q_id],
                    rubric_data=rubrics[q_id],
                    student_answer=student_ans_text
                )
                question_evaluations[q_id] = q_eval
                grand_total_awarded += q_eval.total_marks_awarded
                grand_total_max += q_eval.max_total_marks

        grand_percentage = round((grand_total_awarded / grand_total_max) * 100, 2) if grand_total_max > 0 else 0.0

        report = StudentEvaluationReport(
            student_id=student_id,
            total_marks_awarded=round(grand_total_awarded, 2),
            max_total_marks=round(grand_total_max, 2),
            percentage=grand_percentage,
            question_evaluations=question_evaluations
        )

        # 5. Save output report in output/
        out_file_path = output_dir / f"{student_id}_evaluated.json"
        save_json_file(out_file_path, report.model_dump_json(indent=2))

        print(f"✅ Evaluation complete for: {student_id}")
        print(f"Overall Score: {report.total_marks_awarded} / {report.max_total_marks} ({report.percentage}%)")
        print(f"Evaluation report saved to: {out_file_path.resolve()}")

    print("\n🎉 Automated Pipeline Finished Successfully!")


if __name__ == "__main__":
    run_automated_pipeline()