import argparse
from pathlib import Path
import sys
from typing import List
import pandas as pd

from src.engine import CrossEncoderRubricEvaluator, LLMRubricEvaluator
from src.utils import load_json_file, save_json_file

# -------------------------------------------------------------
# Comprehensive Benchmark Catalog
# -------------------------------------------------------------
CROSS_ENCODER_CONFIGS = [
    # 1. NLI Models
    {
        "name": "cross-encoder/nli-deberta-v3-large",
        "type": "nli",
        "label": "DeBERTa-v3-Large (NLI)",
    },
    {
        "name": "cross-encoder/nli-deberta-v3-base",
        "type": "nli",
        "label": "DeBERTa-v3-Base (NLI)",
    },
    {
        "name": "cross-encoder/nli-roberta-base",
        "type": "nli",
        "label": "RoBERTa-Base (NLI)",
    },
    {
        "name": "cross-encoder/nli-distilroberta-base",
        "type": "nli",
        "label": "DistilRoBERTa (NLI)",
    },
    # 2. STS Models
    {
        "name": "cross-encoder/stsb-roberta-large",
        "type": "sts",
        "label": "RoBERTa-Large (STS)",
    },
    {
        "name": "cross-encoder/stsb-roberta-base",
        "type": "sts",
        "label": "RoBERTa-Base (STS)",
    },
    {
        "name": "cross-encoder/stsb-distilroberta-base",
        "type": "sts",
        "label": "DistilRoBERTa (STS)",
    },
    # 3. Rerankers
    {
        "name": "BAAI/bge-reranker-base",
        "type": "ranking",
        "label": "BGE-Reranker-Base",
    },
    {
        "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "type": "ranking",
        "label": "MiniLM-L6 (MS-MARCO)",
    },
]

# Open-Source LLMs to benchmark (Ensure models are pulled in Ollama)
LLM_CONFIGS = [
    "qwen2.5:7b",
    "mistral:7b",
    "phi3.5:latest",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Comprehensive Multi-Model Academic Answer Evaluator"
    )
    parser.add_argument(
        "--rubric",
        "-r",
        type=str,
        default="input/rubric.json",
        help="Path to rubric JSON file",
    )
    parser.add_argument(
        "--model_answers",
        "-m",
        type=str,
        default="input/model_answers.json",
        help="Path to model answers JSON file",
    )
    parser.add_argument(
        "--students",
        "-s",
        nargs="+",
        help="Path(s) to student JSON file(s). Example: -s input/student1.json input/student2.json",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="output",
        help="Directory to save evaluation reports",
    )
    return parser.parse_args()


def get_user_inputs():
    """Prompts the user interactively if arguments were not passed via CLI."""
    args = parse_arguments()

    rubric_path = Path(args.rubric)
    model_ans_path = Path(args.model_answers)
    output_dir = Path(args.output_dir)

    # Interactive check for Rubric
    if not rubric_path.exists():
        print(f"\nDefault rubric file not found at: '{rubric_path}'")
        user_r = input("Enter path to rubric.json: ").strip()
        rubric_path = Path(user_r)

    # Interactive check for Model Answers
    if not model_ans_path.exists():
        print(f"\nDefault model answers not found at: '{model_ans_path}'")
        user_m = input("Enter path to model_answers.json: ").strip()
        model_ans_path = Path(user_m)

    # Check for Student Files
    student_paths: List[Path] = []
    if args.students:
        student_paths = [Path(p) for p in args.students]
    else:
        print("\n" + "=" * 60)
        print("STUDENT FILE SELECTION")
        print("=" * 60)
        user_s = input(
            "Enter path(s) to student JSON file(s) (comma or space separated) [leave blank to scan 'input/']: "
        ).strip()
        if user_s:
            tokens = [
                t.strip()
                for t in user_s.replace(",", " ").split()
                if t.strip()
            ]
            student_paths = [Path(t) for t in tokens]
        else:
            # Fallback: scan input directory
            input_dir = Path("input")
            if input_dir.exists():
                student_paths = [
                    f
                    for f in input_dir.glob("*.json")
                    if f.name not in ["rubric.json", "model_answers.json"]
                ]

    return rubric_path, model_ans_path, student_paths, output_dir


def main():
    rubric_file, model_ans_file, student_files, output_dir = get_user_inputs()

    # Validate file presence
    if not rubric_file.exists():
        print(f"❌ Error: Rubric file not found at '{rubric_file.resolve()}'")
        sys.exit(1)
    if not model_ans_file.exists():
        print(
            f"❌ Error: Model answers file not found at '{model_ans_file.resolve()}'"
        )
        sys.exit(1)
    if not student_files:
        print(f"❌ Error: No student submission files specified or found.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("🚀 INITIALIZING COMPARATIVE EVALUATION PIPELINE")
    print("=" * 80)
    print(f"• Rubric File:        {rubric_file.resolve()}")
    print(f"• Model Answers:      {model_ans_file.resolve()}")
    print(f"• Target Students:    {[f.name for f in student_files]}")
    print(f"• Output Directory:   {output_dir.resolve()}")

    # 1. Ingest Base Ground Truths
    rubric_data = load_json_file(rubric_file)
    model_answers = load_json_file(model_ans_file)

    # 2. Instantiate All Evaluator Models
    evaluators = []

    print("\n--- Loading Cross-Encoder Architectures ---")
    for cfg in CROSS_ENCODER_CONFIGS:
        try:
            evaluators.append(
                CrossEncoderRubricEvaluator(
                    model_name=cfg["name"], model_type=cfg["type"]
                )
            )
        except Exception as e:
            print(f"⚠️ Could not load {cfg['name']}: {e}")

    print("\n--- Initializing Open-Source LLM Evaluators ---")
    for llm_name in LLM_CONFIGS:
        try:
            evaluators.append(LLMRubricEvaluator(model_name=llm_name))
        except Exception as e:
            print(f"⚠️ Skipping {llm_name} (Ollama unavailable or offline): {e}")

    if not evaluators:
        print("❌ No evaluator models could be loaded. Exiting.")
        sys.exit(1)

    summary_records = []
    benchmark_records = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Process Student Submissions Across All Evaluators
    print("\n" + "=" * 80)
    print("EXECUTING BATCH COMPARATIVE EVALUATION")
    print("=" * 80)

    for s_path in student_files:
        if not s_path.exists():
            print(f"⚠️ Student file does not exist, skipping: {s_path}")
            continue

        student_data = load_json_file(s_path)
        student_id = student_data.get("student", s_path.stem)
        print(f"\n==================================================")
        print(f"📘 Student: {student_id} (File: {s_path.name})")
        print(f"==================================================")

        for eval_idx, evaluator in enumerate(evaluators, start=1):
            print(
                f"[{eval_idx}/{len(evaluators)}] Evaluating via [{evaluator.evaluator_type}] {evaluator.model_name}..."
            )

            try:
                report = evaluator.evaluate_student(
                    student_data=student_data,
                    model_answers_data=model_answers,
                    rubric_data=rubric_data,
                )

                # Export individual detailed JSON report
                sanitized_name = (
                    evaluator.model_name.replace("/", "_").replace(":", "_")
                )
                report_path = (
                    output_dir
                    / f"{student_id}_{sanitized_name}_evaluated.json"
                )
                save_json_file(report_path, report.model_dump(mode="json"))

                # Record student-level metrics
                summary_records.append(
                    {
                        "student_id": report.student_id,
                        "evaluator_model": report.evaluator_model,
                        "evaluator_type": report.evaluator_type,
                        "marks_awarded": report.total_marks_awarded,
                        "max_total_marks": report.max_total_marks,
                        "percentage": f"{report.percentage:.2f}%",
                    }
                )

                # Record criteria-level metrics
                for q_id, q_eval in report.question_evaluations.items():
                    for crit in q_eval.criteria_breakdown:
                        benchmark_records.append(
                            {
                                "student_id": report.student_id,
                                "evaluator_model": report.evaluator_model,
                                "evaluator_type": report.evaluator_type,
                                "question_id": q_id,
                                "criterion_id": crit.criterion_id,
                                "criterion_type": crit.type,
                                "max_marks": crit.max_marks,
                                "marks_awarded": crit.marks_awarded,
                                "status": crit.status,
                                "satisfied": crit.satisfied,
                                "diagnostics": crit.diagnostics,
                                "entailment_score": crit.entailment_score,
                                "contradiction_score": crit.contradiction_score,
                                "similarity_score": crit.semantic_similarity,
                            }
                        )

            except Exception as e:
                print(f"❌ Error during evaluation with {evaluator.model_name}: {e}")

    # 4. Generate Comparative Tables
    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        df_benchmark = pd.DataFrame(benchmark_records)

        summary_csv = output_dir / "comparative_students_summary.csv"
        benchmark_csv = output_dir / "comparative_criteria_benchmark.csv"

        df_summary.to_csv(summary_csv, index=False)
        df_benchmark.to_csv(benchmark_csv, index=False)

        print("\n" + "=" * 90)
        print("🎉 COMPARATIVE EVALUATION COMPLETE")
        print("=" * 90)
        print(f"• Student Summaries Exported:  {summary_csv.resolve()}")
        print(f"• Criteria Benchmarks Exported: {benchmark_csv.resolve()}")
        print("\n" + df_summary.to_string(index=False))


if __name__ == "__main__":
    main()