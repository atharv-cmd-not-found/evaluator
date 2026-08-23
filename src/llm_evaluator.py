import json
import time
from typing import Any, Dict, List
import requests

from src.schemas import (
    CriterionEvaluationResult,
    CriterionSchema,
    QuestionEvaluationResult,
)

SYSTEM_PROMPT = """You are an academic evaluation judge. Grade the student's answer strictly against the provided Rubric Criteria.
Return ONLY a valid JSON object matching this schema without markdown code blocks, backticks, or preamble:
{
  "criteria_evaluations": [
    {
      "criterion_id": "string",
      "max_marks": float,
      "awarded_marks": float,
      "satisfied": true | false,
      "reasoning": "string"
    }
  ],
  "evaluator_feedback": "string"
}"""


class LLMJudgeEvaluator:
    """Evaluates student answers against rubric criteria using Open-Source LLMs (via Ollama)."""

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        endpoint: str = "http://localhost:11434/api/generate",
    ):
        self.model_name = model_name
        self.endpoint = endpoint
        self.evaluator_type = "LLM"

        # 1. Health-check: Ping the Ollama base server
        base_url = endpoint.split("/api")[0]
        try:
            res = requests.get(base_url, timeout=3)
            res.raise_for_status()
        except Exception as e:
            raise ConnectionError(
                f"Ollama server is unreachable at {base_url}. Make sure Ollama is running ('ollama serve'). Details: {e}"
            ) from e

    def evaluate_question(
        self,
        question_id: str,
        question_text: str,
        model_answer: str,
        criteria: List[CriterionSchema],
        max_marks: float,
        student_answer: str,
    ) -> QuestionEvaluationResult:
        criteria_payload = [crit.model_dump() for crit in criteria]

        user_prompt = f"""- Question ID: {question_id}
- Max Marks: {max_marks}
- Question: {question_text}
- Model Answer: {model_answer}
- Rubric Criteria: {json.dumps(criteria_payload, indent=2)}
- Student Answer: {student_answer}"""

        payload = {
            "model": self.model_name,
            "system": SYSTEM_PROMPT,
            "prompt": user_prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0},
        }

        start_time = time.perf_counter()
        response = requests.post(self.endpoint, json=payload, timeout=90)
        latency_ms = (time.perf_counter() - start_time) * 1000
        response.raise_for_status()

        res_data = json.loads(response.json()["response"])

        criteria_results: List[CriterionEvaluationResult] = []
        total_awarded = 0.0

        for crit_res in res_data.get("criteria_evaluations", []):
            crit_id = crit_res.get("criterion_id")
            orig = next((c for c in criteria if c.id == crit_id), None)
            desc = orig.description if orig else "N/A"
            ctype = orig.type if orig else "semantic"

            awarded = float(crit_res.get("awarded_marks", 0.0))
            total_awarded += awarded
            is_satisfied = bool(crit_res.get("satisfied", False))

            criteria_results.append(
                CriterionEvaluationResult(
                    criterion_id=crit_id,
                    description=desc,
                    type=ctype,
                    max_marks=float(crit_res.get("max_marks", 0.0)),
                    marks_awarded=awarded,
                    satisfied=is_satisfied,
                    status="Satisfied" if is_satisfied else "Not Satisfied",
                    diagnostics=crit_res.get("reasoning", ""),
                )
            )

        # Cap total awarded marks at the question maximum
        total_awarded = min(max_marks, total_awarded)
        percentage = (
            round((total_awarded / max_marks) * 100, 2)
            if max_marks > 0
            else 0.0
        )

        return QuestionEvaluationResult(
            question_id=question_id,
            question_text=question_text,
            student_answer=student_answer,
            total_marks_awarded=round(total_awarded, 2),
            max_total_marks=round(max_marks, 2),
            percentage=percentage,
            criteria_breakdown=criteria_results,
            evaluator_feedback=f"{res_data.get('evaluator_feedback', '')} (Evaluator: {self.model_name}, Latency: {latency_ms:.1f}ms)",
        )