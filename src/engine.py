from typing import Dict, Any, Optional
from sentence_transformers import CrossEncoder

from src.schemas import (
    QuestionRubricSchema,
    QuestionEvaluationResult,
    StudentEvaluationReport
)
from src.criteria_evaluators import (
    evaluate_semantic_criterion,
    evaluate_minimum_count_criterion
)
from src.llm_evaluator import LLMJudgeEvaluator


class BaseRubricEvaluator:
    """Base interface for all rubric evaluators."""

    def evaluate_question(
        self,
        question_id: str,
        question_data: Dict[str, Any],
        rubric_data: Dict[str, Any],
        student_answer: str
    ) -> QuestionEvaluationResult:
        raise NotImplementedError

    def evaluate_student(
        self,
        student_data: Dict[str, Any],
        model_answers_data: Dict[str, Any],
        rubric_data: Dict[str, Any]
    ) -> StudentEvaluationReport:
        student_id = student_data.get("student", "student_unknown")
        student_answers = student_data.get("answers", {})

        question_evaluations: Dict[str, QuestionEvaluationResult] = {}
        total_student_awarded = 0.0
        total_student_max = 0.0

        for q_id, s_answer in student_answers.items():
            if q_id not in rubric_data or q_id not in model_answers_data:
                continue

            q_eval = self.evaluate_question(
                question_id=q_id,
                question_data=model_answers_data[q_id],
                rubric_data=rubric_data[q_id],
                student_answer=s_answer
            )

            question_evaluations[q_id] = q_eval
            total_student_awarded += q_eval.total_marks_awarded
            total_student_max += q_eval.max_total_marks

        overall_percentage = round((total_student_awarded / total_student_max) * 100, 2) if total_student_max > 0 else 0.0

        return StudentEvaluationReport(
            student_id=student_id,
            evaluator_model=getattr(self, "model_name", "custom_evaluator"),
            evaluator_type=getattr(self, "evaluator_type", "CROSS_ENCODER"),
            max_total_marks=round(total_student_max, 2),
            total_marks_awarded=round(total_student_awarded, 2),
            percentage=overall_percentage,
            question_evaluations=question_evaluations
        )


class CrossEncoderRubricEvaluator(BaseRubricEvaluator):
    """Evaluator supporting NLI, STS, and Passage Ranking Cross-Encoder models."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large", model_type: str = "nli"):
        print(f"Loading Cross-Encoder Model '{model_name}' ({model_type.upper()})...")
        self.model_name = model_name
        self.model_type = model_type.lower()
        self.evaluator_type = "CROSS_ENCODER"
        self.model = CrossEncoder(model_name)
        self.LABEL_CONTRADICTION = 0
        self.LABEL_ENTAILMENT = 1
        self.LABEL_NEUTRAL = 2

    def evaluate_question(
        self,
        question_id: str,
        question_data: Dict[str, Any],
        rubric_data: Dict[str, Any],
        student_answer: str
    ) -> QuestionEvaluationResult:
        rubric = QuestionRubricSchema(**rubric_data)
        question_text = question_data.get("question", "")

        criteria_breakdown = []
        total_awarded = 0.0
        total_max = 0.0

        for crit in rubric.criteria:
            if crit.type == "minimum_count":
                res = evaluate_minimum_count_criterion(
                    student_answer=student_answer,
                    criterion=crit
                )
            else:
                res = evaluate_semantic_criterion(
                    model=self.model,
                    student_answer=student_answer,
                    criterion=crit,
                    model_type=self.model_type,
                    label_entailment=self.LABEL_ENTAILMENT,
                    label_contradiction=self.LABEL_CONTRADICTION
                )

            total_awarded += res.marks_awarded
            total_max += res.max_marks
            criteria_breakdown.append(res)

        total_awarded = min(rubric.max_marks, total_awarded)
        percentage = round((total_awarded / rubric.max_marks) * 100, 2) if rubric.max_marks > 0 else 0.0

        return QuestionEvaluationResult(
            question_id=question_id,
            question_text=question_text,
            student_answer=student_answer,
            total_marks_awarded=round(total_awarded, 2),
            max_total_marks=round(rubric.max_marks, 2),
            percentage=percentage,
            criteria_breakdown=criteria_breakdown,
            evaluator_feedback=f"Evaluated using {self.model_name} ({self.model_type.upper()})"
        )


class LLMRubricEvaluator(BaseRubricEvaluator):
    """Evaluator supporting Open-Source LLMs (e.g., Qwen 2.5, Mistral, Phi-3.5 via Ollama)."""

    def __init__(self, model_name: str = "qwen2.5:7b", endpoint: str = "http://localhost:11434/api/generate"):
        print(f"Initializing LLM Evaluator '{model_name}'...")
        self.model_name = model_name
        self.evaluator_type = "LLM"
        self.judge = LLMJudgeEvaluator(model_name=model_name, endpoint=endpoint)

    def evaluate_question(
        self,
        question_id: str,
        question_data: Dict[str, Any],
        rubric_data: Dict[str, Any],
        student_answer: str
    ) -> QuestionEvaluationResult:
        rubric = QuestionRubricSchema(**rubric_data)
        return self.judge.evaluate_question(
            question_id=question_id,
            question_text=question_data.get("question", ""),
            model_answer=question_data.get("model_answer", ""),
            criteria=rubric.criteria,
            max_marks=rubric.max_marks,
            student_answer=student_answer
        )


# Backward-compatibility alias
NLIRubricEvaluator = CrossEncoderRubricEvaluator