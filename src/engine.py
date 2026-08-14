from typing import Dict, Any
from sentence_transformers import CrossEncoder

from src.schemas import (
    QuestionRubricSchema,
    QuestionEvaluationResult
)
from src.criteria_evaluators import (
    evaluate_semantic_criterion,
    evaluate_minimum_count_criterion
)


class NLIRubricEvaluator:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large"):
        print(f"Loading DeBERTa Cross-Encoder Model '{model_name}'...")
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
            if crit.type == "semantic":
                res = evaluate_semantic_criterion(
                    model=self.model,
                    student_answer=student_answer,
                    criterion=crit,
                    label_entailment=self.LABEL_ENTAILMENT,
                    label_contradiction=self.LABEL_CONTRADICTION
                )
            elif crit.type == "minimum_count":
                res = evaluate_minimum_count_criterion(
                    student_answer=student_answer,
                    criterion=crit
                )
            else:
                res = evaluate_semantic_criterion(
                    model=self.model,
                    student_answer=student_answer,
                    criterion=crit
                )

            total_awarded += res.marks_awarded
            total_max += res.max_marks
            criteria_breakdown.append(res)

        percentage = round((total_awarded / total_max) * 100, 2) if total_max > 0 else 0.0

        return QuestionEvaluationResult(
            question_id=question_id,
            question_text=question_text,
            student_answer=student_answer,
            total_marks_awarded=round(total_awarded, 2),
            max_total_marks=round(total_max, 2),
            percentage=percentage,
            criteria_breakdown=criteria_breakdown
        )