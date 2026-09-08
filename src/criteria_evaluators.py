from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder

from src.schemas import CriterionEvaluationResult, CriterionSchema
from src.utils import split_into_sentences


def evaluate_semantic_criterion(
    model: CrossEncoder,
    student_answer: str,
    criterion: CriterionSchema,
    model_type: str = "nli",
    label_entailment: int = 1,
    label_contradiction: int = 0,
) -> CriterionEvaluationResult:
    """
    Evaluates conceptual/semantic criteria across NLI, STS, and Passage Ranking models.
    Scans sentence-by-sentence to find the strongest supporting match in the student's answer.
    """
    student_sentences = split_into_sentences(student_answer)
    if not student_sentences:
        student_sentences = [student_answer]

    # Evaluate (sentence, criterion_description) pairs
    pairs = [(sentence, criterion.description) for sentence in student_sentences]
    raw_predictions = model.predict(pairs)

    # Ensure output is a list or 2D array if only 1 sentence exists
    if len(student_sentences) == 1 and not isinstance(raw_predictions, (list, np.ndarray)):
        raw_predictions = [raw_predictions]

   
    # -------------------------------------------------------------
    # 3. STS Cross-Encoders (e.g., stsb-roberta-large)
    # -------------------------------------------------------------
    
    scores = np.array(raw_predictions, dtype=float).flatten()
    best_idx = int(np.argmax(scores))
    best_score = round(float(np.clip(scores[best_idx], 0.0, 1.0)), 4)

    satisfied = best_score >= 0.65
    if best_score >= 0.80:
        status = "Fully Satisfied"
        awarded = criterion.marks
    elif best_score >= 0.50:
        status = "Partially Satisfied"
        awarded = round(criterion.marks * best_score, 2)
    else:
        status = "Missing / Low Similarity"
        awarded = 0.0

    return CriterionEvaluationResult(
        criterion_id=criterion.id,
        description=criterion.description,
        type=criterion.type,
        max_marks=criterion.marks,
        marks_awarded=awarded,
        satisfied=satisfied,
        status=status,
        diagnostics=f"STS Similarity: {best_score:.3f}",
        semantic_similarity=best_score,
    )


def evaluate_minimum_count_criterion(
    student_answer: str,
    criterion: CriterionSchema,
) -> CriterionEvaluationResult:
    """Evaluates criteria requiring a minimum count of required items/keywords."""
    matched_options = []
    student_answer_lower = student_answer.lower()

    if criterion.options:
        for option in criterion.options:
            if option.lower() in student_answer_lower:
                matched_options.append(option)

    match_count = len(matched_options)
    req_count = criterion.required_count or 1
    satisfied = match_count >= req_count

    if match_count >= req_count:
        status = f"Fully Satisfied (Matched {match_count}/{req_count} required)"
        awarded = criterion.marks
    elif match_count > 0:
        status = f"Partially Satisfied (Matched {match_count}/{req_count} required)"
        ratio = match_count / req_count
        awarded = round(criterion.marks * ratio, 2)
    else:
        status = "Missing / No Options Identified"
        awarded = 0.0

    return CriterionEvaluationResult(
        criterion_id=criterion.id,
        description=criterion.description,
        type=criterion.type,
        max_marks=criterion.marks,
        marks_awarded=awarded,
        satisfied=satisfied,
        status=status,
        diagnostics=f"Matched {match_count}/{req_count} options: {matched_options}",
        matched_options=matched_options,
        matched_count=match_count,
    )