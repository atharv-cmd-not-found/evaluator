import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder
from src.schemas import CriterionSchema, CriterionEvaluationResult
from src.utils import split_into_sentences


def evaluate_semantic_criterion(
    model: CrossEncoder,
    student_answer: str,
    criterion: CriterionSchema,
    label_entailment: int = 1,
    label_contradiction: int = 0
) -> CriterionEvaluationResult:
    """Evaluates conceptual criteria using DeBERTa NLI probabilities."""
    student_sentences = split_into_sentences(student_answer)
    pairs = [(sentence, criterion.description) for sentence in student_sentences]

    logits = model.predict(pairs)
    if len(student_sentences) == 1:
        logits = [logits]

    probs = F.softmax(torch.tensor(logits), dim=1)

    # Find the maximum entailment probability across all sentences in student answer
    best_idx = torch.argmax(probs[:, label_entailment]).item()
    best_probs = probs[best_idx]

    ent_score = round(best_probs[label_entailment].item(), 4)
    con_score = round(best_probs[label_contradiction].item(), 4)

    if con_score > 0.60:
        status = "Contradiction / Incorrect Statement"
        awarded = 0.0
    elif ent_score >= 0.85:
        status = "Fully Satisfied"
        awarded = criterion.marks
    elif ent_score >= criterion.entailment_threshold:
        status = "Partially Satisfied"
        scale = (ent_score - criterion.entailment_threshold) / (0.85 - criterion.entailment_threshold)
        awarded = round(criterion.marks * (0.5 + 0.4 * scale), 2)
    else:
        status = "Missing / Not Mentioned"
        awarded = 0.0

    return CriterionEvaluationResult(
        criterion_id=criterion.id,
        description=criterion.description,
        type=criterion.type,
        marks_awarded=awarded,
        max_marks=criterion.marks,
        status=status,
        entailment_score=ent_score,
        contradiction_score=con_score
    )


def evaluate_minimum_count_criterion(
    student_answer: str,
    criterion: CriterionSchema
) -> CriterionEvaluationResult:
    """Evaluates criteria requiring a minimum count of required items."""
    matched_options = []
    student_answer_lower = student_answer.lower()

    if criterion.options:
        for option in criterion.options:
            if option.lower() in student_answer_lower:
                matched_options.append(option)

    match_count = len(matched_options)
    req_count = criterion.required_count or 1

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
        marks_awarded=awarded,
        max_marks=criterion.marks,
        status=status,
        matched_options=matched_options
    )