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
    # 1. Natural Language Inference (NLI) Cross-Encoders
    # -------------------------------------------------------------
    if model_type == "nli":
        probs = F.softmax(torch.tensor(raw_predictions, dtype=torch.float32), dim=1)

        # Find the sentence with the highest entailment probability
        best_idx = torch.argmax(probs[:, label_entailment]).item()
        best_probs = probs[best_idx]

        ent_score = round(best_probs[label_entailment].item(), 4)
        con_score = round(best_probs[label_contradiction].item(), 4)
        neut_score = round(best_probs[2].item(), 4) if best_probs.shape[0] > 2 else 0.0

        satisfied = False
        if con_score > 0.60:
            status = "Contradiction / Incorrect Statement"
            awarded = 0.0
        elif ent_score >= 0.85:
            status = "Fully Satisfied"
            awarded = criterion.marks
            satisfied = True
        elif ent_score >= criterion.entailment_threshold:
            status = "Partially Satisfied"
            scale = (ent_score - criterion.entailment_threshold) / (0.85 - criterion.entailment_threshold + 1e-8)
            awarded = round(criterion.marks * (0.5 + 0.4 * scale), 2)
            satisfied = True
        else:
            status = "Missing / Not Mentioned"
            awarded = 0.0

        diagnostics = f"Entail: {ent_score:.2f} | Contra: {con_score:.2f} | Neut: {neut_score:.2f}"

        return CriterionEvaluationResult(
            criterion_id=criterion.id,
            description=criterion.description,
            type=criterion.type,
            max_marks=criterion.marks,
            marks_awarded=awarded,
            satisfied=satisfied,
            status=status,
            diagnostics=diagnostics,
            entailment_score=ent_score,
            contradiction_score=con_score,
            neutral_score=neut_score,
        )

    # -------------------------------------------------------------
    # 2. Passage Rerankers (e.g., MS-MARCO, BGE Reranker)
    # -------------------------------------------------------------
    elif model_type == "ranking":
        logits = np.array(raw_predictions, dtype=float).flatten()
        # Convert unbounded logits to sigmoid confidence [0.0, 1.0]
        sigmoid_scores = 1.0 / (1.0 + np.exp(-logits))

        best_idx = int(np.argmax(sigmoid_scores))
        best_score = round(float(sigmoid_scores[best_idx]), 4)

        satisfied = best_score >= 0.65
        if best_score >= 0.80:
            status = "Fully Satisfied"
            awarded = criterion.marks
        elif best_score >= 0.55:
            status = "Partially Satisfied"
            awarded = round(criterion.marks * best_score, 2)
        else:
            status = "Missing / Low Relevance"
            awarded = 0.0

        return CriterionEvaluationResult(
            criterion_id=criterion.id,
            description=criterion.description,
            type=criterion.type,
            max_marks=criterion.marks,
            marks_awarded=awarded,
            satisfied=satisfied,
            status=status,
            diagnostics=f"Sigmoid: {best_score:.3f} (Logit: {logits[best_idx]:.2f})",
            semantic_similarity=best_score,
        )

    # -------------------------------------------------------------
    # 3. STS Cross-Encoders (e.g., stsb-roberta-large)
    # -------------------------------------------------------------
    else:
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