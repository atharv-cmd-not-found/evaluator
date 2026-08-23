from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Rubric Input Schemas
# ---------------------------------------------------------

class CriterionSchema(BaseModel):
    id: str = Field(..., description="Unique criterion identifier (e.g. q1_definition)")
    description: str = Field(..., description="Criterion description or requirement")
    marks: float = Field(..., ge=0.0, description="Max marks allocated for this criterion")
    type: str = Field(default="semantic", description="'semantic' or 'minimum_count'")
    required_count: Optional[int] = Field(default=None, ge=1, description="Minimum count of required options")
    options: Optional[List[str]] = Field(default=None, description="Valid options list for count criteria")
    entailment_threshold: float = Field(default=0.60, ge=0.0, le=1.0, description="NLI entailment confidence cutoff")


class QuestionRubricSchema(BaseModel):
    max_marks: float = Field(..., ge=0.0, description="Maximum possible marks for the question")
    criteria: List[CriterionSchema] = Field(..., description="List of criteria for this question")


# ---------------------------------------------------------
# Evaluation Result Schemas
# ---------------------------------------------------------

class CriterionEvaluationResult(BaseModel):
    criterion_id: str = Field(..., description="Evaluated criterion identifier")
    description: str = Field(..., description="Criterion requirement text")
    type: str = Field(..., description="Criterion type ('semantic' or 'minimum_count')")
    max_marks: float = Field(..., ge=0.0, description="Maximum marks for this criterion")
    marks_awarded: float = Field(..., ge=0.0, description="Marks awarded to student")
    satisfied: bool = Field(..., description="Whether criterion conditions were met")
    status: str = Field(default="Evaluated", description="Status text (e.g. 'Satisfied', 'Partial', 'Not Satisfied')")
    diagnostics: Optional[str] = Field(default=None, description="Diagnostic notes, logits, or match details")
    
    # NLI & STS Metrics
    entailment_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    contradiction_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    neutral_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    semantic_similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    # Minimum Count Metrics
    matched_options: Optional[List[str]] = Field(default=None, description="List of options successfully detected in answer")
    matched_count: Optional[int] = Field(default=None, ge=0, description="Total count of matched options")


class QuestionEvaluationResult(BaseModel):
    question_id: str = Field(..., description="Question identifier (e.g. Q1)")
    question_text: str = Field(..., description="Original question prompt")
    student_answer: str = Field(..., description="Student's raw submission text")
    max_total_marks: float = Field(..., ge=0.0, description="Total possible marks for question")
    total_marks_awarded: float = Field(..., ge=0.0, description="Total marks awarded for question")
    percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Score percentage for this question")
    criteria_breakdown: List[CriterionEvaluationResult] = Field(default_factory=list, description="Per-criterion scoring breakdown")
    evaluator_feedback: Optional[str] = Field(default=None, description="Overall LLM or system feedback")


class StudentEvaluationReport(BaseModel):
    student_id: str = Field(..., description="Student identifier")
    evaluator_model: str = Field(..., description="Model identifier used for evaluation")
    evaluator_type: str = Field(..., description="'CROSS_ENCODER' or 'LLM'")
    max_total_marks: float = Field(..., ge=0.0, description="Sum of max possible marks across all questions")
    total_marks_awarded: float = Field(..., ge=0.0, description="Sum of all awarded marks across questions")
    percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Overall percentage score")
    question_evaluations: Dict[str, QuestionEvaluationResult] = Field(default_factory=dict, description="Question-level evaluation results keyed by Q_ID")