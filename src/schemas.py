from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class CriterionSchema(BaseModel):
    id: str
    description: str
    marks: float
    type: str  # "semantic" or "minimum_count"
    required_count: Optional[int] = None
    options: Optional[List[str]] = None
    entailment_threshold: float = Field(default=0.60, ge=0.0, le=1.0)


class QuestionRubricSchema(BaseModel):
    max_marks: float
    criteria: List[CriterionSchema]


class CriterionEvaluationResult(BaseModel):
    criterion_id: str
    description: str
    type: str
    marks_awarded: float
    max_marks: float
    status: str
    entailment_score: Optional[float] = None
    contradiction_score: Optional[float] = None
    matched_options: Optional[List[str]] = None


class QuestionEvaluationResult(BaseModel):
    question_id: str
    question_text: str
    student_answer: str
    total_marks_awarded: float
    max_total_marks: float
    percentage: float
    criteria_breakdown: List[CriterionEvaluationResult]


class StudentEvaluationReport(BaseModel):
    student_id: str
    total_marks_awarded: float
    max_total_marks: float
    percentage: float
    question_evaluations: Dict[str, QuestionEvaluationResult]