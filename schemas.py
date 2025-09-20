from typing import List, Optional
from pydantic import BaseModel, Field


class CounselRequest(BaseModel):
    interests: List[str] = Field(..., description="A list of academic or extra-curricular interests")
    board_marks: Optional[float] = Field(None, ge=0, le=100, description="Academic board percentage (optional)")
    entrance_exam_rank: int = Field(..., ge=1, description="Entrance exam rank (lower rank = better performance)")


class CollegeRecommendation(BaseModel):
    institute_short: str = Field(..., description="Short name of the institute (e.g., IIT-Bombay)")
    program_name: str = Field(..., description="Name of the program (e.g., Aerospace Engineering)")
    category: str = Field(..., description="Reservation category (e.g., GEN, OBC-NCL, SC, ST)")
    closing_rank: int = Field(..., description="Closing rank for this program in the dataset")
    eligibility_prob: float = Field(..., description="Predicted probability of eligibility (0 to 1)")


class CounselResponse(BaseModel):
    recommendation: List[CollegeRecommendation]

class CounselTextResponse(BaseModel):
    recommendation: str

class CombinedResponse(BaseModel):
    ml: List[CollegeRecommendation]
    llm: str