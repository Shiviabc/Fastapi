from typing import List, Optional
from pydantic import BaseModel, Field


class CounselRequest(BaseModel):
    interests: List[str] = Field(..., description="A list of academic or extra-curricular interests")
    board_marks: Optional[float] = Field(None, ge=0, le=100, description="Academic board percentage (optional)")
    entrance_exam_rank: int = Field(..., ge=1, description="Entrance exam rank (lower rank = better performance)")

    # --- ADDED FIELDS ---
    city: Optional[str] = Field(None, description="The student's home city (e.g., 'Mumbai')")
    state: Optional[str] = Field(None, description="The student's home state (e.g., 'Maharashtra')")
    quota: Optional[str] = Field("AI", description="Quota preference (e.g., HS, AI, OS). Defaults to AI")
    category: str = Field("GEN", description="Reservation category (e.g., GEN, OBC-NCL, SC, ST). Defaults to GEN")
    # --- END OF ADDED FIELDS ---


class CollegeRecommendation(BaseModel):
    institute_short: str = Field(..., description="Short name of the institute (e.g., IIT-Bombay)")
    stream: str = Field(..., description="Stream/program name (e.g., Aerospace Engineering)")
    category: str = Field(..., description="Reservation category")
    quota: str = Field(..., description="Quota type (HS/AI/OS)")
    closing_rank: int = Field(..., description="Closing rank for this program in the dataset")
    city: str = Field(..., description="City where the institute is located")
    state: str = Field(..., description="State where the institute is located")
    institute_type: Optional[str] = Field(None, description="Type of institute (e.g., IIT, NIT, University)")
    eligibility_prob: Optional[float] = Field(None, description="Predicted probability of eligibility (0 to 1)")


class CombinedResponse(BaseModel):
    ml: List[CollegeRecommendation]
    llm: str
