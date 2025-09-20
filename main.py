import json
import logging  # Import logging
from fastapi import FastAPI, Depends
from schemas import CounselRequest, CollegeRecommendation, CombinedResponse
from ai.ollamas import Ollama
from ai.retrieve import Retriever
from auth.dependencies import get_user_identifier
from auth.throttling import apply_rate_limit
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# --- CORS Setup ---
origins = [
    "http://localhost:8080",
    "http://localhost:5000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- System Prompt Loader ---
def load_system_prompt():
    try:
        with open("prompt/system_prompt.md", "r") as f:
            return f.read()
    except FileNotFoundError:
        print("Warning: system_prompt.md not found.")
        return ""


SYSTEM_PROMPT = load_system_prompt()
ai_platform = Ollama(model="mistral")

# --- Retriever Initialization ---
retriever = None
retriever_error = None
try:
    retriever = Retriever()
except Exception as e:
    retriever_error = str(e)
    print(f"Retriever failed to initialize: {retriever_error}")

# --- ML Model Initialization ---
ml_model = None
try:
    ml_model = joblib.load("college_eligibility_predictor.pkl")
except Exception as e:
    print(f"ML model failed to load: {e}")


# --- Combined Endpoint ---
@app.post("/counseling/combined", response_model=CombinedResponse)
async def combined_counseling(request: CounselRequest, user_id: str = Depends(get_user_identifier)):
    logging.info("Combined counseling endpoint hit.")
    apply_rate_limit(user_id)

    if not retriever or not ml_model:
        error_detail = retriever_error or "ML model not loaded."
        return CombinedResponse(ml=[], llm=f"Error: The recommendation engine is not available. Details: {error_detail}")

    query = ", ".join(request.interests)
    candidates = retriever.find_similar_colleges(query, top_k=20)

    if not candidates:
        return CombinedResponse(ml=[], llm="No suitable colleges found based on your interests.")

    df_candidates = pd.DataFrame(candidates)
    X_new = pd.DataFrame({
        "student_rank": [request.entrance_exam_rank] * len(df_candidates),
        "program_name": df_candidates.get("program_name", df_candidates.get("name")),
        "category": df_candidates.get("category", "GEN"),
    })

    probs = ml_model.predict_proba(X_new)[:, 1]
    df_candidates["eligibility_prob"] = probs
    top_ml_colleges = df_candidates.sort_values("eligibility_prob", ascending=False).head(3)

    ml_recommendations = [CollegeRecommendation(**row) for _, row in top_ml_colleges.iterrows()]
    logging.info("ML recommendations generated.")

    # === LLM Logic ===
    context_str = top_ml_colleges.to_json(orient="records", indent=2)
    prompt_text = (
        f"A student has the following profile:\n"
        f"- Interests: {query}\n"
        f"- Board Marks: {request.board_marks}%\n"
        f"- Entrance Exam Rank: {request.entrance_exam_rank}\n\n"
        f"Based on their rank, our algorithm has recommended these top 3 colleges:\n{context_str}\n\n"
        f"Your task is to act as a career counselor. Please provide a detailed, encouraging, and helpful justification "
        f"for why these specific colleges are a good fit for the student. Explain the strengths of each program "
        f"and give advice on what the student should focus on next."
    )
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt_text}"

    llm_counseling_text = await ai_platform.chat(full_prompt)
    logging.info("LLM counseling generated.")

    return CombinedResponse(
        ml=ml_recommendations,
        llm=llm_counseling_text
    )


# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "College Counseling API with RAG is running."}
