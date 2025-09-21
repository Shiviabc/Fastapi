import json
import logging
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
        return "You are an expert college counselor for engineering admissions in India."


# --- AI Component Initialization ---
SYSTEM_PROMPT = load_system_prompt()
ai_platform = Ollama(model="mistral")
retriever = None
ml_model = None

try:
    retriever = Retriever()
    ml_model = joblib.load("college_eligibility_predictor.pkl")
    print("Retriever and ML Model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load AI components. Details: {e}")


# --- Main Counseling Endpoint ---
@app.post("/counseling", response_model=CombinedResponse)
async def cl_rec(request: CounselRequest, user_id: str = Depends(get_user_identifier)):
    apply_rate_limit(user_id)

    if not retriever or not ml_model:
        return CombinedResponse(ml=[], llm="Error: The recommendation engine is not available.")

    query = ", ".join(request.interests)

    # === Step 1: RAG Search (Interest-based) ===
    rag_candidates = retriever.find_similar_colleges(query, top_k=200)
    if not rag_candidates:
        return CombinedResponse(ml=[], llm="Sorry, we couldn't find any colleges matching your interests.")
    df_candidates = pd.DataFrame(rag_candidates)

    # === Step 2: Soft Filtering (Location & Quota) ===
    df_filtered = df_candidates.copy()

    # State filter
    if request.state and "state" in df_filtered.columns:
        tmp = df_filtered[df_filtered["state"].str.strip().str.lower() == request.state.strip().lower()]
        if not tmp.empty:
            df_filtered = tmp

    # City filter
    if request.city and "city" in df_filtered.columns:
        tmp = df_filtered[df_filtered["city"].str.strip().str.lower() == request.city.strip().lower()]
        if not tmp.empty:
            df_filtered = tmp

    # Quota filter
    if request.quota and "quota" in df_filtered.columns:
        tmp = df_filtered[df_filtered["quota"].str.strip().str.upper() == request.quota.strip().upper()]
        if not tmp.empty:
            df_filtered = tmp

    # === Case A: We still have results ===
    if not df_filtered.empty:
        # === Step 3: ML Prediction (Rank-based Eligibility) ===
        X_new = pd.DataFrame({
            "student_rank": [request.entrance_exam_rank] * len(df_filtered),
            "program_name": df_filtered["stream"],   # FIX: match training column name
            "category": [request.category] * len(df_filtered),
            "institute_type": df_filtered["institute_type"]
        })
        probs = ml_model.predict_proba(X_new)[:, 1]
        df_filtered["eligibility_prob"] = probs

        top_ml_colleges = df_filtered.sort_values("eligibility_prob", ascending=False).head(3)
        ml_recommendations = [CollegeRecommendation(**row) for row in top_ml_colleges.to_dict(orient="records")]

        # === Step 4: LLM Counseling (Exact Matches) ===
        context_str = top_ml_colleges.to_json(orient="records", indent=2)
        prompt_text = (
            f"A student from {request.city}, {request.state} has the following profile:\n"
            f"- Interests: {query}\n"
            f"- Entrance Exam Rank: {request.entrance_exam_rank}\n"
            f"- Category: {request.category}\n"
            f"- Desired Quota: {request.quota}\n\n"
            f"Based on their profile, our algorithm has recommended these top 3 colleges:\n{context_str}\n\n"
            f"Your task is to act as a career counselor. Please provide a detailed, encouraging, and helpful justification "
            f"for why these specific colleges are a good fit. Explain the strengths of each program."
        )
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt_text}"
        llm_counseling_text = await ai_platform.chat(full_prompt)

        return CombinedResponse(ml=ml_recommendations, llm=llm_counseling_text)

    # === Case B: No exact matches → Fallback ===
    fallback_colleges = df_candidates.head(3)
    context_str = fallback_colleges.to_json(orient="records", indent=2)

    prompt_text = (
        f"A student from {request.city}, {request.state} is looking for colleges under quota {request.quota}, "
        f"but no exact matches were found.\n\n"
        f"Here are some similar alternative options instead:\n{context_str}\n\n"
        f"Please provide helpful suggestions and encourage the student about considering these alternatives."
    )
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt_text}"
    llm_counseling_text = await ai_platform.chat(full_prompt)

    return CombinedResponse(ml=[], llm=llm_counseling_text)


# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "College Counseling API with RAG and ML is running."}
