import os
import re
import time
import json
from typing import Any, Dict, Optional, List, Tuple
from collections import defaultdict, deque

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StringConstraints, field_validator
from typing_extensions import Annotated
from dotenv import load_dotenv

load_dotenv()

# ------------------------------
# Config
# ------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # set in your environment / .env
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
OUTBOUND_TIMEOUT_SECONDS = 20
MAX_INPUT_CHARS = 4000
MAX_OUTPUT_TOKENS = 512
TEMPERATURE = 0.7
ENABLE_STREAMING = False

# Rate limits (rough guardrails per user/session)
RL_MAX_REQUESTS_PER_MINUTE = 20
RL_MAX_CONCURRENT_PER_USER = 2

# Very lightweight profanity/off-limits keywords (child-safe persona rules)
FORBIDDEN_TOPICS = [
    "weapon", "suicide", "self-harm", "sex", "porn", "drugs", "alcohol",
    "politics", "religion", "gore", "violence", "crime", "abuse", "terrorism",
]
# Profanity sample (not exhaustive)
PROFANITY = [
    "fuck","shit","bitch","cunt","nigger","nigga","slut","whore","dick","pussy",
    "asshole","bastard","prick","faggot","wanker","retard"
]

# Daily token/cost budget per user (approximate; char->token ~ 4 chars ≈ 1 token)
DAILY_TOKEN_BUDGET = 20_000
TOKENS_PER_CHAR = 1/4

# ------------------------------
# Redaction rules for logs
# ------------------------------
RE_EMAIL = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
RE_ADDRESS = re.compile(r'\b\d{1,5}\s+[A-Za-z0-9\.\s]+(Street|St|Road|Rd|Ave|Avenue|Blvd|Lane|Ln)\b', re.I)
RE_SCHOOL = re.compile(r'\b[A-Z][A-Za-z\s]+ (Elementary|Middle|High) School\b')
RE_LOCATION = re.compile(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\,?\s([A-Z][a-z]+)\b')

def redact(text: str) -> str:
    text = RE_EMAIL.sub("[EMAIL]", text)
    text = RE_ADDRESS.sub("[ADDRESS]", text)
    text = RE_SCHOOL.sub("[SCHOOL]", text)
    text = RE_LOCATION.sub("[LOCATION]", text)
    return text

# ------------------------------
# Pydantic schemas (Interface Contract)
# ------------------------------
class ChatRequest(BaseModel):
    user_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    session_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_INPUT_CHARS)]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def message_len(cls, v: str) -> str:
        if len(v) > MAX_INPUT_CHARS:
            raise ValueError(f"message exceeds {MAX_INPUT_CHARS} characters")
        return v

class SafetyFlags(BaseModel):
    blocked: bool = False
    reason: Optional[str] = None
    profane: bool = False
    redactions_applied: bool = False

class Usage(BaseModel):
    input_tokens_est: int
    output_tokens_cap: int
    total_tokens_est: int

class ChatResponse(BaseModel):
    text: str
    safety: SafetyFlags
    usage: Usage
    error: Optional[str] = None

# ------------------------------
# App & CORS
# ------------------------------
app = FastAPI(title="Edge Chat API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ------------------------------
# In-memory rate limiting & budget tracking
# ------------------------------
_window: Dict[str, deque] = defaultdict(deque)  # timestamps of requests
_inflight: Dict[str, int] = defaultdict(int)
_daily_tokens: Dict[Tuple[str, str], int] = defaultdict(int)  # (user_id, yyyy-mm-dd) -> tokens

def rate_limit(user_id: str):
    now = time.time()
    window = _window[user_id]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RL_MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(429, detail="Rate limit exceeded (per-minute).")
    if _inflight[user_id] >= RL_MAX_CONCURRENT_PER_USER:
        raise HTTPException(429, detail="Too many concurrent requests.")
    window.append(now)

def enter(user_id: str):
    _inflight[user_id] += 1

def leave(user_id: str):
    _inflight[user_id] = max(0, _inflight[user_id] - 1)

def today() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())

def charge_daily_budget(user_id: str, est_total_tokens: int):
    key = (user_id, today())
    _daily_tokens[key] += est_total_tokens
    if _daily_tokens[key] > DAILY_TOKEN_BUDGET:
        raise HTTPException(402, detail="Daily token budget exceeded.")

# ------------------------------
# Safety/validation
# ------------------------------
def evaluate_safety(text: str) -> SafetyFlags:
    lower = text.lower()
    blocked_reason = None
    profane = any(p in lower for p in PROFANITY)
    forbidden = any(k in lower for k in FORBIDDEN_TOPICS)
    if profane:
        blocked_reason = "Profanity is not allowed."
    if forbidden:
        blocked_reason = "Topic is not allowed for this assistant."
    return SafetyFlags(
        blocked=bool(blocked_reason),
        reason=blocked_reason,
        profane=profane,
        redactions_applied=False
    )

# ------------------------------
# Logging (redacted)
# ------------------------------
def log_event(kind: str, user_id: str, session_id: str, payload: Dict[str, Any]):
    safe_payload = json.loads(json.dumps(payload))
    for k in ("message", "text", "error"):
        if k in safe_payload and isinstance(safe_payload[k], str):
            safe_payload[k] = redact(safe_payload[k])
    uid_display = f"user:{hash(user_id)%1_000_000}"
    sid_display = f"session:{hash(session_id)%1_000_000}"
    print(json.dumps({"kind": kind, "user": uid_display, "session": sid_display, "data": safe_payload}))

# ------------------------------
# Gemini call
# ------------------------------
async def call_gemini(prompt: str, metadata: Dict[str, Any]) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(500, detail="GEMINI_API_KEY not configured.")

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": MAX_OUTPUT_TOKENS
        },
        # Relax safety thresholds to avoid empty completions on benign prompts.
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }

    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=OUTBOUND_TIMEOUT_SECONDS) as client:
        r = await client.post(GEMINI_ENDPOINT, headers=headers, json=payload)

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"LLM upstream error: {r.text[:500]}")

    data = r.json()

    # Robust extraction across response variants
    def extract_text(d: Dict[str, Any]) -> str:
        try:
            cands = d.get("candidates") or []
            if cands:
                parts = cands[0].get("content", {}).get("parts") or []
                if parts:
                    return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        except Exception:
            pass
        if isinstance(d.get("output_text"), str):
            return d["output_text"]
        if "candidates" in d and d["candidates"]:
            content = d["candidates"][0].get("content")
            if isinstance(content, list) and content and isinstance(content[0], dict) and "text" in content[0]:
                return content[0]["text"]
        return ""

    text = extract_text(data).strip()
    if not text:
        raise HTTPException(502, detail=f"LLM returned empty completion. Payload head: {json.dumps(data)[:500]}")
    return text

# ------------------------------
# Health
# ------------------------------
@app.get("/healthz")
async def health():
    return {"status": "ok", "model": GEMINI_MODEL}

# ------------------------------
# Endpoint (POST /edge/chat)
# ------------------------------
@app.post("/edge/chat", response_model=ChatResponse)
async def edge_chat(req: ChatRequest, request: Request):
    rate_limit(req.user_id)
    enter(req.user_id)
    t0 = time.time()
    try:
        # 1) Validate & safety-check
        safety = evaluate_safety(req.message)
        if safety.blocked:
            safety.redactions_applied = True
            log_event("blocked", req.user_id, req.session_id, {
                "message": req.message, "reason": safety.reason
            })
            usage = Usage(
                input_tokens_est=int(len(req.message) * TOKENS_PER_CHAR),
                output_tokens_cap=0,
                total_tokens_est=int(len(req.message) * TOKENS_PER_CHAR)
            )
            return ChatResponse(
                text="I can’t help with that. Let’s try a safer topic.",
                safety=safety,
                usage=usage,
                error="safety_block"
            )

        # 2) Budget guardrail
        input_tokens_est = int(len(req.message) * TOKENS_PER_CHAR)
        total_est = input_tokens_est + MAX_OUTPUT_TOKENS
        charge_daily_budget(req.user_id, total_est)

        # 3) Build prompt persona wrapper
        persona_prefix = (
            "You are a playful, extremely knowledgeable friend for a 9-year-old. "
            "Use simple words, 5–12 word sentences, friendly and encouraging. "
            "Avoid forbidden topics. Encourage curiosity, effort, kindness."
        )
        full_prompt = f"{persona_prefix}\n\nUser: {req.message}"

        # 4) Call Gemini
        text = await call_gemini(full_prompt, req.metadata)

        # 5) Response assembly
        latency_ms = int((time.time() - t0) * 1000)
        usage = Usage(
            input_tokens_est=input_tokens_est,
            output_tokens_cap=MAX_OUTPUT_TOKENS,
            total_tokens_est=total_est
        )

        # 6) Logging (redacted)
        log_event("ok", req.user_id, req.session_id, {
            "message": req.message, "text": text[:500],
            "latency_ms": latency_ms, "usage": usage.dict()
        })

        return ChatResponse(
            text=text.strip(),
            safety=safety,
            usage=usage,
            error=None
        )
    except HTTPException as e:
        log_event("error", req.user_id, req.session_id, {
            "message": req.message, "error": str(e.detail)
        })
        raise
    except Exception as e:
        log_event("error", req.user_id, req.session_id, {
            "message": req.message, "error": repr(e)
        })
        raise HTTPException(500, detail="Internal error")
    finally:
        leave(req.user_id)
