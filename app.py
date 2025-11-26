# backend/app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
from pathlib import Path
from .ingest import parse_and_store_documents
from .vectorstore import VectorStore
from .rag_agent import RAGAgent

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Autonomous QA Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons
vectorstore = VectorStore(str(BASE_DIR / "vectorstore.db"))
agent = RAGAgent(vectorstore=vectorstore)

# ------------------------
# Models for JSON Requests
# ------------------------
class QueryModel(BaseModel):
    query: str

class ScriptModel(BaseModel):
    testcase_json: dict

# ------------------------
# File Uploads
# ------------------------
@app.post("/upload_support_doc")
async def upload_support_doc(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "uploaded", "filename": file.filename}

@app.post("/upload_checkout")
async def upload_checkout(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "uploaded", "filename": file.filename}

# ------------------------
# Build Knowledge Base
# ------------------------
@app.post("/build_kb")
async def build_kb():
    uploaded_files = list(UPLOAD_DIR.glob("*"))
    if not uploaded_files:
        return JSONResponse({"error": "No uploaded files"}, status_code=400)

    docs = parse_and_store_documents([str(p) for p in uploaded_files])
    vectorstore.reset()
    vectorstore.add_documents(docs)

    return {
        "status": "Knowledge base created",
        "documents_ingested": len(docs)
    }

# ------------------------
# Generate Test Cases
# ------------------------
@app.post("/generate_testcases")
async def generate_testcases(req: QueryModel):
    results = agent.generate_test_cases(req.query)
    return JSONResponse(results)

# ------------------------
# Generate Selenium Script
# ------------------------
@app.post("/generate_script")
async def generate_script(req: ScriptModel):
    script_text = agent.generate_selenium_script(req.testcase_json)
    return {"script": script_text}
