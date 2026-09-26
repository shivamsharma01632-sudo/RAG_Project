from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from .clgproject import create_chain
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from tempfile import TemporaryDirectory

app = FastAPI()
active_chain = None
MAX_PDF_SIZE = 25 * 1024 * 1024

app.add_middleware(
    CORSMiddleware,
    # Live Server may use either hostname; they are separate browser origins.
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://shivamsharma01632-sudo.github.io/RAG_Project/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    global active_chain

    if Path(file.filename or "").suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    try:
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "document.pdf"
            content = file.file.read(MAX_PDF_SIZE + 1)
            if len(content) > MAX_PDF_SIZE:
                raise HTTPException(status_code=413, detail="The PDF must be 25 MB or smaller.")
            if not content.startswith(b"%PDF-"):
                raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")
            pdf_path.write_bytes(content)
            active_chain = create_chain(pdf_path)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Could not prepare this PDF. Ensure Ollama is running and the embedding model is installed.",
        ) from error
    finally:
        file.file.close()

    return {"filename": file.filename, "message": "PDF ready for questions."}

@app.post("/path")
def answer_question(payload: Question):
    if active_chain is None:
        raise HTTPException(status_code=409, detail="Upload a PDF before asking a question.")
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Enter a question first.")
    try:
        return {"answer": active_chain.invoke(payload.question)}
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Could not answer the question. Check that the language-model service is available.",
        ) from error
