"""FastAPI endpoints for uploading and querying PDF documents.

PDF chunks are keyed by the SHA-256 hash of their file contents. Re-uploading
the same document therefore reuses the vectors already stored in Chroma.
"""

from __future__ import annotations

import logging
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()
active_chain = None
MAX_PDF_SIZE = 25 * 1024 * 1024
PERSIST_DIRECTORY = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "pdf_documents"
index_lock = Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://shivamsharma01632-sudo.github.io/RAG_Project/",
        "https://rag-project-b4lafscmb-ssharma1632.vercel.app","*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


def get_vector_store() -> Chroma:
    """Open the persisted collection used by every uploaded PDF."""
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIRECTORY),
    )


def format_context(documents: list[Document]) -> str:
    if not documents:
        return "No relevant passages were retrieved from this document."
    return "\n\n".join(document.page_content for document in documents)


def create_chain(pdf_path: Path, original_filename: str, content_hash: str):
    """Index a PDF once, then return a chain restricted to that PDF's chunks."""
    vector_store = get_vector_store()

    # The lock closes the small race window where two identical uploads arrive
    # before either request has saved its embeddings.
    with index_lock:
        existing = vector_store.get(
            where={"source_hash": content_hash},
            include=[],
        )
        existing_count = len(existing["ids"])

        if existing_count:
            index_status = "reused"
            chunk_count = existing_count
            logger.info(
                "Index reuse | file=%s | hash=%s | chunks=%d",
                original_filename,
                content_hash[:12],
                chunk_count,
            )
        else:
            pages = PyPDFLoader(str(pdf_path)).load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", " ", ""],
            )
            chunks = splitter.split_documents(pages)
            if not chunks:
                raise ValueError("No readable text was found in this PDF.")

            for chunk_number, chunk in enumerate(chunks):
                chunk.metadata.update(
                    {
                        "source_hash": content_hash,
                        "source_filename": original_filename,
                        "chunk_number": chunk_number,
                    }
                )

            # Stable IDs make indexing safe even if a process stops mid-upload.
            ids = [f"{content_hash}:{chunk_number}" for chunk_number in range(len(chunks))]
            vector_store.add_documents(chunks, ids=ids)
            index_status = "created"
            chunk_count = len(chunks)
            logger.info(
                "Index created | file=%s | hash=%s | chunks=%d",
                original_filename,
                content_hash[:12],
                chunk_count,
            )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4, "filter": {"source_hash": content_hash}},
    )
    prompt = PromptTemplate(
        template=(
            "Answer only from the text inside <context>. "
            "If the answer is not in the context, say so.\n\n"
            "<context>\n{context}\n</context>\n\nQuestion: {question}"
        ),
        input_variables=["context", "question"],
    )
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_context),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, index_status, chunk_count


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    global active_chain

    filename = file.filename or "document.pdf"
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    try:
        content = file.file.read(MAX_PDF_SIZE + 1)
        if len(content) > MAX_PDF_SIZE:
            raise HTTPException(status_code=413, detail="The PDF must be 25 MB or smaller.")
        if not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

        content_hash = sha256(content).hexdigest()
        logger.info("Upload received | file=%s | hash=%s", filename, content_hash[:12])
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "document.pdf"
            pdf_path.write_bytes(content)
            active_chain, index_status, chunk_count = create_chain(pdf_path, filename, content_hash)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("PDF preparation failed | file=%s", filename)
        raise HTTPException(
            status_code=503,
            detail="Could not prepare this PDF. Ensure Ollama is running and the embedding model is installed.",
        ) from error
    finally:
        file.file.close()

    message = "Existing embeddings loaded." if index_status == "reused" else "PDF indexed and ready for questions."
    return {
        "filename": filename,
        "message": message,
        "index_status": index_status,
        "chunk_count": chunk_count,
    }


@app.post("/path")
def answer_question(payload: Question):
    if active_chain is None:
        raise HTTPException(status_code=409, detail="Upload a PDF before asking a question.")
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Enter a question first.")
    try:
        return {"answer": active_chain.invoke(payload.question)}
    except Exception as error:
        logger.exception("Question answering failed")
        raise HTTPException(
            status_code=503,
            detail="Could not answer the question. Check that the language-model service is available.",
        ) from error
