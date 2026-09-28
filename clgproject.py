from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pathlib import Path
from hashlib import sha256

load_dotenv()

persist_directory = Path(__file__).parent / "chroma_db"

llm=ChatGroq(
     model="openai/gpt-oss-120b",
                temperature=1
)
prompt=PromptTemplate(
    template="""Answer only from the text inside <context>.
" Do not use outside knowledge.

<context>
{context}
</context>

Question: {question}""",
    input_variables=["context","question"]
)
parser=StrOutputParser()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""],
)

def context(cont):
    if not cont:
        return "No relevant passages were retrieved."
    return "\n\n".join(doc.page_content for doc in cont)

def create_chain(pdf_path: Path):
    collection_name = f"pdf_{sha256(pdf_path.read_bytes()).hexdigest()[:16]}"
    embedder = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    store = Chroma(
        embedding_function=embedder,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )

    if store._collection.count() == 0:
        pages = PyPDFLoader(str(pdf_path)).load()
        chunks = splitter.split_documents(pages)
        if not chunks:
            raise ValueError("The PDF does not contain readable text.")
        store.add_documents(chunks)

    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return (
        {
            "context": retriever | RunnableLambda(context),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | parser
    )

# chain.get_graph().print_ascii()

# # print(run.invoke("show me your context"))
# print(chain.invoke("give me the most interesting story"))
# print(chain.invoke("show me your context"))
