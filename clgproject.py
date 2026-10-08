from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel,RunnableSequence
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pathlib import Path
from hashlib import sha256

load_dotenv()

persist_directory = Path(__file__).parent / "chroma_db"

loader=PyPDFLoader(r"C:\Users\ongraph\Downloads\atonement.pdf")
cursor=loader.load()
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
embedder=OllamaEmbeddings( model="qwen3-embedding:0.6b",
    dimensions=1000)
store=Chroma(embedding_function=embedder,persist_directory="./")

# splitter.create_documents(cursor.page_content)

def context(cont):
    if not cont:
        return "No relevant passages were retrieved."
    return "\n\n".join(doc.page_content for doc in cont) 

# chunk=splitter.create_documents([context(cursor)])
# store.add_documents(chunk)


retreiver=store.as_retriever(search_type="similarity",kwargs={"k":4})

chain=RunnableParallel({"context":retreiver|RunnableLambda(context),
                        "question":RunnablePassthrough()})

seq=RunnableSequence(chain,prompt,llm,parser)

# seq.get_graph().print_ascii()
print(seq.invoke("give a character sketch of ceceilia"))

# print(chain.invoke("show me your context"))
