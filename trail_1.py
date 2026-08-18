#use your groq api key
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableParallel,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage,AIMessage,SystemMessage

db=[]
def youtube_video_summarizer():
    link=input("enter youtube link:: ")
    ids=link.split("=")
    a=ids[1].split("&")
    b=a[0]
    db.append(b)
    return b

video_id=youtube_video_summarizer()


llm=ChatGroq(
      model="openai/gpt-oss-120b",
            temperature=1
)
embeddings=OllamaEmbeddings(
     model="qwen3-embedding:0.6b"
)
parser=StrOutputParser()

store=Chroma(
    embedding_function=embeddings,
    collection_name="youtube_transcripts",
    persist_directory="./"
)

api_instance = YouTubeTranscriptApi()


transcript_list = api_instance.fetch(video_id=video_id,languages=['en'])
transcript=""
for text in transcript_list:
    transcript+=text.text



splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)




chunk=splitter.create_documents([transcript])
store.from_documents(chunk,embeddings)

def context(context_docs):
    st="\n\n".join(doc.page_content for doc in chunk)
    return st
#reteriever :: for context

reteriever=store.as_retriever(search_type="similarity",search_kwargs={"k":4})
print(reteriever)




# # prompt generation 

prompt=PromptTemplate(
    template="""You are a helpful assistance , provide information about ONLY from the provide context, if you have inufficient context , just say i dont know, {context} , question :{question}""",
    input_variables=['context','question']
)

parallel_chain=RunnableParallel(
    {
        "context":reteriever|RunnableLambda(context),
        "question":RunnablePassthrough()
    }
)

seq=RunnableSequence(
    parallel_chain,prompt,llm,parser
)
chat:list[BaseMessage]=[SystemMessage(content="hello")]
payload=[]
while True:
    user_prompt=input("Hii, what would you like to ask:: ")
    for i in chat:
        payload.append(chat[-3:])
    if user_prompt=="exit":
        break
    response=seq.invoke(user_prompt)
    payload.append(AIMessage(response))
    print(response)


