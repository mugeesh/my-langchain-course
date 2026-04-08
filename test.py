import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

print(PINECONE_API_KEY)

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("mediumblog1.txt", encoding="UTF-8")
    documents = loader.load()
    # print(documents)
    print("splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(documents)
    print(f"created {len(text)} chunks")

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        # Add these specific parameters to satisfy OpenRouter's requirements
        model_kwargs={"encoding_format": "float"},
        headers={
            "HTTP-Referer": "http://localhost:3000",  # OpenRouter requires a referer
            "X-Title": "LangChain Course",
        },
    )

    print(f"Using embedding model: {embeddings.model}")

    print("ingesting...")

    PineconeVectorStore.from_documents(
        text, embeddings, index_name=os.environ["INDEX_NAME_OPENAPI"]
    )
    print("finished")
