import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings  # Updated Import
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("mediumblog1.txt", encoding="UTF-8")
    documents = loader.load()

    print("splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(documents)
    print(f"created {len(text)} chunks")

    # This model runs locally on your machine for FREE
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print(f"Using local embedding model: {embeddings.model}")
    print("ingesting to Pinecone...")

    # Ensure this matches your Pinecone Index name exactly
    index_name = os.environ["INDEX_NAME"]

    try:
        vectorstore = PineconeVectorStore.from_documents(
            text, embeddings, index_name=index_name
        )
        print("✅ Successfully finished ingesting to Pinecone using Ollama!")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
