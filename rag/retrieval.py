from multiprocessing import context
import os
from idlelib.searchengine import search_reverse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_classic.chains.summarize.map_reduce_prompt import prompt_template
from langchain_classic.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

from operator import itemgetter


load_dotenv()

print("initializing components...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

MODEL = "qwen3:1.7b"
llm = init_chat_model(f"ollama:{MODEL}", temperature=0)

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a string for the LLM"""
    return "\n\n".join([doc.page_content for doc in docs])


def retrieve_without_langchain(query):
    """Simple retrieval without LangChain
    Manually retrieve the documents, format them and generates a response.
    Limitation:
    - Manual step by step execution.
    - No built-in stream support.
    - No async Support without adding custom code.
    - Harder to compose with other chains
    - More verbose and error prone.
    """
    # 1. Retrieve the documents (Pass the string, not a list)
    docs = retriever.invoke(query)
    # 2. Prepare the context string
    context_text = format_docs(docs)
    # 3. Format the prompt with context and question
    messages = prompt_template.format_messages(context=context_text, question=query)
    # 4. Invoke the LLM
    response = llm.invoke(messages)
    return response.content


def retrieve_using_llm_only_no_rag(query):
    response = llm.invoke([HumanMessage(content=query)])
    print("================================================")
    print(response.content)
    print("================================================")


def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
   
    retrieval_chain =(
        RunnablePassthrough.assign(context= itemgetter( "question") | retriever | format_docs)
        | prompt_template 
        | llm 
        | StrOutputParser()
    )
    return retrieval_chain



if __name__ == "__main__":
    print("Running query...")
    query = "What is pinecone in machine learning?"

    print("Option 1: No RAG")
    retrieve_using_llm_only_no_rag(query)

    print ("=" + "="*70)
    print("Option 2: Retrieval without LangChain")
    response = retrieve_without_langchain(query)
    print(response)

    print ("=" + "="*70)
    print("Option 3: Retrieval using LangChain - Better Approach")
    print ("=" + "="*70)
    print ("=" + "="*70)

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)

   
