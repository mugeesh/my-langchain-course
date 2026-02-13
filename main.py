import os
from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI 
from langchain_ollama import ChatOllama

# Load environment variables from .env
load_dotenv(find_dotenv(), override=True)

def main():
    # Fetch key from environment
    openai_api_key = os.environ.get("OPENROUTER_API_KEY")
    
    print("Hello from langchain-course!")
    
    if not openai_api_key:
        print("Error: OPENROUTER_API_KEY not found in .env file.")
        return
    
    print(f"Key loaded: {openai_api_key[:10]}...")

    # The raw data to be processed
    information = """
        Elon Reeve Musk (born June 28, 1971) is a businessman and entrepreneur known for his 
        leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person 
        in the world since 2025; as of February 2026, Forbes estimates his net worth to 
        be around US$852 billion. Born in Pretoria, South Africa, he co-founded Zip2, 
        PayPal, and later led SpaceX and Tesla to industry dominance. In 2025, he served 
        briefly in the Trump administration before returning to his private ventures.
    """

    # DEFINING THE TEMPLATE
    # Note: No 'f' before the triple quotes. 
    # {information} is a placeholder that LangChain fills.
    summary_template = """
        Given the following information about a person:
        
        {information}
        
        Please create:
        1. A short summary (3 sentences max)
        2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], 
        template=summary_template
    )

    # INITIALIZING THE MODEL
    # Using 'google/gemini-2.0-flash-001:free' - currently the most stable free ID
    # llm = ChatOpenAI(
    #     # model="arcee-ai/trinity-large-preview:free",
    #     model="openrouter/free",
    #     openai_api_key=openai_api_key,
    #     openai_api_base="https://openrouter.ai/api/v1",
    #     temperature=0
    # )

    llm = ChatOllama(
        model="gemma3:270m",
        temperature=0
    )
    #create keey https://openrouter.ai/settings/keys

    # CREATING THE CHAIN
    chain = summary_prompt_template | llm
    
    print("\n--- Generating Summary ---\n")
    
    try:
        # Pass the 'information' variable into the dictionary
        res = chain.invoke(input={"information": information})
        print(res.content)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nTip: If you see a 404, OpenRouter may have renamed the free model.")
        print("Check https://openrouter.ai/models?max_price=0 for the latest ID.")

if __name__ == "__main__":
    main()