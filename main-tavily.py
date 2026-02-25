import os
from unittest import result

from dotenv import load_dotenv

load_dotenv()
from langchain import agents
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """Tool that searches over internet
    Args:
        query: the query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    # return "Tokyo weather is sunny"
    return tavily.search(query=query)

llm = ChatOpenAI(
#     # model="arcee-ai/trinity-large-preview:free",
    model="openrouter/free",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)
# llm = ChatOpenAI()

tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    # result = agent.invoke({"messages": HumanMessage(content="what is the weather in Tokyo")});
    result = agent.invoke({"messages": HumanMessage(content=
    "search for 3 job postings for an software engineer using langchain in the hong kong on linkedin and list their details")});
    print(result)

if __name__ == "__main__":
    main()
