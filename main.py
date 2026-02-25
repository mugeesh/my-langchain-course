import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
from langchain import agents
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description="The URL of he source")

class AgentResponse(BaseModel):
    """Schema for agent response with answere and sources"""

    answer:str = Field(description="The agent's answere to the query")
    source: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")



llm = ChatOpenAI(
#     # model="arcee-ai/trinity-large-preview:free",
    model="openrouter/free",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)
# llm = ChatOpenAI()

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    # result = agent.invoke({"messages": HumanMessage(content="what is the weather in Tokyo")});
    result = agent.invoke({"messages": HumanMessage(content=
    "search for 3 job postings for an software engineer using langchain in the hong kong on linkedin and list their details")});
    print(result)

if __name__ == "__main__":
    main()
