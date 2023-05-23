from __future__ import annotations
import langchain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType,LLMSingleActionAgent,AgentOutputParser
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain
from tools.tools import get_entity_url,location_finder,website_data_info,search
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor,ConversationalChatAgent
from langchain.memory import ConversationBufferMemory,ConversationSummaryMemory,ConversationBufferWindowMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
import re
from typing import List, Union
import json

from typing import Any

from langchain.chains.llm import LLMChain
from langchain.output_parsers.prompts import NAIVE_FIX_PROMPT
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseOutputParser, OutputParserException
from langchain.output_parsers import PydanticOutputParser





llm = OpenAI(temperature=0.2, model_name="text-curie-001", openai_api_key = "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")



tools_for_agent = [Tool(name="use gmaps to get the location",func=location_finder, description="useful to find the locations based on the information provided"),
                    Tool(name="Crawl Google for to get the information about the website url",func=get_entity_url, description="useful for when you need to get website url"),
                    Tool(name="Use this tool after get_entity_url tool, get the url for the entities asked by the user and pass it into this function to get the relevant information",func=website_data_info, description="Use this tool to get the information about a particular service after you get the website URL, it could be multiple urls"),
                    Tool(name="Crawl Google for generic search",func=search, description="Useful to get the generic information for seach related queries")]
prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools_for_agent, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "chat_history", "agent_scratchpad"]
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools_for_agent, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools_for_agent, verbose=True, memory=memory,output_parser=PydanticOutputParser)
agent_chain.run(input="Hello")