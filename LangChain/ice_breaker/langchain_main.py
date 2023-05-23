import langchain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import LLMChain
from tools.tools import get_entity_url,location_finder,website_data_info
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor,ConversationalChatAgent,ConversationalAgent
from langchain.memory import ConversationBufferMemory,ConversationSummaryMemory,ConversationBufferWindowMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper
from output_parsers import test
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.schema import BaseOutputParser
from langchain.agents.conversational_chat.prompt import (
    FORMAT_INSTRUCTIONS,
    PREFIX,
    SUFFIX,
    TEMPLATE_TOOL_RESPONSE,
)
from typing import Any, List, Optional, Sequence, Tuple, Dict
import json
import re


llm = ChatOpenAI(temperature=0.2, openai_api_key = "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",model_name="text-curie-001")

conversational_memory = ConversationBufferWindowMemory(
memory_key='chat_history',
k=20,
return_messages=True
)



tools_for_agent = [Tool(name="use gmaps to get the location",func=location_finder, description="useful to find the locations based on the information provided"),
                    Tool(name="Crawl Google for to get the information about the website url",func=get_entity_url, description="useful for when you need to get website url"),
                    Tool(name="Use this tool after get_entity_url tool, get the url for the entities asked by the user and pass it into this function to get the relevant information",func=website_data_info, description="Use this tool to get the information about a particular service after you get the website URL, it could be multiple urls")]

agent_chain = initialize_agent(tools_for_agent, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory=conversational_memory,verbose=True)
# agent = ConversationalChatAgent.from_llm_and_tools(
#     llm=llm, tools=tools_for_agent, memory=conversational_memory, verbose=True)
# agent_chain = AgentExecutor.from_agent_and_tools(
#     agent=agent,
#     tools=tools_for_agent,
#     memory=conversational_memory,
#     verbose=True
# )
print("Start asking questions /n")
x= input()
while(x != "bye"):
    print(agent_chain.run(input=x))
    x= input()
# print(conversational_memory.memory.buffer)