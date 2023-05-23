import langchain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType,LLMSingleActionAgent,AgentOutputParser
from langchain.chains import LLMChain
from tools.tools import get_entity_url,location_finder,website_data_info
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor,ConversationalChatAgent
from langchain.memory import ConversationBufferMemory,ConversationSummaryMemory,ConversationBufferWindowMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
import re
from typing import List, Union


if __name__ == "__main__":
    
    class CustomOutputParser(AgentOutputParser):

        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            # Check if agent should finish
            print(llm_output)
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            # Parse out the action and action input
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
        
    output_parser = CustomOutputParser()  
    llm = OpenAI(temperature=0.2, openai_api_key = "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",model_name="text-curie-001")
    tools = [Tool(name="use gmaps to get the location",func=location_finder, description="useful to find the locations based on the information provided"),
                        Tool(name="Crawl Google for to get the information about the website url",func=get_entity_url, description="useful for when you need to get website url"),
                        Tool(name="Use this tool after get_entity_url tool, get the url for the entities asked by the user and pass it into this function to get the relevant information",func=website_data_info, description="Use this tool to get the information about a particular service after you get the website URL, it could be multiple urls")]
    
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory,AgentOutputParser=output_parser)
    print("Start asking the questions")
    x= input()
    while(x != "bye"):
        print(agent_chain.run(input=x))
        x= input()
    