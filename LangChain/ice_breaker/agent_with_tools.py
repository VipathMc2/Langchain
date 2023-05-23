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
    
    llm = OpenAI(temperature=0.2, openai_api_key = "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")
    tools = [Tool(name="use gmaps to get the location",func=location_finder, description="useful to find the locations based on the information provided"),
                        Tool(name="Crawl Google for to get the information about the website url",func=get_entity_url, description="useful for when you need to get website url"),
                        Tool(name="Use this tool after get_entity_url tool, get the url for the entities asked by the user and pass it into this function to get the relevant information",func=website_data_info, description="Use this tool to get the information about a particular service after you get the website URL, it could be multiple urls")]
    template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: you can use  [{tool_names}], if necessary or you can proceed without its help
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

            Previous conversation history:
            {history}

            New question: {input}
            {agent_scratchpad}""" 
    # Set up a prompt template
    class CustomPromptTemplate(StringPromptTemplate):
        # The template to use
        template: str
        # The list of tools available
        tools: List[Tool]
        
        def format(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            return self.template.format(**kwargs) 
        
    class CustomOutputParser(AgentOutputParser):
    
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            # Check if agent should finish
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
        

            
    prompt_with_history = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps", "history"]
            )
    
    llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)
    
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )
    
    memory=ConversationBufferWindowMemory(k=5)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
    x= input()
    while(x != "bye"):
        agent_executor.run(x)
        x= input()