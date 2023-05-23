from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain import HuggingFaceHub

llm = OpenAI(model_name='text-davinci-003',temperature=0,openai_api_key = "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")

repo_id = "mosaicml/mpt-7b" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

hf_llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.2}, huggingfacehub_api_token = "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")
conversation = ConversationChain(
    llm=llm, verbose=True
)
memory=ConversationSummaryMemory(llm=llm)
conversation = ConversationChain(llm=llm)
x= input()
while(x != "bye"):
    response = conversation.predict(input=x)
    print(response)
    x= input()
    
print(conversation.memory.buffer)
