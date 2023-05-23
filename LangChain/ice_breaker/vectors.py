from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import VectorDBQA, OpenAI
from langchain.vectorstores import Pinecone
import pinecone
from langchain import HuggingFaceHub


pinecone.init(api_key="65659b6f-62e1-4785-ae3b-6be4a360c291", environment="us-west4-gcp")

# if __name__ == "__main__":
loader = TextLoader("E:\Courses\LangChain\ice_breaker\Mediumblogs\mediumblog.txt",encoding='utf8')
document = loader.load()
# print(document)
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(document)
# print(len(texts))

repo_id = "gpt2-large" 
hf_llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.2}, huggingfacehub_api_token = "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")

# embeddings = OpenAIEmbeddings(openai_api_key = "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")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
docsearch = Pinecone.from_documents(texts,embeddings,index_name="hugging-face-index")
qa = VectorDBQA.from_chain_type(llm=hf_llm, chain_type="stuff",vectorstore=docsearch)
# qa = VectorDBQA.from_chain_type(llm=OpenAI(openai_api_key = "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"), chain_type="stuff",vectorstore=docsearch)
query = "Who is hunting in the novel"
result = qa({"query":query})
print(result)




