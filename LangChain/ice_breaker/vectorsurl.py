from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import VectorDBQA, OpenAI
from langchain.vectorstores import Pinecone
import pinecone
pinecone.init(api_key="65659b6f-62e1-4785-ae3b-6be4a360c291", environment="us-west4-gcp")
from langchain.document_loaders import UnstructuredURLLoader,WebBaseLoader
from langchain import HuggingFaceHub

repo_id = "gpt2-large" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

hf_llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.2, "max_length":64}, huggingfacehub_api_token = "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")

urls = [
 "https://atxanimalclinic.com/"
]

# if __name__ == "__main__":
loader = WebBaseLoader(urls)
document = loader.load()
print(document)
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(document)
print(len(texts))

embeddings = OpenAIEmbeddings(openai_api_key = "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")
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
docsearch = Pinecone.from_documents(texts,embeddings,index_name="open-ai-index")

# qa = VectorDBQA.from_chain_type(llm=hf_llm, chain_type="stuff",vectorstore=docsearch)

qa = VectorDBQA.from_chain_type(llm=OpenAI(openai_api_key = "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",model_name="curie:ft-mcsquared-2023-05-05-17-09-02"), chain_type="stuff",vectorstore=docsearch)
query = "What are the services provided by the each hospital"
result = qa({"query":query})
print(result)