from langchain.serpapi import SerpAPIWrapper
from serpapi import GoogleSearch
from langchain.utilities import GoogleSearchAPIWrapper

import geocoder 
import googlemaps
from langchain.vectorstores import Pinecone
import pinecone
pinecone.init(api_key="65659b6f-62e1-4785-ae3b-6be4a360c291", environment="us-west4-gcp")
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import VectorDBQA, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


def get_entity_url(text):
    texts = text.split(",")
    urls = []
    for i in texts:
        params = {
            "q": i,
            "google_domain": "google.com",
            "api_key": "f321362dfd170f70173d543ddc52f4faf85c7dc11b411d6ae9957848b90a4e5a"
            }
        search = GoogleSearch(params)
        result= search.get_dict()
        if result.get("knowledge_graph",None) is None:
            urls.append(None)
        else:
            urls.append(result.get("knowledge_graph",None).get('website',None))
    res = [i for i in urls if i is not None]
    return res


def search(text):
    search = GoogleSearchAPIWrapper(serpapi_api_key="f321362dfd170f70173d543ddc52f4faf85c7dc11b411d6ae9957848b90a4e5a")
    res = search.run(f"{text}")
    return res
    
def location_finder(text):

    # Get the location information 
    g = geocoder.ip('me') 

    # Get the location of the user 
    location = g.latlng  
    
    print(location)


    # API key
    api_key = 'AIzaSyA1WLfZuDVgGp12qieXj89_9J_Tg0T_Y48'

    # Create a gmaps object
    gmaps = googlemaps.Client(key=api_key)
    result_list = []


    # Search for nearby pet hospitals
    nearby_pet_hospitals = gmaps.places_nearby(location=(location[0], location[1]), 
                                            keyword=text,
                                            radius = 4000)

    # Print the names of the pet hospitals
    for place in nearby_pet_hospitals['results']:
        result_list.append(place['name'])
    
    return result_list


def website_data_info(text):
    texts = text.split(",")
    urls = []
    for i in texts:
        params = {
            "q": i,
            "google_domain": "google.com",
            "api_key": "f321362dfd170f70173d543ddc52f4faf85c7dc11b411d6ae9957848b90a4e5a"
            }
        search = GoogleSearch(params)
        result= search.get_dict()
        if result.get("knowledge_graph",None) is None:
            urls.append(None)
        else:
            urls.append(result.get("knowledge_graph",None).get('website',None))
    res = [i for i in urls if i is not None]
    loader = WebBaseLoader(res)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings(openai_api_key = "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")
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo", openai_api_key = "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")
    docsearch = Pinecone.from_documents(texts,embeddings,index_name="open-ai-index")
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever())
    return qa

print(get_entity_url("Affordable Pet Protection"))
# print(location_finder("HEB stores near me"))
