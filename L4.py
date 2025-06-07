## QnA , Embedding , Retrieval , Vector DB 

## imports
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.llms import OpenAI

## load data for LLM
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

## create index
from langchain.indexes import VectorstoreIndexCreator

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

## response from LLM
query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

llm_replacement_model = OpenAI(temperature=0, 
                               model='gpt-3.5-turbo-instruct')

response = index.query(query, 
                       llm = llm_replacement_model)

## or (use indexes to llm : uses complete data , use retrival to llm : uses only relevant data)

## text takes up a lot of space for that we use embeddings(text to vector)

## Large Dataset -> split into chunks -> Embedding -> Vector DB

## Step by Step

## load data
from langchain.document_loaders import CSVLoader
loader = CSVLoader(file_path=file)

## data
docs = loader.load()

## docs is small no need of chunking direct embedding
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

## create embedding for whole data and also store in vector db ie DocArrayInMemorySearch here
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)

## query the vector db
query = "Please suggest a shirt with sunblocking"

## find similar data to query
docs = db.similarity_search(query)

## to give to llm we use retrival
retriever = db.as_retriever()

llm = ChatOpenAI(temperature = 0.0, model=llm_model)

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

## response
response = qa_stuff.run(query)
display(Markdown(response))