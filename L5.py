## Evaluation LLM povered application

##imports
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearc