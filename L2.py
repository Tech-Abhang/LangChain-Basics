##Memory

##import
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import warnings
warnings.filterwarnings('ignore')

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

## 1 . conversation buffer memory(limited space to store conversation)
from langchain.memory import ConversationBufferMemory

##helper function
llm = ChatOpenAI(temperature=0.0, model=llm_model)

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

##call for response
conversation.predict(input="Hi, my name is Andrew")

##check whats in memory
print(memory.buffer)
memory.load_memory_variables({})

##add some contexxt to memory
memory.save_context({"input": "Hi"}, 
                    {"output": "What's up"})


## 2 . conversation buffer window memory(stores conversation with value k = no.of conversation exchanges)
from langchain.memory import ConversationBufferWindowMemory

##helper function
llm = ChatOpenAI(temperature=0.0, model=llm_model)

memory = ConversationBufferWindowMemory(k=1) ##k=1 means only last message is stored

conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=False
)

##call for response
conversation.predict(input="Hi, my name is Andrew")

## 3 . Conversation token buffer memory(maps memory with tokens)

from langchain.memory import ConversationTokenBufferMemory

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)  ## max_token_limit is the maximum number of tokens to store in memory


## 4 . Conversation summary memory(summarizes conversation and stores it in memory less space usage)

from langchain.memory import ConversationSummaryBufferMemory

##demo message
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100) ##if enough token number it stores complete message otherwise it summarizes the message.

##helper function
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

##call for response
conversation.predict(input="What would be a good demo to show?")

##check whats in memory
print(memory.buffer)
memory.load_memory_variables({})