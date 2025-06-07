## Chains

## imports
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import pandas as pd
df = pd.read_csv('Data.csv')

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromp

## chains
from langchain.chains import LLMChain

## init llm
llm = ChatOpenAI(temperature=0.9, model=llm_model)

## prompt template with variable product
prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

## reated chain
chain = LLMChain(llm=llm, prompt=prompt)

##run the chain with prompt and llm using variable value 
product = "Queen Size Sheet Set"
chain.run(product)

## Simple Sequencial Chains (output of one chain is input to another chain) 
## Mainly used when single input single output

## imports
from langchain.chains import SimpleSequentialChain
llm = ChatOpenAI(temperature=0.9, model=llm_model)

## chain 1 (variable product)
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

chain_one = LLMChain(llm=llm, prompt=first_prompt)

## chain 2 (variable company_name comming from chain 1)
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
    company:{company_name}"
)

chain_two = LLMChain(llm=llm, prompt=second_prompt)

## overall chain 
overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                             verbose=True
                                            )

##run the overall chain with variable value
product = "Queen Size Sheet Set"
overall_simple_chain.run(product)

## Sequencial Chains (multiple inputs and outputs)

##imports
from langchain.chains import SequentialChain

## init LLM
llm = ChatOpenAI(temperature=0.9, model=llm_model)

## chain 1 (input= Review and output= English_Review)
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)

chain_one = LLMChain(llm=llm, prompt=first_prompt, 
                     output_key="English_Review"
                    )

## chain 2 (input= English_Review and output= summary)
second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)

chain_two = LLMChain(llm=llm, prompt=second_prompt, 
                     output_key="summary"
                    )

## chain 3 (input= Review and output= language)
# prompt template 3: translate to english
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)

chain_three = LLMChain(llm=llm, prompt=third_prompt,
                       output_key="language"
                      )

## overall chain
# overall_chain: input= Review 
# and output= English_Review,summary, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three],
    input_variables=["Review"],
    output_variables=["English_Review", "summary","language"],
    verbose=True
)

## run with data from Data.csv
review = df.Review[5]
overall_chain(review)


## Router Chains (more complex chains mainly for chain containing sub-chains)


