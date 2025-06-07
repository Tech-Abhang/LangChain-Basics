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

##load data fro LLM
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

## create index with data loaded
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

## init LLM
llm = ChatOpenAI(temperature = 0.0, model=llm_model)

## create RetrievalQA chain with index for retriever and LLM
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

## generate QnA pair from documents
from langchain.evaluation.qa import QAGenerateChain

##create a chain
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))

## generate QnA pairs as well parsed in dict so that we dont get str
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)

## Manual eval
import langchain
langchain.debug = True

qa.run(examples[0]["query"])
langchain.debug = False

## LLM eval

##prediction for each QnA pair
predictions = qa.apply(examples)

##import LLM for eval
from langchain.evaluation.qa import QAEvalChain

##init LLM
llm = ChatOpenAI(temperature=0, model=llm_model)

#create chain using LLM
eval_chain = QAEvalChain.from_llm(llm)

##grade each prediction
graded_outputs = eval_chain.evaluate(examples, predictions)

##Check the results
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()