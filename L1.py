## Models , Prompts , Parsers

##imports 
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

##helper functions
def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message["content"]

##call for responce
response = get_completion(prompt)


##Models
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(temperature=0.0, model=llm_model)


##Prompts

##demo prompt
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

##prompt template
prompt_template = ChatPromptTemplate.from_template(template_string)

##Better prompt 
customer_style = """American English \
in a calm and respectful tone
"""
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

##customized prompt
customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)

##call LLM with better prompt
customer_response = chat(customer_messages)


##Parser