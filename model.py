import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

con = sqlite3.connect('gold.db')
cur = con.cursor()

from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import chainlit as cl

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Write an SQL query to get the required information from the database. use the sqlite3 module (i have the curser named 'cur') 
to fetch the data. also, write the code to get a visualisation of the answer. The user would 
mention the type of visualisation they want to see. You can use the following libraries to visualise the data:
matplotlib, seaborn. 


sqlite table schema:
{schema}

Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['schema', 'question'])
    return prompt

# use the llama code model without any RAG. 
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = 'codellama/CodeLlama-13b-Instruct-hf',
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

def execute_sql_query(query):
    """
    Executes an SQL query using the cursor and returns a Pandas DataFrame.
    """
    cur.execute(query)
    rows = cur.fetchall()
    columns = [description[0] for description in cur.description]
    return pd.DataFrame(rows, columns=columns)

def generate_visualization(df, plot_type):
    """
    Generates a visualization of the dataframe based on the specified plot type.
    """
    if plot_type == 'bar':
        df.plot(kind='bar')
    elif plot_type == 'line':
        df.plot(kind='line')
    elif plot_type == 'scatter':
        df.plot(kind='scatter', x=df.columns[0], y=df.columns[1])
    # Add more plot types as needed
    plt.show()

# Function to process the question and generate SQL query and visualization
def process_question(schema, question):
    prompt = set_custom_prompt()
    llm = load_llm()

    # Generating the SQL query and visualization type
    query_and_viz_type = llm.generate(prompt.get_prompt({'schema': schema, 'question': question}))
    query, viz_type = query_and_viz_type.split('\n')

    # Execute the SQL query and get the DataFrame
    df = execute_sql_query(query)

    # Generate the visualization
    generate_visualization(df, viz_type)

# Example usage
schema = str(cur.execute("pragma table_info('stock');").fetchall())

# Chainlit code
@cl.on_chat_start
async def start():
    chain = load_llm()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Visualization Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 

    # Process the message using the loaded model
    # Adjust this part based on how your model processes input and generates output
    answer = chain.generate(message.content)

    await cl.Message(content=answer).send()