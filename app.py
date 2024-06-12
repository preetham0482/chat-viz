import chainlit as cl
import pandas as pd
import textwrap
from stuff import *

import warnings
warnings.filterwarnings("ignore")

import sqlite3
con = sqlite3.connect('/home/ppathi/gold.db')
cur = con.cursor()

import logging

def execute_sql_query(query):
    """
    Executes an SQL query using the cursor and returns a Pandas DataFrame.
    """
    cur.execute(query)
    rows = cur.fetchall()
    columns = [description[0] for description in cur.description]
    return pd.DataFrame(rows, columns=columns)

def generate_code(input_text):
    # Generate SQL Query
    schema = str(cur.execute("pragma table_info('stock');").fetchall())

    chat = [
        {"role": "system", "content": "You are a helpful and honest code assistant expert in SQL (sqlite3 engine). \
         Please, provide just a single SQL (sqlite3) query answering the question without any text using the \
         given Schema (table name: stock): "+schema+". Only the columns in the schema can be used. Any other columns would \
         result in an error.  Since this data will be used for visualization, \
         be sure to include the index or date or other relevant columns in the table.\
            I WANT NO EXTRA TEXT. THIS QUERY WILL BE DIRECTLY EXECTUED ON THE DATABASE."},
        {"role": "user", "content": 'visualize '+ input_text},
    ]
    inputs = tokenizer.apply_chat_template(chat, return_tensors="pt").to("cuda")
    output = model.generate(input_ids=inputs, max_new_tokens=200)
    result_sql = output[0].to("cpu")
    result_sql = tokenizer.decode(result_sql, skip_special_tokens=True)
    result_sql = result_sql.split('[/INST]')[1][1:]

    # log the SQL query
    logging.info(result_sql)

    # df = execute_sql_query(result_sql)
    # df.to_csv('df.csv', index=False)

    # Generate Python Code for Visualization
    # chat = [
    #     {"role": "system", "content": "You are a helpful and honest code assistant expert in Python. \
    #      Please, provide just the Python code answering the question without any extra text \
    #      (not even comments or unwanted symbols) using the \
    #      given columns of df (df.csv): "+str(df.columns)+". Just look at the columns and give the code to visualize. \
    #      Don't do ANY operations on the data because the df is the output of an SQL query that's already custom to this query. \
    #      If you're parsing the date column, it is in 'YYYY-MM-DD' format.\
    #      Each row is a day in the market. Do not blindly use plt.plot. You should choose the type of visualization depending on the data. \
    #      ### Save the visualization as 'viz.png'.\
    #      I WANT NO EXTRA TEXT. THIS CODE WILL BE DIRECTLY EXECTUED."},
    #     {"role": "user", "content": 'visualize '+ input_text},
    # ]
    # inputs = tokenizer.apply_chat_template(chat, return_tensors="pt").to("cuda")
    # output = model.generate(input_ids=inputs, max_new_tokens=200)
    # result_python = output[0].to("cpu")
    # result_python = tokenizer.decode(result_python, skip_special_tokens=True)
    # result_python = textwrap.dedent(result_python.split('[/INST]')[1].replace('```',''))
    # logging.info(result_python)
    # try:
    #     exec(result_python)
    #     return True
    # except Exception as e:
    #     logging.error(str(e))
    #     return False
    return result_sql

@cl.on_chat_start
async def start():
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Visualization Bot. What is your query?"
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    input_text = message.content
    status = generate_code(input_text)
    # if status:
    #     image = cl.Image(path="./viz.png", name="image1", display="inline")
    #     # Attach the image to the message
    #     await cl.Message(
    #         content="Here's the visualization you requested:",
    #         elements=[image],
    #     ).send()
    # else:
    #     await cl.Message(content="Sorry, I couldn't generate the visualization. Please try again.").send()
    await cl.Message(content=status).send()