from transformers import AutoTokenizer, AutoModelForCausalLM
import chainlit as cl

model_name = 'TheBloke/Llama-2-7B-Chat-GPTQ'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto'
)

def get_reply(input):
    prompt_template=f'''[INST]<<SYS>>
    You are a helpful assistant. Always answer as helpfully as possible, while being safe.  
    Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
    Please ensure that your responses are socially unbiased and positive in nature. 
    If a question does not make any sense, or is not factually coherent, explain why instead of answering 
    something not correct. If you don't know the answer to a question, please don't share false information.
    Answer the user query delimited by ```,
    <</SYS>>
    ```{input}``` [/INST]
'''
    tokens = tokenizer(prompt_template,return_tensors='pt').input_ids.cuda()
    response = model.generate(input_ids=tokens, max_new_tokens=100, top_k=40)
    return tokenizer.decode(response[0]).split('[/INST]')[1]


@cl.on_chat_start
async def start():
    await cl.Message(content='Hi! This is Botty. What\'s up?').send()

@cl.on_message
async def main(message:cl.Message):
    reply = get_reply(message.content)
    await cl.Message(content=reply).send()