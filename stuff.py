from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "TheBloke/CodeLlama-13B-Instruct-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
   model_id,
   torch_dtype=torch.float16,
   device_map="auto",
)