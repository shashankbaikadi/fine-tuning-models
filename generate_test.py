from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("./distilgpt2-finetuned")
model = AutoModelForCausalLM.from_pretrained("./distilgpt2-finetuned")

prompt = "User: who is shashu.\nAssistant:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

out = model.generate(input_ids, max_new_tokens=60, do_sample=True, top_p=0.1, temperature=0.1)
print(tokenizer.decode(out[0], skip_special_tokens=True))
