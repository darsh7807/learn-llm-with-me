
import torch
import tiktoken

from gptmodel import GPT_CONFIG_124M, GPTModel,generate_text_simple


torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")
start_context="hello I am"
encoded = tokenizer.encode(start_context)
encoded_tensor=torch.tensor(encoded).unsqueeze(0)

GPT_CONFIG_124M["num_heads"] = GPT_CONFIG_124M["n_heads"]
model=GPTModel(GPT_CONFIG_124M)
model.eval()
out = generate_text_simple(model, encoded_tensor, 4, context_size=GPT_CONFIG_124M["context_length"])
print(out)
decoded_text=tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
