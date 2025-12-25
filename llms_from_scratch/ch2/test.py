
from tokenisation import create_dataloader_v1,SimpleTokeniserV2
import torch
import  urllib.request
import re
url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
file_path="the-verdict.txt"
urllib.request.urlretrieve(url, file_path)
with open("the-verdict.txt", "r", encoding="utf-8") as f:
  raw_text = f.read()

hindi_url=("https://raw.githubusercontent.com/gayatrivenugopal/"
            "Hindi-Aesthetics-Corpus/refs/heads/master/Corpus/duraga-ka-mandir.txt")
file_path="duraga-ka-mandir.txt"
urllib.request.urlretrieve(hindi_url, file_path)



with open("duraga-ka-mandir.txt", "r", encoding="utf-8") as f:
  raw_hindi_text = f.read()

print("Total num of chars: ", len(raw_text))
print(raw_text[:99])


print("Total num of chars: ", len(raw_hindi_text))
print(raw_hindi_text[:99])

import re
text = "hello, world. This, is test."
result = re.split(r'(\s)', text)
print(result)
print(len(result))
result = re.split(r'([,.]\s)', text)
print(result)
print(len(result))

result = [item for item in result if item.strip()]
print(result)



text = "hello, world, Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
print(result)
result = [item.strip() for item in result if item.strip()]
print(result)



preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

all_words = sorted(set(preprocessed))
vocab_size=len(all_words)
print(vocab_size)

vocab = {token:integer  for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
  print(item)
  if i>5:
    break

text1 = "hello, do you like tea?"


preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>","<|unk|>"])
vocab_size=len(all_words)
print(vocab_size)
vocab = {token:integer  for integer, token in enumerate(all_words)}



with open("the-verdict.txt", "r", encoding="utf-8") as f:
  raw_text = f.read()

dataloader = create_dataloader_v1(raw_text,
                                 batch_size=1,
                                 max_length=4,
                                 stride=1,
                                 shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
second_batch=next(data_iter)
print(second_batch)

dataloader = create_dataloader_v1(raw_text,
                                 batch_size=8,
                                 max_length=4,
                                 stride=4,
                                 shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch[0])
print(first_batch[1])

torch.manual_seed(123)
output_dim=3
vocab_size=6
embedding_layer=torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)


torch.manual_seed(123)
output_dim=256
vocab_size=50257
token_embedding_layer=torch.nn.Embedding(vocab_size, output_dim)
max_length=4
dataloader = create_dataloader_v1(raw_text,
                                 batch_size=8,
                                 max_length=max_length,
                                 stride=max_length,
                                 shuffle=False)
data_iter = iter(dataloader)
input, targets = next(data_iter)
token_embedding = token_embedding_layer(input)
print(token_embedding.shape)



context_length = max_length
pos_embedding_layer=torch.nn.Embedding(context_length, output_dim)
pos_embedding = pos_embedding_layer(torch.arange(context_length))
print(pos_embedding.shape)

input_embeddings = token_embedding + pos_embedding
print(input_embeddings.shape)
