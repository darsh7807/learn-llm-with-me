import  urllib.request
import re
import tiktoken

class SimpleTokeniserV1:
  def __init__(self, vocab):
    self.str_to_int = vocab
    self.int_to_str= {i:s for s, i in vocab.items()}

  def encode(self, text):
    preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    id =[self.str_to_int[i] for i in preprocessed]
    return id


  def decode(self, ids):
    text = " ".join([self.int_to_str[i] for i in ids])
    return text



# ids = tokeniser.encode(text)
# print(len(ids))

class SimpleTokeniserV1:
  def __init__(self, vocab):
    self.str_to_int = vocab
    self.int_to_str= {i:s for s, i in vocab.items()}

  def encode(self, text):
    # Corrected regex to match the one used for vocabulary creation
    preprocessed = re.split(r'([,.:;?_!"()\\]|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    ids =[self.str_to_int[i] for i in preprocessed]
    return ids


  def decode(self, ids):
    text = " ".join([self.int_to_str[i] for i in ids])
    return text


class SimpleTokeniserV2:
  def __init__(self, vocab):
    self.str_to_int = vocab
    self.int_to_str= {i:s for s, i in vocab.items()}

  def encode(self, text):
    # Corrected regex to match the one used for vocabulary creation (using \' for single quote)
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    # Corrected the unknown token string from "|<unk>|" to "<|unk|>" and fixed typo 'itme' to 'item'
    preprocessed = [ item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
    ids =[self.str_to_int[i] for i in preprocessed]
    return ids


  def decode(self, ids):
    text = " ".join([self.int_to_str[i] for i in ids])
    return text

import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
  def __init__(self, txt, tokeniser, max_length, stride):
    self.input_ids = []
    self.target_ids = []
    token_ids = tokeniser.encode(txt)
    for i in range (0, len(token_ids) - max_length, stride):
      input_chunk = token_ids[i:i+max_length]
      target_chunk = token_ids[i+1:i+max_length+1]
      self.input_ids.append(torch.tensor(input_chunk))
      self.target_ids.append(torch.tensor(target_chunk))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=256,
                        shuffle=True, drop_last=True,
                        num_workers=0
                        ):
  tokeniser = tiktoken.get_encoding("gpt2")
  dataset = GPTDatasetV1(txt, tokeniser, max_length, stride)
  dataloader = DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      drop_last=drop_last,
      num_workers=num_workers
  )
  return dataloader
