from datasets import Dataset,load_dataset
import huggingface_hub
import json 
from transformers import (
  AutoTokenizer, 
)

huggingface_hub_token = "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"
huggingface_hub.login(token=huggingface_hub_token)

def preprocess_dataset(file):
  data=[]

  with open(file, "r", encoding="utf-8") as file:
      for line in file:
          # 从每行中加载JSON数据
          json_data = json.loads(line.strip())
          
          # 将加载的JSON数据添加到列表中
          data.append(json_data)
  new_data=[]

  for sample in data:
    new_diag=[]
    for index,diag in enumerate(sample["messages"]):
      if diag["role"]!="system":
        diag={"role":diag["role"],"content":diag["content"]}
          
        new_diag.append(diag)
    
    new_data.append({"messages": new_diag})
  return new_data

train_dataset = Dataset.from_list(preprocess_dataset("train.jsonl"))

test_dataset = Dataset.from_list(preprocess_dataset("test.jsonl"))
print(test_dataset)
validation_dataset = Dataset.from_list(preprocess_dataset("val.jsonl"))

base_model_name = "Sacralet/mistral-7B"

tokenizer = AutoTokenizer.from_pretrained(
  base_model_name, 
  trust_remote_code=True,
)

def convert_to_mistral_prompt(example):
  content = example["messages"]
  
  text = tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=False)

  return {"prompt": text}

train_dataset = train_dataset.map(convert_to_mistral_prompt)
validation_dataset=validation_dataset.map(convert_to_mistral_prompt)
test_dataset=test_dataset.map(convert_to_mistral_prompt)





train_dataset.push_to_hub("Sacralet/mistral_chat_nesting_dataset", private=False, token=huggingface_hub_token,split="train")
validation_dataset.push_to_hub("Sacralet/mistral_chat_nesting_dataset", private=False, token=huggingface_hub_token,split="validation")
test_dataset.push_to_hub("Sacralet/mistral_chat_nesting_dataset", private=False, token=huggingface_hub_token,split="test")
