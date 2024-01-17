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
  system_message="You are a nesting expert with a mission to translate the polygons for maximum utilization on the surface while maintaining minimal spacing between them. It's critical to avoid overlap and ensure that the polygons stay within the surface's boundaries.\nThe user will provide the size of surface and  polygons one by one in the format of polygonN(\"x1,y1\",\"x2,y2,\"x3,y3\",\"x4,y4\",...), with several vertices' coordinates connected counterclockwise.\nYou should carefully consider how to translate them so that they don't overlap and don't go beyond the surface's boundaries based on the vertices' coordinates of each polygon. You may need to tell the user the polygons' position after translating.\n"

  for index,sample in enumerate(data):
    new_diag=[]
    for index,diag in enumerate(sample["messages"]):
      if diag["role"]!="system":
        if index==1:
          diag={"role":diag["role"],"content":system_message+diag["content"]}
        else:
          diag={"role":diag["role"],"content":diag["content"]}  
        new_diag.append(diag)
    
    new_data.append({"messages": new_diag})
  return new_data

train_dataset = Dataset.from_list(preprocess_dataset("train.jsonl"))

test_dataset = Dataset.from_list(preprocess_dataset("test.jsonl"))
print(test_dataset[0])
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
