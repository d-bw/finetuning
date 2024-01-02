from datasets import Dataset,load_dataset
import huggingface_hub
import json 
from transformers import (
  AutoTokenizer,
  LlamaTokenizer
  
)

huggingface_hub_token = "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"
huggingface_hub.login(token=huggingface_hub_token)

data=[]

with open("train.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        # 从每行中加载JSON数据
        json_data = json.loads(line.strip())
        
        # 将加载的JSON数据添加到列表中
        data.append(json_data)

dataset = Dataset.from_list(data)

base_model_name = "Sacralet/llama2-chat-7B"

tokenizer = LlamaTokenizer.from_pretrained(
  base_model_name, 
  trust_remote_code=True,
)

def convert_to_llama_chat(example):


  B_INST, E_INST = "[INST]", "[/INST]"
  B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
  B_SEP,E_SEP="<s>", "</s>"
  dialog = example["messages"]


  if dialog[0]["role"] == "system":
    dialog = [
    {
        "role": dialog[1]["role"],
        "content": B_SYS
        + dialog[0]["content"]
        + E_SYS
        + dialog[1]["content"],
    }
    ] + dialog[2:]
  fin_prom=""
  for prompt, answer in zip(dialog[::2], dialog[1::2]):
    fin_prom+=f"{B_SEP}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {E_SEP}"

  return {"prompt": fin_prom}


dataset = dataset.map(convert_to_llama_chat)

dataset.push_to_hub("Sacralet/llama_chat_nesting_dataset", private=False, token=huggingface_hub_token)