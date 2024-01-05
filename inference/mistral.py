from transformers import AutoTokenizer
import transformers
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset

model = "Sacralet/mistral-7B"

dataset_name = "Sacralet/mistral_chat_nesting_dataset"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.float32, "load_in_4bit": True},
)
dataset=load_dataset(dataset_name, split="test")

def inference(dataset):
  message=[]
  EOS_TOKEN="</s>"
  B_INST="[INST]"
  E_INST="[/INST]"
  first_message=dataset['messages'][0][0]
  print(first_message)
  first_prompt = pipeline.tokenizer.apply_chat_template([first_message], tokenize=False, add_generation_prompt=True)
  outputs = pipeline(first_prompt, max_new_tokens=2048, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
  message.append(first_message)
  prompt=outputs[0]["generated_text"]+EOS_TOKEN+B_INST
  response={
      'content':outputs[0]["generated_text"].replace(first_prompt,""),
      'role':"assistant"
    }
  message.append(response)
  for i in range(2,len(dataset['messages'][0]),2):
    message.append(dataset['messages'][0][i])
    prompt+=" "+dataset['messages'][0][i]["content"]+" "+E_INST
    history = pipeline(prompt, max_new_tokens=2048, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    response=history[0]["generated_text"].replace(prompt,"")
    response={
      'content':response,
      'role':"assistant"
    }
    message.append(response)
    prompt=history[0]["generated_text"]+EOS_TOKEN+B_INST
    print(message)
  
  return message
    

inference(dataset)
