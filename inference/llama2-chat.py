import json
import pandas as pd
import argparse
from transformers import pipeline
from transformers import LlamaForCausalLM, LlamaTokenizer,TextStreamer
import torch
from peft  import PeftModel ,PeftConfig


parser = argparse.ArgumentParser(description='传递已有对话数据集和目标数据集')

parser.add_argument('-model_id', '--type', dest='model_id', type=str,required=True,
                    default='/Path/to/xxxxx', help='选择模型路径')
parser.add_argument('-peft_model', '--folder', dest='peft_id', type=str,
                    required=False, default=None , help='选择peft路径')
parser.add_argument('-dialogue_dataset', '--origin', dest='dialogue_dataset', type=str,required=True,
                    default='/Path/to/xxxxx', help='选择数据集路径')
parser.add_argument('-output_dataset', '--target', dest='output_dataset', type=str,
                    required=False, default=None , help='选择输出数据集')
args = parser.parse_args()


dialogue_data = pd.read_json(args.dialogue_dataset, lines=True)
model=args.model_id
tokenizer= LlamaTokenizer.from_pretrained(model)

model=LlamaForCausalLM.from_pretrained(model, load_in_8bit=False, device_map='auto',torch_dtype=torch.float16)

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

if args.peft_id!=None:
  pipeline.model = PeftModel.from_pretrained(model,args.peft_id)
    
target_file_path=""
if args.output_dataset!=None:
    target_file_path=args.output_dataset
    
target_data=[]

SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You are an AI expert in Translation Inference of Polygons in 2D surface, tasked with avoiding overlap and staying within the boundaries. Users will provide the surface dimensions and polygon coordinates in a standard format: polygonN('x1,y1','x2,y2',...), listed counterclockwise. Translate the polygons without overlap or boundary breaches. Update user's on the new positions of the polygons post-translation.
<</SYS>>
"""


for messages in dialogue_data.to_dict(orient='records'):
    flag=0
    new_dialogue={
        "messages":[]
        }
    history=[]
    for dialogue in messages["messages"]:
         
        
        if dialogue["role"]=="system":
            new_dialogue["messages"].append(dialogue)
        elif dialogue["role"]=="user":
            new_dialogue["messages"].append(dialogue)
            user_message=dialogue["content"]
            if len(history) == 0:
                formatted_message= SYSTEM_PROMPT + f"{user_message} [/INST]"
            else:
                formatted_message = SYSTEM_PROMPT + f"{history[0][0]} [/INST] {history[0][1]} </s>"
                for user_msg, model_answer in history[1:]:
                    formatted_message += f"<s>[INST] {user_msg} [/INST] {model_answer} </s>"
                formatted_message += f"<s>[INST] {user_message} [/INST]"
                
            sequences = llama_pipeline(
                            formatted_message,
                            do_sample=True,
                            top_k=10,
                            num_return_sequences=1,
                            eos_token_id=tokenizer.eos_token_id,
                            max_length=2000,
                        )
            generated_text = sequences[0]['generated_text']
            response = generated_text[len(formatted_message):]  # Remove the prompt from the output

            _history=[]
            _history.append(user_message)
            
            _history.append(response.strip())
            history.append(_history)
            print(history)
            
                
        else:
            
            new_assistant_message={
                "role":"assistant",
                "content":history[flag][1]
            }
            flag+=1
            new_dialogue["messages"].append(new_assistant_message)
    if target_file_path!="":       
        with open(target_file_path, 'a') as file:
            json.dump(new_dialogue, file)
            file.write('\n')  
