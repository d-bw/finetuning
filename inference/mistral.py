from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM , BitsAndBytesConfig
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
import json
from peft  import PeftModel ,PeftConfig
import os
from pathlib import Path
import huggingface_hub

huggingface_hub_token = "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"
huggingface_hub.login(token=huggingface_hub_token)
hg_cache_dir = "/notebooks/dbw/finetuning"

os.environ["HUGGINGFACE_HUB_CACHE"] = hg_cache_dir

Path(os.environ["HUGGINGFACE_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_use_double_quant=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.bfloat16
)

# Model and tokenizer initialization
model_id = "Sacralet/mistral-7B"
model=AutoModelForCausalLM.from_pretrained(
  model_id,
  quantization_config=bnb_config,
  resume_download=True,
  device_map="auto",
  trust_remote_code=True,
  use_auth_token=True,
  cache_dir=hg_cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(
  "Sacralet/dbw-mistral-7B-01", 
  padding_side="right",
  trust_remote_code=True,
  cache_dir=hg_cache_dir
)


chat_pipeline = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.float32, "load_in_4bit": True},
)

chat_pipeline.model=PeftModel.from_pretrained(model,"Sacralet/dbw-mistral-7B-01")

# Load dataset
dataset_name = "Sacralet/mistral_chat_nesting_dataset"
dataset = load_dataset(dataset_name, split="test").select(range(200))

#dataset=dataset["messages"][:3]

def inference(dataset, filename):
    # Special tokens
    EOS_TOKEN = "</s>"
    B_INST = "[INST]"
    E_INST = "[/INST]"

    with open(filename, 'w') as file:
        for sample in tqdm(dataset['messages']):
            format_messages = []
            prompt = ""

            # First message
            first_message = sample[0]
            format_messages.append(first_message)
            first_prompt = tokenizer.apply_chat_template([first_message], tokenize=False, add_generation_prompt=True)
            outputs = chat_pipeline(first_prompt, max_new_tokens=2048, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            prompt = outputs[0]["generated_text"] + EOS_TOKEN + B_INST

            # Response
            response = {
                'content': outputs[0]["generated_text"].replace(first_prompt, ""),
                'role': "assistant"
            }
            format_messages.append(response)

            # Subsequent messages
            for i in range(2, len(sample), 2):
                user_message = sample[i]['content']
                prompt += " " + user_message + " " + E_INST
                history = chat_pipeline(prompt, max_new_tokens=2048, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
                assistant_response = history[0]["generated_text"].replace(prompt, "")
                format_messages.append({'content': user_message, 'role': "user"})
                format_messages.append({'content': assistant_response, 'role': "assistant"})
                prompt = history[0]["generated_text"] + EOS_TOKEN + B_INST

            # Write to file
            json.dump(format_messages, file)
            file.write('\n')

# Run inference
inference(dataset, "format_inference_test.jsonl")





