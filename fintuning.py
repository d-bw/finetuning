import logging
import os
from pathlib import Path
import torch
import argparse
from datasets import load_dataset
from transformers import (
  AutoModelForCausalLM,
  LlamaForCausalLM,
  BertForMaskedLM, 
  LlamaTokenizer,
  BertTokenizer,
  AutoTokenizer,
  BitsAndBytesConfig,
  TrainingArguments,
  Trainer
)
import wandb
import huggingface_hub
from peft import (
  prepare_model_for_kbit_training, 
  LoraConfig
)
from trl import SFTTrainer

parser = argparse.ArgumentParser(description='choose for fintuning details')

parser.add_argument('-model_type', '--type', dest='model_type', type=str,required=True,
                     help='choose from maskedLM and casualLM')

parser.add_argument('-model_id', '--id', dest='model_id', type=str,required=True,
                     help='input your model name space')

parser.add_argument('-dataset', '--dataset', dest='dataset', type=str,required=True,
                     help='input your dataset name space')

parser.add_argument('-output', '--output', dest='output', type=str,required=True,
                     help='input your name space fot save model')


args = parser.parse_args()

#base_model_name = "Sacralet/mistral-7B"
base_model_name=args.model_id
#dataset_name = "Sacralet/mistral_chat_nesting_dataset"
dataset_name=args.dataset
#finetuned_model_name = "dbw-mistral-7B-1"
finetuned_model_name=args.output

hg_cache_dir = "/content/workspace"

os.environ["HUGGINGFACE_HUB_CACHE"] = hg_cache_dir
Path(os.environ["HUGGINGFACE_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)

huggingface_hub_token = "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"
wandb_key = "252f15f35d6ab2ede0d4b91dfc5b42ca60724ae2"

huggingface_hub.login(token=huggingface_hub_token)

wandb.login(key=wandb_key)
wandb.init(resume=True, project=finetuned_model_name, name=finetuned_model_name)

train_dataset = load_dataset(dataset_name, split="train")
val_dataset = load_dataset(dataset_name, split="validation")


if args.model_type=="casualLM":
  
  bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_use_double_quant=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.bfloat16
)
  if "llama" in base_model_name.lower():
    model = LlamaForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    resume_download=True,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True,
    cache_dir=hg_cache_dir
    )
    
    tokenizer = LlamaTokenizer.from_pretrained(
    base_model_name, 
    padding_side="right",
    trust_remote_code=True,
    cache_dir=hg_cache_dir
    )

  elif "mistral" in base_model_name.lower():

    model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    resume_download=True,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True,
    cache_dir=hg_cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
    base_model_name, 
    padding_side="right",
    trust_remote_code=True,
    cache_dir=hg_cache_dir
    )
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = tokenizer.unk_token
  


  model.config.use_cache = False
  model.config.pretraining_tp = 1

  model.gradient_checkpointing_enable()
  model = prepare_model_for_kbit_training(model, 4)

  peft_config = LoraConfig(
  lora_alpha=128,
  lora_dropout=0.1,
  r=64,
  bias="none",
  task_type="CAUSAL_LM",
  )
  training_arguments = TrainingArguments(
  do_eval=True,
  evaluation_strategy="steps",
  eval_delay=100,
  eval_steps=100,
  push_to_hub=True,
  output_dir=finetuned_model_name,
  warmup_ratio=0.03,
  per_device_train_batch_size=8,
  gradient_accumulation_steps=1,
  num_train_epochs=1,
  lr_scheduler_type="cosine",
  learning_rate=1e-6,
  weight_decay=0.001,
  fp16=False,
  bf16=False,
  optim="paged_adamw_32bit",
  logging_dir="./log",
  logging_steps=1,
  save_strategy="steps",
  save_steps=100,
  save_total_limit=1,
  max_grad_norm=0.3,
  max_steps=-1,
  group_by_length=True,
  report_to="wandb",
  )

  trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    dataset_text_field="prompt",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
  )


elif args.model_type=="maskedLM":
  if "bert" in base_model_name.lower():
    model = BertForMaskedLM.from_pretrained(
      base_model_name,
      resume_download=True,
      trust_remote_code=True,
      use_auth_token=True,
      cache_dir=hg_cache_dir
      )
    tokenizer = BertTokenizer.from_pretrained(
      base_model_name, 
      padding_side="right",
      trust_remote_code=True,
      cache_dir=hg_cache_dir
    )
  model.gradient_checkpointing_enable()
  model.config.use_cache = False
  model.config.pretraining_tp = 1

  training_arguments = TrainingArguments(
  do_eval=True,
  evaluation_strategy="steps",
  eval_delay=500,
  eval_steps=500,
  push_to_hub=True,
  output_dir=finetuned_model_name,
  warmup_ratio=0.03,
  per_device_train_batch_size=32,
  gradient_accumulation_steps=1,
  num_train_epochs=1,
  lr_scheduler_type="cosine",
  learning_rate=1e-5,
  weight_decay=0.001,
  fp16=False,
  bf16=False,
  optim="paged_adamw_32bit",
  logging_dir="./log",
  logging_steps=1,
  save_strategy="steps",
  save_steps=500,
  save_total_limit=1,
  max_grad_norm=0.3,
  max_steps=-1,
  group_by_length=True,
  report_to="wandb",
  )
  trainer = SFTTrainer(
  model=model,
  train_dataset=train_dataset,
  eval_dataset=val_dataset,
  dataset_text_field="prompt",
  args=training_arguments,
  packing=False,
  max_seq_length=512,  
  )

checkpoint_dir = Path(finetuned_model_name)
resume_from_checkpoint = False
if checkpoint_dir.is_dir():
  checkpoint_files = list(checkpoint_dir.glob("checkpoint-*"))
  if checkpoint_files:
    resume_from_checkpoint = True


trainer.train(resume_from_checkpoint=resume_from_checkpoint)
trainer.push_to_hub()

