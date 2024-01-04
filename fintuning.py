import logging
import os
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
  TrainingArguments,
)
from peft import (
  prepare_model_for_kbit_training, 
  LoraConfig
)
from trl import SFTTrainer

base_model_name = "Sacralet/mistral-7B"
dataset_name = "Sacralet/mistral_chat_nesting_dataset"
finetuned_model_name = "dbw-mistral-7B-1"

import wandb
import huggingface_hub

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


bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_use_double_quant=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
  base_model_name,
  quantization_config=bnb_config,
  resume_download=True,
  device_map="auto",
  trust_remote_code=True,
  use_auth_token=True,
  cache_dir=hg_cache_dir
)
model.config.use_cache = False
model.config.pretraining_tp = 1

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model, 4)

tokenizer = AutoTokenizer.from_pretrained(
  base_model_name, 
  padding_side="right",
  trust_remote_code=True,
  cache_dir=hg_cache_dir
)

tokenizer.eos_token = "</s>"
tokenizer.pad_token = tokenizer.unk_token


peft_config = LoraConfig(
  lora_alpha=128,
  lora_dropout=0.1,
  r=64,
  bias="none",
  task_type="CAUSAL_LM",
)

checkpoint_dir = Path(finetuned_model_name)
resume_from_checkpoint = False
if checkpoint_dir.is_dir():
  checkpoint_files = list(checkpoint_dir.glob("checkpoint-*"))
  if checkpoint_files:
    resume_from_checkpoint = True

training_arguments = TrainingArguments(
  do_eval=True,
  evaluation_strategy="steps",
  eval_delay=2000,
  push_to_hub=True,
  output_dir=finetuned_model_name,
  warmup_ratio=0.03,
  per_device_train_batch_size=4,
  gradient_accumulation_steps=1,
  num_train_epochs=1,
  lr_scheduler_type="cosine",
  learning_rate=1e-6,
  weight_decay=0.001,
  fp16=False,
  bf16=False,
  optim="paged_adamw_32bit",
  logging_dir="./log",
  logging_steps=10,
  save_strategy="steps",
  save_steps=2000,
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

trainer.train(resume_from_checkpoint=resume_from_checkpoint)
trainer.push_to_hub()

