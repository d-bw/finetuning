from transformers import AutoTokenizer, pipeline
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
import json

# Model and tokenizer initialization
model = "Sacralet/mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model)
chat_pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.float32, "load_in_4bit": True},
)

# Load dataset
dataset_name = "Sacralet/mistral_chat_nesting_dataset"
dataset = load_dataset(dataset_name, split="test").select(range(3))

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
            outputs = chat_pipeline(first_prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
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
                history = chat_pipeline(prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
                assistant_response = history[0]["generated_text"].replace(prompt, "")
                format_messages.append({'content': user_message, 'role': "user"})
                format_messages.append({'content': assistant_response, 'role': "assistant"})
                prompt = history[0]["generated_text"] + EOS_TOKEN + B_INST

            # Write to file
            json.dump(format_messages, file)
            file.write('\n')

# Run inference
inference(dataset, "test.jsonl")
