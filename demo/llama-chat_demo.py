from transformers import pipeline
from transformers import LlamaForCausalLM, LlamaTokenizer,TextStreamer
import torch
import gradio as gr
import argparse
from peft  import PeftModel ,PeftConfig

parser = argparse.ArgumentParser(description='传递模型路径和PEFT路径')
parser.add_argument('-model_id', '--type', dest='model_id', type=str,required=True,
                    default='/Path/to/xxxxx', help='选择模型路径')
parser.add_argument('-peft_model', '--folder', dest='peft_id', type=str,
                    required=False, default=None , help='选择peft路径')
args = parser.parse_args()
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


SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You are an AI expert in Translation Inference of Polygons in 2D surface, tasked with avoiding overlap and staying within the boundaries. Users will provide the surface dimensions and polygon coordinates in a standard format: polygonN('x1,y1','x2,y2',...), listed counterclockwise. Translate the polygons without overlap or boundary breaches. Update user's on the new positions of the polygons post-translation.
<</SYS>>
"""


# Formatting function for message and history
def format_message(message: str, history: list, memory_limit: int = 3) -> str:
    """
    Formats the message and history for the Llama model.
    Parameters:
        message (str): Current message to send.
        history (list): Past conversation history.
        memory_limit (int): Limit on how many past interactions to consider.
    Returns:
        str: Formatted message string
    """
    # always keep len(history) <= memory_limit
    if len(history) > memory_limit:
        history = history[-memory_limit:]

    if len(history) == 0:
        return SYSTEM_PROMPT + f"{message} [/INST]"

    formatted_message = SYSTEM_PROMPT + f"{history[0][0]} [/INST] {history[0][1]} </s>"

    # Handle conversation history
    for user_msg, model_answer in history[1:]:
        formatted_message += f"<s>[INST] {user_msg} [/INST] {model_answer} </s>"

    # Handle the current message
    formatted_message += f"<s>[INST] {message} [/INST]"
    print(history)

    return formatted_message

# Generate a response from the Llama model
def get_llama_response(message: str, history: list) -> str:
    """
    Generates a conversational response from the Llama model.
    Parameters:
        message (str): User's input message.
        history (list): Past conversation history.
    Returns:
        str: Generated response from the Llama model.
    """
    query = format_message(message, history)
    response = ""

    sequences = llama_pipeline(
        query,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=2000,
    )

    generated_text = sequences[0]['generated_text']
    response = generated_text[len(query):]  # Remove the prompt from the output

    print({'content':response.strip()})
    
    return response.strip()


gr.ChatInterface(fn=get_llama_response,outputs="streamlit").launch(share=True)