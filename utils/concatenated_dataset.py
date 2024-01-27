from transformers import BertTokenizer

CHUNK_SIZE=128
tokenizer=tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
def tokenize_function(examples):
    result = tokenizer(examples["prompt"],truncation=True, padding='max_length', max_length=512)
    label = tokenizer(examples["messages"],truncation=True, padding='max_length', max_length=512)

    label["labels"] = label["input_ids"].copy()
    label["input_ids"] = result["input_ids"]
    return label

def group_texts(examples):
    chunk_size=CHUNK_SIZE
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    
    return result

#train_dataset = load_dataset("Sacralet/bertMasked_nestingDataset", split="train")
#tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
#tokenized_datasets = train_dataset.map(
    #tokenize_function, batched=True, remove_columns=["messages", "prompt"],)

#concatenated_datasets = tokenized_datasets

#print(concatenated_datasets)
#print(tokenizer.decode(concatenated_datasets[1]["input_ids"]))
#print(tokenizer.decode(concatenated_datasets[1]["labels"]))
#print(len(tokenizer.decode(concatenated_datasets[35999]["input_ids"])))
#print(len(concatenated_datasets[35999]["input_ids"]))