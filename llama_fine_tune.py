from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
import numpy as np
import evaluate
from datasets import load_dataset

# Initialize the LLaMA model
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
model_id = "meta-llama/Llama-3.2-1B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    # torch_dtype="auto",
    # torch_dtype=torch.int8,
    # quantization_config=bnb_config,
    # device_map="auto",
    # device_map=0,
    device_map="auto"
)


metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)



dataset = load_dataset("wikimedia/wikipedia", "20231101.ro")

tokenized_datasets = dataset.map(tokenize_function, batched=True)


training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)