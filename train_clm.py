# train_clm.py
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# distilgpt2 tokenizer may not have a pad token; set it to eos to avoid warnings
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# load our jsonl file (expects field "text" in each line)
dataset = load_dataset("json", data_files={"train": "dataset.jsonl"}, split="train")

# tokenize (short max_length for demo; increase for longer context)
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./distilgpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,                # small number for demo
    per_device_train_batch_size=1,     # set 1 on low-memory machines
    save_steps=200,
    save_total_limit=2,
    logging_steps=50,
    fp16=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./distilgpt2-finetuned")
