# Install libraries
#!pip uninstall -y transformers accelerate datasets
#!pip install transformers==4.56.1 datasets accelerate
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from datasets import Dataset

# Example conversations
texts = [
    "User: Hello\nAssistant: Hi! How can I help?",
    "User: What is AI?\nAssistant: AI means Artificial Intelligence.",
    "User: Bye\nAssistant: Goodbye!"
]

dataset = Dataset.from_dict({"text":texts})

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation = True,
        padding = "max_length",
        max_length = 128
    )
tokenized_dataset = dataset.map(tokenize)
data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm = False
)
training_args = TrainingArguments(
          output_dir = "./gpt2_model",
          overwrite_output_dir = True,
          num_train_epochs = 3,
          per_device_train_batch_size = 2,
          save_steps = 100,
          logging_steps = 10
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = tokenized_dataset,
    data_collator = data_collator
)

trainer.train()
trainer.save_model("./gpt2_model")
tokenizer.save_pretrained("./gpt2_model")
