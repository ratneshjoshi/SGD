from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments

from datasets import load_dataset

dataset = load_dataset('csv', data_files={'train': 'data/train_augmented.csv', 'eval':'data/val_augmented.csv'})

model_checkpoint = "gpt2-large"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

special_tokens_dict = {'additional_special_tokens': ['[CONTEXT]','[END_OF_CONTEXT]','[speaker1]','[speaker2]', '[RESPONSE]', '[END_OF_RESPONSE]', '[PERSONA]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    return tokenizer(examples["text"])

block_size=256

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

model_name = model_checkpoint

training_args = TrainingArguments(
    f"{model_name}-persona",
    evaluation_strategy = "epoch",
    save_total_limit = 2,
    save_strategy = "epoch",
    load_best_model_at_end=True,
    num_train_epochs=30,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["eval"],
)

trainer.train()
trainer.save_model()
