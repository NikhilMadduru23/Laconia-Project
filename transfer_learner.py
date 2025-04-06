from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate 
import torch

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
path = "./data/processed/atis.csv"
dataset = load_dataset("csv", data_files={
    'train': path.replace("atis.csv", "atis_train.csv"), 
    'test': path.replace("atis.csv", "atis_test.csv")
})
print(f"Original train size: {len(dataset['train'])}, test size: {len(dataset['test'])}")

# Subsample the dataset
def subsample_dataset(dataset_split, fraction=0.1):
    dataset_split = dataset_split.shuffle(seed=42)
    num_samples = int(len(dataset_split) * fraction)
    print(f"Subsampling {len(dataset_split)} to {num_samples} samples")
    return dataset_split.select(range(num_samples))

reduced_train = subsample_dataset(dataset["train"], fraction=1)
reduced_test = subsample_dataset(dataset["test"], fraction=1)
reduced_dataset = {"train": reduced_train, "test": reduced_test}
print(f"Reduced train size: {len(reduced_train)}, reduced test size: {len(reduced_test)}")

# Number of unique intents
num_intents = len(set(reduced_dataset['train']['encoded_intent']))
print(f"Number of intents: {num_intents}")

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_intents)
model.to(device)  # Move model to GPU

# Tokenize dataset
def tokenize_function(examples):
    texts = examples["text"]
    valid_texts = [t if isinstance(t, str) and t.strip() else "DUMMY" for t in texts]
    tokenized = tokenizer(valid_texts, padding="max_length", truncation=True, max_length=128)
    tokenized["labels"] = examples["encoded_intent"]
    return tokenized

# Apply tokenization to each split
try:
    tokenized_train = reduced_dataset["train"].map(tokenize_function, batched=True)
    tokenized_test = reduced_dataset["test"].map(tokenize_function, batched=True)
    tokenized_dataset = {"train": tokenized_train, "test": tokenized_test}

    for split in ["train", "test"]:
        tokenized_dataset[split] = tokenized_dataset[split].remove_columns(["text", "encoded_intent"])
        tokenized_dataset[split].set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Training args
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir='./logs',
        logging_steps=10,
        fp16=True if torch.cuda.is_available() else False,
        save_strategy="epoch",
    )

    # Load accuracy metric
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        return metric.compute(predictions=predictions, references=labels)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics
    )
    
    # Calculate and print expected steps
    steps_per_epoch = len(tokenized_dataset["train"]) / training_args.per_device_train_batch_size
    total_steps = steps_per_epoch * training_args.num_train_epochs
    print(f"Expected steps per epoch: {steps_per_epoch:.0f}, Total steps: {total_steps:.0f}")
    
    trainer.train()
    metrics = trainer.evaluate(tokenized_dataset["test"])
    print("Evaluation metrics:", metrics)

    trainer.save_model("./saved_models/intent_classifier")
    tokenizer.save_pretrained("./saved_models/intent_classifier")
    print("Model and tokenizer saved to ./saved_models/intent_classifier")

except ValueError as e:
    print(f"Tokenization failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")