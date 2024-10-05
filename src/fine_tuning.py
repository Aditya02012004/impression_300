from transformers import AutoTokenizer, Trainer, TrainingArguments, BertLMHeadModel,GPT2LMHeadModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

# Step 1: Load dataset
def load_data(file_path):
    dataset = load_dataset('csv', data_files=file_path)
    dataset = dataset['train'].train_test_split(test_size=0.1)  # Split into train/eval sets
    return dataset

# Step 3: Model Fine-tuning
def fine_tune_model(model_name, dataset, output_dir, batch_size=4, num_epochs=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples['Impression'], truncation=True, padding='max_length', max_length=512,
                                     return_tensors='pt')
        tokenized_inputs['labels'] = tokenized_inputs['input_ids'].clone()  # Copy input_ids for labels
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_steps=10_000,
        save_total_limit=2,
        fp16=True,  # Enable mixed precision if using GPU
    )

    # Define the metrics function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        # Flatten predictions and labels for comparison (ignoring padding tokens)
        preds_flat = preds[labels != -100]
        labels_flat = labels[labels != -100]

        accuracy = accuracy_score(labels_flat, preds_flat)
        precision = precision_score(labels_flat, preds_flat, average='weighted', zero_division=0)
        recall = recall_score(labels_flat, preds_flat, average='weighted', zero_division=0)
        f1 = f1_score(labels_flat, preds_flat, average='weighted', zero_division=0)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # Train Model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    dataset = load_data('../data/impression_300_llm.csv')
    print(dataset)
    print(dataset.column_names)
    model_name = "bert-base-uncased"  # Replace with "gemma-7b-it" if using high resource hardware
    fine_tune_model(model_name, dataset, "../models/finetuned_model", batch_size=2, num_epochs=3)

