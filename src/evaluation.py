from transformers import GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import evaluate  # Correct import for metrics library

# Step 1: Load the fine-tuned model
def load_fine_tuned_model(model_dir):
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

# Step 2: Generate Predictions for Evaluation
def generate_predictions(model, tokenizer, dataset, max_length=512):
    model.eval()
    predictions = []
    references = []

    for sample in dataset:
        inputs = tokenizer(sample['Impression'], return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        with torch.no_grad():
            generated_ids = model.generate(input_ids=inputs['input_ids'], max_length=max_length)

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        predictions.append(generated_text)
        references.append(sample['Impression'])  # Assuming original 'text' as the reference

    return predictions, references

# Step 3: Perplexity Calculation
def calculate_perplexity(model, dataset, tokenizer, max_length=512):
    model.eval()
    total_loss = 0
    total_tokens = 0

    for sample in dataset:
        inputs = tokenizer(sample['Impression'], return_tensors="pt", truncation=True, padding=True, max_length=max_length)  # Use 'Impression' column
        labels = inputs['input_ids']
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item() * labels.size(1)  # Multiply by the number of tokens in the input
        total_tokens += labels.size(1)

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

# Step 4: ROUGE Score Calculation
def calculate_rouge(predictions, references):
    rouge = evaluate.load("rouge")  # Updated to use evaluate.load()
    results = rouge.compute(predictions=predictions, references=references)
    return results

# Step 5: Evaluation Function
def evaluate_model(model, tokenizer, dataset):
    # Generate predictions and calculate references from the test set
    print("Generating predictions...")
    predictions, references = generate_predictions(model, tokenizer, dataset['test'])

    # Calculate Perplexity
    print("Calculating perplexity...")
    perplexity = calculate_perplexity(model, dataset['test'], tokenizer)
    print(f"Perplexity: {perplexity}")

    # Calculate ROUGE Score
    print("Calculating ROUGE score...")
    rouge_scores = calculate_rouge(predictions, references)
    print(f"ROUGE Scores: {rouge_scores}")

if __name__ == "__main__":
    # Load the fine-tuned model and tokenizer
    model_dir = "../models/finetuned_model"
    model, tokenizer = load_fine_tuned_model(model_dir)

    # Load dataset
    dataset = load_dataset('csv', data_files='../data/impression_300_llm.csv')['train'].train_test_split(test_size=0.1)

    # Evaluate the model
    evaluate_model(model, tokenizer, dataset)
