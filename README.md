# LLM Fine-tuning Project

## Project Overview
This project demonstrates fine-tuning of a language model (LLM) to generate impressions based on reports. The model is fine-tuned on a dataset containing 300 reports and evaluated on 30 reports.

## Folder Structure

my_llm_finetuning_project/ │ ├── data/ # Contains dataset ├── models/ # Stores the fine-tuned models ├── notebooks/ # Contains exploration notebook (data_exploration.ipynb) ├── src/ # Python scripts for fine-tuning and evaluation ├── scripts/ # Shell scripts to run the training process ├── venv/ # Virtual environment ├── requirements.txt # Python dependencies └── README.md # This file


## How to Run
1. Set up a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run fine-tuning:
    ```bash
    bash scripts/run_finetuning.sh
    ```

4. Run evaluation:
    ```bash
    python src/evaluation.py
    ```

## Results
- **Perplexity**: [Add results here]
- **ROUGE scores**: [Add results here]

## Text Analysis and Visualization
The data exploration and text analysis, including visualization of word pairs, is located in the `notebooks/data_exploration.ipynb`.

