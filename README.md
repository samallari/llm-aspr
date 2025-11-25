# LLM-ASPR: LLM-based Classifier for Automated Scholarly Paper Review
Authors: [James Ligeralde](https://github.com/jligeral), [Samantha Mallari](https://github.com/samallari), [Tevin Takata](https://github.com/tevin-takata), [Sean Flynn](https://github.com/seanhflynn)

This is a final project for ICS 691G: NLP and LLMs (Fall 2025).

## 1. Setup Python environment
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements
```

## 2. Setup .env
Copy the example .env file and replace the placeholder values with your OpenReview credentials.
```
cp .env.example .env
```
Run `python config.py` to instantiate environment variables.

## 3. Dataset Generation
Run data_scraper.py to collect raw data from OpenReview from a single conference.
```
cd data
python data_scraper.py
```

## 4. Model Evaluation
Run `notebooks/baseline_bert_classifier.ipynb` to fine-tune a BERT-based classifier on the dataset and evaluate its performance.
