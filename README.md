# llm-aspr
Final Project for ICS 691G: NLP and LLMs, an LLM-based classifier for automated scholarly paper review (ASPR)

### Setup Python environment
```
python -m venv .venv
pip install -r requirements
```

### Setup .env with your openreview credentials
Create a .env  file with the following format:
```
EMAIL=<OPENREVIEW-EMAIL>
PASSWORD=<OPENREVIEW-PASSWORD>
```
Then run `python config.py` to instantiate environement variables.

### Run data_scraper.py
```
python data_scraper.py
```