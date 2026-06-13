**Problem**

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.


**About the Dataset**

This is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 1st Jan 2019 - 31st Dec 2020. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants.

The dataset can be found in this URL: [https://www.kaggle.com/datasets/kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

**Setup**

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Set up Kaggle credentials (see `.env.example`):
```bash
cp .env.example .env
# Edit .env with your Kaggle username and API key from kaggle.json
```
3. Create the virtual environment:
```bash
uv sync
```
4. Open `analysis.ipynb` in VS Code and select the `.venv` interpreter, or run:
```bash
uv run analysis.py
```