# 🎬 Fine-Tuned BERT for Movie Review Sentiment Analysis

This project fine-tunes a pre-trained BERT model (`bert-base-uncased`) for **binary sentiment classification** on the **IMDb movie review dataset** using Hugging Face Transformers and Datasets. The goal is to classify movie reviews as **positive** or **negative**.

---

## 📌 Project Overview

- **Project Name:** Fine-Tuned BERT for Movie Review Sentiment Analysis  
- **Objective:** Train a transformer-based model to detect sentiment in natural language  
- **Model:** `bert-base-uncased`  
- **Dataset:** IMDb movie reviews (via 🤗 `datasets`)  
- **Frameworks:** Hugging Face `transformers`, `datasets`, PyTorch  
- **Environment:** Google Colab / Jupyter Notebook

---

## 🛠️ Tech Stack

| Category           | Tool / Library                              |
|--------------------|---------------------------------------------|
| Language           | Python 3.11                                  |
| Model              | BERT (bert-base-uncased)                    |
| Dataset            | IMDb via `datasets`                         |
| Tokenization       | Hugging Face Tokenizer                      |
| Training Framework | `Trainer`, `TrainingArguments`              |
| Environment        | Google Colab                                |
| Dependencies       | `transformers`, `datasets`, `torch`, `aiohttp`, `fsspec` |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/movie-sentiment-bert.git
cd movie-sentiment-bert
```

### 2. Install Dependencies
```
pip install transformers datasets torch
```

### 3. Run the Notebook
Launch `bert_sentiment.ipynb` in Google Colab or Jupyter Notebook to begin training.

## Dataset
#### IMDb Dataset

* 50,000 labeled movie reviews

* 25,000 training / 25,000 test

* Labels: positive (1), negative (0)

## Model Details
1. Base Model: bert-base-uncased

2. Architecture: BERT + classification head on [CLS] token

3. Tokenization: Padding + truncation to max length

4. Loss: CrossEntropy

5. Optimizer: AdamW

##  Training Configuration
```
TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)
```

## Results
* Accuracy: ~90% (depending on hyperparameters)

* Model generalizes well on unseen reviews

* Supports quick inference via model(**inputs)

## Thank You
