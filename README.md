# T5-Entitle

## Introduction
This project focuses on fine-tuning Google's Flan-T5 model, a state-of-the-art Large Language Model (LLM), to perform abstractive title generation for articles. The goal is to automatically create concise and relevant titles from longer article descriptions. This task is crucial for content summarization, information retrieval, and improving user experience by providing quick insights into article content.

## Model Architecture
The core of this project is the Flan-T5 model. Flan-T5 is an encoder-decoder Transformer model pre-trained by Google and extensively fine-tuned on a diverse set of instruction-based tasks. Its text-to-text framework allows it to reformulate all NLP problems into a text generation format. Seq2SeqTrainer was used for fine-tuning alongside DataCollatorForSeq2Seq for batching and preparing data, and Seq2SeqTrainingArguments for all the configuration settings.

## Dataset
The dataset is sourced from Kaggle, called the Financial News Headlines Data (https://www.kaggle.com/datasets/notlucasp/financial-news-headlines). As per the author of the dataset: "Scraped from CNBC, the Guardian, and Reuters official websites, the headlines in these datasets reflects the overview of the U.S. economy and stock market every day for the past year to 2 years."

Only the Reuters dataset is used in this project.

Size: The dataset contains three columns, amongst which 'Description' and 'Headline' are used, while 'Time' is ignored. The dataset contains 32,770 samples which is split into training, validation, and test sets with the ratio 60:15:25.

Preprocessing: The instruction directive 'Entitle' is added to the beginning of 'Description' of each sample. The dataframe is then converted to a huggingface dataset and subjected to splits as described above.

Tokenization: The huggingface dataset is tokenized using AutoTokenizer.

## Modelling
The model was fine-tuned using the Hugging Face Transformers library and the Seq2SeqTrainer. Key aspects of the fine-tuning process include:

Framework: PyTorch

Optimizer: AdamW

Learning Rate: Initialized at 2e−5 and adjusted to 3e−5 during training.

Batch Size: An effective global batch size of 32 (16 per GPU on two NVIDIA T4 GPUs).

Epochs: Trained for 6 epochs.

Learning Rate Scheduler: Linear decay of 0.01 with warmup (default in Trainer).

Mixed Precision Training (FP16): Utilized for faster training and reduced memory consumption on GPU.

Experiment Tracking: Weights & Biases (W&B) was used for real-time logging, visualization, and tracking of metrics and loss curves.

## Evaluation Metrics & Results

The model's performance was evaluated using ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores, which are standard for summarization and text generation tasks. ROUGE measures the overlap of n-grams (ROUGE-N) and longest common subsequences (ROUGE-L) between the generated titles and human-written reference titles.

Key Evaluation Metrics:

ROUGE-1 (F1-score): Measures unigram overlap.

ROUGE-2 (F1-score): Measures bigram overlap.

ROUGE-L (F1-score): Measures the longest common subsequence.

Results:

The fine-tuned model demonstrated substantial performance improvements compared to the pre-trained Flan-T5 baseline.

Metric |	Absolute Improvement |	Percentage Improvement
|---|---|---|
ROUGE-1 |	+7.78 points |	+19.87%
ROUGE-2 |	+5.60 points |	+33.43%
ROUGE-L |	+7.11 points |	+19.96%




