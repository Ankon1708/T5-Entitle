# Fine-tuning Flan-T5 for Article Title Generation

## Introduction

This project focuses on fine-tuning Google's Flan-T5 model, a state-of-the-art Large Language Model (LLM), to perform abstractive title generation for articles. The goal is to automatically create concise and relevant titles from longer article descriptions. This task is crucial for content summarization, information retrieval, and improving user experience by providing quick insights into article content. A key focus of this project was exploring Parameter-Efficient Fine-Tuning (PEFT) methods, specifically **Low-Rank Adaptation (LoRA)**, to achieve high performance with significantly reduced computational resources and storage requirements.

## Model Architecture

The core of this project is the Flan-T5 model. Flan-T5 is an encoder-decoder Transformer model pre-trained by Google and extensively fine-tuned on a diverse set of instruction-based tasks. Its text-to-text framework allows it to reformulate all NLP problems into a text generation format.

For fine-tuning, the Hugging Face `transformers` library was used. Critically, to address the challenges of fine-tuning large models on limited GPU memory (e.g., a 32GB GPU), **PEFT (Parameter-Efficient Fine-tuning)** was implemented. The `peft` library allowed the integration of LoRA adapters, which wrap the base Flan-T5 model into a `PeftModel`. This approach enabled efficient training by only updating a small fraction of the model's parameters.

`Seq2SeqTrainer` was used for the training loop alongside `DataCollatorForSeq2Seq` for batching and preparing data, and `Seq2SeqTrainingArguments` for all the configuration settings.

## Dataset

The dataset is sourced from Kaggle, called the Financial News Headlines Data (https://www.kaggle.com/datasets/notlucasp/financial-news-headlines). As per the author of the dataset: "Scraped from CNBC, the Guardian, and Reuters official websites, the headlines in these datasets reflects the overview of the U.S. economy and stock market every day for the past year to 2 years."
Only the Reuters dataset is used in this project.

* **Size:** The dataset contains three columns, amongst which 'Description' and 'Headline' are used, while 'Time' is ignored. The dataset contains 32,770 samples which is split into training, validation, and test sets with the ratio 60:15:25.
* **Preprocessing:** The instruction directive 'Entitle' is added to the beginning of 'Description' of each sample. The dataframe is then converted to a Hugging Face dataset and subjected to splits as described above.
* **Tokenization:** The Hugging Face dataset is tokenized using `AutoTokenizer`.

## Modelling

The model was fine-tuned using the Hugging Face Transformers library and the Seq2SeqTrainer. This project explored two fine-tuning strategies:
1.  **Full Fine-tuning:** The traditional approach where all parameters of the base model are updated.
2.  **Low-Rank Adaptation (LoRA):** A parameter-efficient method that injects small, trainable matrices into the original model's layers while keeping the vast majority of the pre-trained weights frozen.

**Key aspects of the fine-tuning process include:**

* **Base Model:** Google's Flan-T5-Base (220M parameters). (An initial attempt to fine-tune Flan-T5-Large (770M parameters) was constrained by the 32GB GPU memory, underscoring the necessity for PEFT techniques.)
* **Framework:** PyTorch
* **Optimizer:** AdamW
* **Learning Rate:** Initialized at 2e-5 and adjusted to 3e-5 during training for full fine-tuning. LoRA training used specific learning rate optimizations as per PEFT guidelines.
* **Batch Size:** An effective global batch size of 32 (16 per GPU on two NVIDIA T4 GPUs).
* **Epochs:** Trained for 6 epochs.
* **Learning Rate Scheduler:** Linear decay of 0.01 with warmup (default in Trainer).
* **Mixed Precision Training (FP16):** Utilized for faster training and reduced memory consumption on GPU.
* **LoRA Configuration:**
    * `r`: 16 (LoRA attention dimension)
    * `lora_alpha`: 32 (Scaling factor for LoRA updates)
    * `target_modules`: `["q", "v"]` (Applied to query and value projection matrices)
    * `lora_dropout`: 0.05
    * `bias`: "none"
    * `task_type`: `TaskType.SEQ_2_SEQ_LM`
* **Parameter Efficiency:** With LoRA, only approximately **0.6%** of the original model's parameters were trained, drastically reducing the memory footprint during training and the size of the saved checkpoints (only the adapters are saved, typically a few MBs).
* **Model Saving & Deployment:** Trained LoRA adapters were saved locally using `model.save_pretrained()` and pushed to the Hugging Face Hub for easy sharing and loading.
* **Experiment Tracking:** Weights & Biases (W&B) was used for real-time logging, visualization, and tracking of metrics and loss curves.

## Evaluation Metrics & Results

The model's performance was evaluated using ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores, which are standard for summarization and text generation tasks. ROUGE measures the overlap of n-grams (ROUGE-N) and longest common subsequences (ROUGE-L) between the generated titles and human-written reference titles.

**Key Evaluation Metrics:**
* ROUGE-1 (F1-score): Measures unigram overlap.
* ROUGE-L (F1-score): Measures the longest common subsequence.

**Results:**

The comparison below highlights the effectiveness of both fine-tuning strategies:

| Model Version             | ROUGE-1 (F1-score) | ROUGE-L (F1-score) | Notes                                                                                                                              |
| :------------------------ | :----------------- | :----------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| **Pre-trained Flan-T5 Base** | 39.1               | 35.6               | Baseline performance.                                                                                                              |
| **PEFT (LoRA) Fine-tuned** | **46.5** | **42.2** | Achieved ~19% improvement over baseline. Drastically reduced trainable parameters (~0.6% of total) and memory footprint.         |
| **Fully Fine-tuned** | **47.5** | **43.1** | Highest performance, but with significantly higher computational and storage costs (all 220M parameters updated).                  |

**Conclusion:** The PEFT (LoRA) fine-tuned model demonstrated substantial performance improvements over the pre-trained baseline, achieving nearly identical results to the computationally intensive full fine-tuning approach, proving its efficiency and practicality for LLM adaptation on resource-constrained hardware.
