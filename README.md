# 🧠 Qwen2.5 for Math Problem Classification – Kaggle Competition

This repository contains the code and training pipeline for adapting **decoder-style large language models (LLMs)** for **multi-class classification**, specifically applied to the [Kasut Academy Math Problem Classification Competition](https://www.kaggle.com/competitions/classification-of-math-problems-by-kasut-academy/overview) on Kaggle.

> 💡 **TL;DR**: Using Qwen2.5–3B with LoRA adapters, we surpassed the common 85% F1 barrier of encoder-based models and achieved **87% micro F1** on the validation set — demonstrating that LLMs, when carefully adapted, can significantly outperform traditional encoder classifiers in text classification tasks.

---

## 🚀 Overview

The competition task was to classify math word problems into 8 distinct categories using natural language processing techniques. While initial experiments with encoder models like ModernBERT and MathBERT plateaued at ~85% F1, we achieved breakthrough performance by leveraging **decoder-based LLMs**, specifically the **Qwen 2.5 series**.

This repository contains a **production-ready training pipeline** using:
- Hugging Face `transformers` & `accelerate`
- PEFT (LoRA adapters) for efficient fine-tuning
- PyTorch + custom classification head
- Multi-GPU support

---

## 📂 Project Structure

- `main.py` – Contains the training script for fine-tuning Qwen 2.5 with LoRA (can also be adapted for LLAMA 3.2).
- `dataset/` – A directory containing the dataset files, including:
  - `train.csv` – Original Training data
  - `test.csv` – Original Test data
  - `preprocessed/` – Preprocessed dataset files, containing:
    - `train.csv` – Preprocessed training data
    - `val.csv` – Preprocessed validation data
- `notebooks/` – Jupyter notebooks for exploratory data analysis and initial experiments.
- `utils/` - Contains utility functions for data loading, preprocessing, and model creation:
    - `llama32/` - Contains utility functions for creating LLAMA 3.2 models.
    - `qwen25/` - Contains utility functions for creating Qwen 2.5 models.
    - `create_dataloaders.py` – Functions to create PyTorch data loaders for training and validation datasets.
    - `save_adapter_only.py` – Functions to save only the LoRA adapter weights, excluding the base model.
    - `summarize_model.py` – Functions to summarize the model architecture and parameters.
    - `train.py` - Contains the training loop and evaluation logic for fine-tuning the model.
- `pyproject.toml` – Project configuration file for dependencies and settings.

---

## 🔬 Initial Encoder-Based Experiments (Baseline)

Before moving to LLMs, we ran fine-tuning experiments on encoder-based models:

| Model               | Validation Micro F1 |
|--------------------|---------------------|
| MathBERT            | 84%                |
| ModernBERT Base    | 84%                |
| ModernBERT Large   | 85%                |

Despite extensive hyperparameter tuning and partial unfreezing, these models consistently hit a ceiling around **85%**, prompting a shift to decoder models.

---

## 🧪 LLM Experiments

We experimented with the following LLMs:

- ✅ [Qwen2.5–0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- ✅ [Qwen2.5–1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- ✅ [Qwen2.5–3B](https://huggingface.co/Qwen/Qwen2.5-3B) (**best**)
- ❌ [LLAMA3.2–1B](https://huggingface.co/meta-llama/llama-3.2-1b) (struggled to generalize)

> **Key Insight**: LLAMA struggled to generalize in comparison to Qwen2.5 models, even the smallest Qwen2.5-0.5B was better.

---

## 🏆 Best Model: Qwen 2.5–3B + LoRA

**Custom classifier architecture**:  
I used a dropout-regularized linear classifier head on top of the LLM’s final hidden state, with the sequence end position pooled using the attention mask.

**LoRA Configuration**:
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="all",
    inference_mode=False
)
```

**Hyperparameters**:

* Dropout: 0.5
* Epochs: 5 (best model saved after **epoch 3** to avoid overfitting)
* Label smoothing: 0.1
* Weight decay: 0.001
* Learning rate:

  * Base model: `5e-5`
  * Classifier head: `1e-4`

**Validation Result**:
✅ **87% micro F1** on the validation set

> LoRA was crucial — full fine-tuning led to overfitting, while LoRA adapters enabled efficient, stable training even on **2× NVIDIA T4 GPUs**.

⚡ Scaling Note
Based on experimentation, scaling the Qwen2.5 model improves classification performance. While this repo uses Qwen2.5–3B, a larger model such as Qwen2.5–32B, fine-tuned with the same LoRA strategy, is expected to deliver even better results for this task.

---

## 📌 How to Use

If you're looking to use **LLMs for classification**, this repo serves as a practical reference. Key components:

* Custom wrapper to turn decoder-only LLMs into classifiers
* Efficient fine-tuning with LoRA adapters
* Multi-GPU support with `accelerate`

---

## 🧠 Citation

This work was conducted as part of the [Kasut Academy Math Problem Classification Competition](https://www.kaggle.com/competitions/classification-of-math-problems-by-kasut-academy). Great job by the organizers for providing the dataset and hosting the challenge.