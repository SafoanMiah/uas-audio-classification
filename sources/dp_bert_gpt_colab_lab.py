{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "94f08aef",
   "metadata": {},
   "source": [
    "# Lab: Differential Privacy, BERT, and GPT Fine-Tuning\n",
    "\n",
    "**Recommended runtime:** Google Colab free tier  \n",
    "\n",
    "## What you will do\n",
    "In this lab, students will:\n",
    "\n",
    "1. explore the idea of **differential privacy (DP)** for releasing dataset statistics,\n",
    "2. fine-tune a **small BERT model** for text classification,\n",
    "3. fine-tune a **tiny GPT-style model** for causal language modeling, and\n",
    "4. reflect on the relationship between **privacy, utility, compute, and model behaviour**.\n",
    "\n",
    "## Learning outcomes\n",
    "By the end of the lab, students should be able to:\n",
    "\n",
    "- explain the privacy–utility trade-off using the privacy parameter \\(\\epsilon\\),\n",
    "- release noisy summary statistics with the Laplace mechanism,\n",
    "- fine-tune a compact BERT-family model on a small NLP dataset,\n",
    "- fine-tune a compact GPT-style model on a small text corpus,\n",
    "- discuss why **DP data release** is different from **DP model training**.\n",
    "\n",
    "## Why this notebook is Colab-friendly\n",
    "This notebook uses:\n",
    "- **small dataset slices**,\n",
    "- **compact models** (`prajjwal1/bert-tiny` and `sshleifer/tiny-gpt2`),\n",
    "- **short training runs**,\n",
    "- optional GPU acceleration when available.\n",
    "\n",
    "> **Important note:** Free Colab sessions vary. Google states that Colab is free and provides access to compute resources including GPUs/TPUs, but availability is not guaranteed and runtimes can reset. This lab is therefore designed to run with **small models and short training steps**."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2c331757",
   "metadata": {},
   "source": [
    "## Section 0 — Setup\n",
    "Run the next cell once. On Colab, package installation can take a minute or two."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1bfacea7",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "%%capture\n",
    "!pip -q install datasets transformers accelerate evaluate scikit-learn diffprivlib peft"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "be09db03",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import os\n",
    "import random\n",
    "import math\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from datasets import load_dataset\n",
    "from sklearn.metrics import accuracy_score\n",
    "from diffprivlib.mechanisms import Laplace\n",
    "\n",
    "from transformers import (\n",
    "    AutoTokenizer,\n",
    "    AutoModelForSequenceClassification,\n",
    "    AutoModelForCausalLM,\n",
    "    DataCollatorWithPadding,\n",
    "    DataCollatorForLanguageModeling,\n",
    "    TrainingArguments,\n",
    "    Trainer,\n",
    "    set_seed,\n",
    ")\n",
    "\n",
    "from peft import LoraConfig, TaskType, get_peft_model\n",
    "\n",
    "set_seed(42)\n",
    "random.seed(42)\n",
    "np.random.seed(42)\n",
    "\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "print(\"Torch version:\", torch.__version__)\n",
    "print(\"Device:\", device)\n",
    "if device == \"cuda\":\n",
    "    print(\"GPU:\", torch.cuda.get_device_name(0))\n",
    "else:\n",
    "    print(\"No GPU detected. The notebook will still run, but training will use fewer steps.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cf7830c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# Lab controls: keep these small for the Colab free tier.\n",
    "FAST_RUN = True\n",
    "\n",
    "BERT_TRAIN_SIZE = 1200 if FAST_RUN else 4000\n",
    "BERT_TEST_SIZE = 400 if FAST_RUN else 1000\n",
    "BERT_MAX_STEPS = 80 if device == \"cuda\" else 25\n",
    "\n",
    "GPT_TRAIN_SIZE = 800 if FAST_RUN else 2500\n",
    "GPT_VALID_SIZE = 150 if FAST_RUN else 400\n",
    "GPT_MAX_STEPS = 60 if device == \"cuda\" else 20\n",
    "\n",
    "print({\n",
    "    \"FAST_RUN\": FAST_RUN,\n",
    "    \"BERT_TRAIN_SIZE\": BERT_TRAIN_SIZE,\n",
    "    \"BERT_TEST_SIZE\": BERT_TEST_SIZE,\n",
    "    \"BERT_MAX_STEPS\": BERT_MAX_STEPS,\n",
    "    \"GPT_TRAIN_SIZE\": GPT_TRAIN_SIZE,\n",
    "    \"GPT_VALID_SIZE\": GPT_VALID_SIZE,\n",
    "    \"GPT_MAX_STEPS\": GPT_MAX_STEPS,\n",
    "})"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8bcb12af",
   "metadata": {},
   "source": [
    "---\n",
    "# Section 1 — Differential privacy of dataset statistics\n",
    "\n",
    "In this section, we focus on **privacy-preserving release of statistics about a dataset**.\n",
    "\n",
    "We will:\n",
    "- create a small synthetic student dataset,\n",
    "- compute true counts and means,\n",
    "- release **noisy** versions using the **Laplace mechanism**,\n",
    "- compare results across different values of \\(\\epsilon\\).\n",
    "\n",
    "## Key idea\n",
    "Differential privacy is usually described as a guarantee that the output of a computation does not change “too much” when one person's data is added or removed. In practice, one common way to achieve this is to add **carefully calibrated random noise** to a statistic."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6440777d",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# Create a synthetic student-support dataset.\n",
    "# We use synthetic data so the notebook is safe to share and re-run.\n",
    "rng = np.random.default_rng(42)\n",
    "n = 500\n",
    "\n",
    "df = pd.DataFrame({\n",
    "    \"age\": rng.integers(18, 66, size=n),\n",
    "    \"hours_studied\": np.clip(rng.normal(12, 4, size=n), 0, 30).round(1),\n",
    "    \"uses_ai_tutor\": rng.choice([\"yes\", \"no\"], size=n, p=[0.62, 0.38]),\n",
    "    \"exam_band\": rng.choice([\"A\", \"B\", \"C\", \"D\"], size=n, p=[0.18, 0.31, 0.34, 0.17]),\n",
    "    \"stress_score\": rng.integers(1, 11, size=n),   # bounded from 1 to 10\n",
    "})\n",
    "\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2ef9aa10",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df.describe(include=\"all\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0ded4ba3",
   "metadata": {},
   "source": [
    "## 1.1 True statistics\n",
    "Let's compute some exact statistics first. These are *not* private."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c65ceb3c",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "true_ai_tutor_count = int((df[\"uses_ai_tutor\"] == \"yes\").sum())\n",
    "true_mean_stress = df[\"stress_score\"].mean()\n",
    "true_exam_counts = df[\"exam_band\"].value_counts().sort_index()\n",
    "\n",
    "print(\"True count (uses_ai_tutor == yes):\", true_ai_tutor_count)\n",
    "print(\"True mean stress score:\", round(true_mean_stress, 3))\n",
    "print(\"\\nTrue exam-band counts:\")\n",
    "display(true_exam_counts.to_frame(\"count\"))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "00f0070c",
   "metadata": {},
   "source": [
    "## 1.2 The Laplace mechanism\n",
    "\n",
    "For a count query, the sensitivity is usually 1 because adding or removing one person changes the count by at most 1.\n",
    "\n",
    "For a bounded mean, the sensitivity is:\n",
    "\n",
    "\\[\n",
    "\\Delta f = \\frac{\\text{upper} - \\text{lower}}{n}\n",
    "\\]\n",
    "\n",
    "We will use these formulas in helper functions below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8783ce0b",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def dp_count(true_count, epsilon, sensitivity=1.0):\n",
    "    mech = Laplace(epsilon=epsilon, sensitivity=sensitivity)\n",
    "    return mech.randomise(true_count)\n",
    "\n",
    "def dp_mean(values, epsilon, lower, upper):\n",
    "    values = np.clip(np.asarray(values), lower, upper)\n",
    "    true_mean = values.mean()\n",
    "    sensitivity = (upper - lower) / len(values)\n",
    "    mech = Laplace(epsilon=epsilon, sensitivity=sensitivity)\n",
    "    return mech.randomise(true_mean)\n",
    "\n",
    "def dp_histogram(series, categories, epsilon):\n",
    "    # Split the privacy budget equally across categories for simplicity.\n",
    "    per_query_epsilon = epsilon / len(categories)\n",
    "    noisy_counts = {}\n",
    "    for cat in categories:\n",
    "        true_count = int((series == cat).sum())\n",
    "        noisy_counts[cat] = max(0, dp_count(true_count, epsilon=per_query_epsilon))\n",
    "    return noisy_counts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "26c60669",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "epsilons = [0.1, 0.5, 1.0, 5.0]\n",
    "\n",
    "rows = []\n",
    "for eps in epsilons:\n",
    "    rows.append({\n",
    "        \"epsilon\": eps,\n",
    "        \"true_count_yes\": true_ai_tutor_count,\n",
    "        \"dp_count_yes\": round(dp_count(true_ai_tutor_count, eps), 2),\n",
    "        \"true_mean_stress\": round(true_mean_stress, 3),\n",
    "        \"dp_mean_stress\": round(dp_mean(df[\"stress_score\"], eps, lower=1, upper=10), 3),\n",
    "    })\n",
    "\n",
    "pd.DataFrame(rows)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7da873f7",
   "metadata": {},
   "source": [
    "### Interpretation prompt\n",
    "- What happens when \\(\\epsilon\\) is **small**?\n",
    "- What happens when \\(\\epsilon\\) is **large**?\n",
    "- Which setting gives stronger privacy? Which gives better utility?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "60cc1b1d",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "categories = [\"A\", \"B\", \"C\", \"D\"]\n",
    "noisy_counts_eps_1 = dp_histogram(df[\"exam_band\"], categories, epsilon=1.0)\n",
    "\n",
    "plot_df = pd.DataFrame({\n",
    "    \"exact\": true_exam_counts.reindex(categories).values,\n",
    "    \"dp_epsilon_1.0\": [noisy_counts_eps_1[c] for c in categories]\n",
    "}, index=categories)\n",
    "\n",
    "plot_df.plot(kind=\"bar\", figsize=(8, 4), rot=0)\n",
    "plt.title(\"Exact vs DP-noisy exam-band counts\")\n",
    "plt.ylabel(\"Count\")\n",
    "plt.show()\n",
    "\n",
    "plot_df"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "afc2bd68",
   "metadata": {},
   "source": [
    "## Exercise 1A — Explore the privacy–utility trade-off\n",
    "\n",
    "Change the list of epsilon values below and compare the noisy outputs to the true values.\n",
    "\n",
    "**Task ideas**\n",
    "1. Try very small values such as `0.05` or `0.1`.\n",
    "2. Try moderate values such as `0.5` or `1.0`.\n",
    "3. Try larger values such as `3.0` or `10.0`.\n",
    "4. Write 2–3 sentences explaining the trend you observe."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "31d42522",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# TODO: edit eps_to_try and re-run\n",
    "eps_to_try = [0.05, 0.1, 0.5, 1.0, 3.0]\n",
    "\n",
    "results = []\n",
    "for eps in eps_to_try:\n",
    "    noisy_count = dp_count(true_ai_tutor_count, epsilon=eps)\n",
    "    noisy_mean = dp_mean(df[\"stress_score\"], epsilon=eps, lower=1, upper=10)\n",
    "    results.append({\n",
    "        \"epsilon\": eps,\n",
    "        \"noisy_count_yes\": round(noisy_count, 2),\n",
    "        \"abs_error_count\": round(abs(noisy_count - true_ai_tutor_count), 2),\n",
    "        \"noisy_mean_stress\": round(noisy_mean, 3),\n",
    "        \"abs_error_mean\": round(abs(noisy_mean - true_mean_stress), 3),\n",
    "    })\n",
    "\n",
    "pd.DataFrame(results)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e705bfd0",
   "metadata": {},
   "source": [
    "## Exercise 1B — Sensitivity and bounds\n",
    "\n",
    "The mean mechanism depends on the lower and upper bounds.\n",
    "\n",
    "**Your task**\n",
    "1. Change the lower/upper bounds in the next cell.\n",
    "2. Observe how the noisy mean changes.\n",
    "3. Explain why **tighter valid bounds** are usually better for utility."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d8f6d338",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# TODO: experiment with different valid bounds\n",
    "lower_bound = 1\n",
    "upper_bound = 10\n",
    "\n",
    "dp_mean(df[\"stress_score\"], epsilon=0.5, lower=lower_bound, upper=upper_bound)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "58db12fb",
   "metadata": {},
   "source": [
    "## Exercise 1C — Short reflection\n",
    "\n",
    "Answer in a text cell below:\n",
    "\n",
    "1. Why does adding noise to a *summary statistic* differ from publishing the raw dataset?\n",
    "2. Why is differential privacy about a **formal guarantee**, not just “making the data fuzzy”?\n",
    "3. Why would applying DP to **model training** usually be more computationally expensive than applying DP to a simple count query?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4b36870c",
   "metadata": {},
   "source": [
    "### Takeaway\n",
    "In this section, we privatized **statistics about a dataset**. That is not the same as training a model with a differential privacy guarantee. In practice, **DP training** (for example, DP-SGD) is usually more expensive and can be difficult for large transformer models on limited hardware."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "31959171",
   "metadata": {},
   "source": [
    "---\n",
    "# Section 2 — Fine-tuning a tiny BERT model for text classification\n",
    "\n",
    "We will fine-tune a compact BERT-family model on a small slice of the **AG News** dataset.\n",
    "\n",
    "## Why a tiny model?\n",
    "A full BERT fine-tune can still be heavy for some free Colab sessions. To keep this lab practical, we use:\n",
    "\n",
    "- model: `prajjwal1/bert-tiny`\n",
    "- task: news classification\n",
    "- short run: a small dataset slice and few training steps"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fac0a938",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "raw_news = load_dataset(\"ag_news\")\n",
    "label_names = raw_news[\"train\"].features[\"label\"].names\n",
    "label_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "359b575b",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "train_news = raw_news[\"train\"].shuffle(seed=42).select(range(BERT_TRAIN_SIZE))\n",
    "test_news = raw_news[\"test\"].shuffle(seed=42).select(range(BERT_TEST_SIZE))\n",
    "\n",
    "print(\"Train size:\", len(train_news))\n",
    "print(\"Test size:\", len(test_news))\n",
    "print(\"\\nExample:\")\n",
    "print(train_news[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53eb6050",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "bert_model_name = \"prajjwal1/bert-tiny\"\n",
    "bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)\n",
    "\n",
    "def tokenize_for_bert(batch):\n",
    "    return bert_tokenizer(batch[\"text\"], truncation=True, max_length=128)\n",
    "\n",
    "tokenized_train_news = train_news.map(tokenize_for_bert, batched=True)\n",
    "tokenized_test_news = test_news.map(tokenize_for_bert, batched=True)\n",
    "\n",
    "data_collator_bert = DataCollatorWithPadding(tokenizer=bert_tokenizer)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2d6b5634",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def compute_accuracy(eval_pred):\n",
    "    logits, labels = eval_pred\n",
    "    preds = np.argmax(logits, axis=-1)\n",
    "    return {\"accuracy\": accuracy_score(labels, preds)}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "912e6c1c",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "bert_model = AutoModelForSequenceClassification.from_pretrained(\n",
    "    bert_model_name,\n",
    "    num_labels=len(label_names)\n",
    ")\n",
    "\n",
    "bert_args = TrainingArguments(\n",
    "    output_dir=\"bert_tiny_ag_news\",\n",
    "    overwrite_output_dir=True,\n",
    "    max_steps=BERT_MAX_STEPS,\n",
    "    per_device_train_batch_size=16 if device == \"cuda\" else 8,\n",
    "    per_device_eval_batch_size=32 if device == \"cuda\" else 8,\n",
    "    learning_rate=3e-5,\n",
    "    weight_decay=0.01,\n",
    "    evaluation_strategy=\"epoch\",\n",
    "    save_strategy=\"no\",\n",
    "    logging_strategy=\"steps\",\n",
    "    logging_steps=10,\n",
    "    fp16=(device == \"cuda\"),\n",
    "    report_to=\"none\",\n",
    ")\n",
    "\n",
    "bert_trainer = Trainer(\n",
    "    model=bert_model,\n",
    "    args=bert_args,\n",
    "    train_dataset=tokenized_train_news,\n",
    "    eval_dataset=tokenized_test_news,\n",
    "    tokenizer=bert_tokenizer,\n",
    "    data_collator=data_collator_bert,\n",
    "    compute_metrics=compute_accuracy,\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c52236b3",
   "metadata": {},
   "source": [
    "## 2.1 Train the BERT classifier\n",
    "Run the next cell. On a free GPU this should be manageable because the run is intentionally short."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7b8537c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "bert_train_result = bert_trainer.train()\n",
    "bert_eval_result = bert_trainer.evaluate()\n",
    "\n",
    "print(\"BERT evaluation results:\")\n",
    "bert_eval_result"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c90bd362",
   "metadata": {},
   "source": [
    "## 2.2 Inspect predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1ea90518",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "sample_indices = [0, 1, 2, 3, 4]\n",
    "\n",
    "for idx in sample_indices:\n",
    "    item = test_news[idx]\n",
    "    inputs = bert_tokenizer(item[\"text\"], return_tensors=\"pt\", truncation=True, max_length=128)\n",
    "    inputs = {k: v.to(bert_model.device) for k, v in inputs.items()}\n",
    "    with torch.no_grad():\n",
    "        logits = bert_model(**inputs).logits\n",
    "    pred_id = int(logits.argmax(dim=-1).cpu().item())\n",
    "\n",
    "    print(\"=\" * 80)\n",
    "    print(\"Text:\", item[\"text\"][:300], \"...\")\n",
    "    print(\"True label:\", label_names[item[\"label\"]])\n",
    "    print(\"Predicted :\", label_names[pred_id])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b17bbf97",
   "metadata": {},
   "source": [
    "## Exercise 2A — Improve or stress-test the classifier\n",
    "\n",
    "Try one or more of the following:\n",
    "\n",
    "1. Increase `BERT_TRAIN_SIZE`.\n",
    "2. Increase `BERT_MAX_STEPS`.\n",
    "3. Change the learning rate.\n",
    "4. Replace `prajjwal1/bert-tiny` with another compact model such as a DistilBERT variant.\n",
    "\n",
    "**Questions**\n",
    "- Does accuracy improve?\n",
    "- Does training time increase?\n",
    "- At what point does the Colab runtime feel too slow?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ab69f7ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# TODO: duplicate the training cell above and change one setting at a time.\n",
    "# Suggested starting point:\n",
    "# - BERT_TRAIN_SIZE = 2000\n",
    "# - BERT_MAX_STEPS = 120\n",
    "# - learning_rate = 2e-5"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ccd03b1c",
   "metadata": {},
   "source": [
    "## Exercise 2B — Error analysis\n",
    "\n",
    "Pick 3 wrongly classified examples and answer:\n",
    "\n",
    "1. Why might the model have failed?\n",
    "2. Is the article ambiguous?\n",
    "3. Would more data or a larger model likely help?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7b04587d",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# Find some incorrect predictions for inspection\n",
    "pred_output = bert_trainer.predict(tokenized_test_news)\n",
    "preds = np.argmax(pred_output.predictions, axis=-1)\n",
    "wrong = np.where(preds != np.array(tokenized_test_news[\"label\"]))[0][:10]\n",
    "\n",
    "wrong[:10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6da533d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# TODO: inspect some misclassified examples\n",
    "for idx in wrong[:3]:\n",
    "    item = test_news[int(idx)]\n",
    "    print(\"=\" * 80)\n",
    "    print(\"Text:\", item[\"text\"][:300], \"...\")\n",
    "    print(\"True label:\", label_names[item[\"label\"]])\n",
    "    print(\"Predicted :\", label_names[int(preds[idx])])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "70dad729",
   "metadata": {},
   "source": [
    "---\n",
    "# Section 3 — Fine-tuning a tiny GPT-style model for causal language modeling\n",
    "\n",
    "Now we fine-tune a **small GPT-style model** using a compact text corpus derived from AG News.\n",
    "\n",
    "## Goal\n",
    "The model will learn to continue text in a simple news-like style.\n",
    "\n",
    "## Why use a tiny GPT model?\n",
    "To keep the lab runnable on the free tier, we use:\n",
    "- model: `sshleifer/tiny-gpt2`\n",
    "- short text sequences\n",
    "- LoRA adapters so that only a small number of parameters are trained"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f9a995c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def build_lm_example(example):\n",
    "    label_text = label_names[example[\"label\"]]\n",
    "    text = f\"Category: {label_text}\\nArticle: {example['text']}\\n\"\n",
    "    return {\"text\": text}\n",
    "\n",
    "lm_train = train_news.select(range(min(GPT_TRAIN_SIZE, len(train_news)))).map(\n",
    "    build_lm_example,\n",
    "    remove_columns=train_news.column_names\n",
    ")\n",
    "\n",
    "lm_valid = test_news.select(range(min(GPT_VALID_SIZE, len(test_news)))).map(\n",
    "    build_lm_example,\n",
    "    remove_columns=test_news.column_names\n",
    ")\n",
    "\n",
    "print(lm_train[0][\"text\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aa61b0c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "gpt_model_name = \"sshleifer/tiny-gpt2\"\n",
    "gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)\n",
    "\n",
    "if gpt_tokenizer.pad_token is None:\n",
    "    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token\n",
    "\n",
    "def tokenize_for_gpt(batch):\n",
    "    return gpt_tokenizer(batch[\"text\"], truncation=True, max_length=128)\n",
    "\n",
    "tokenized_lm_train = lm_train.map(tokenize_for_gpt, batched=True, remove_columns=[\"text\"])\n",
    "tokenized_lm_valid = lm_valid.map(tokenize_for_gpt, batched=True, remove_columns=[\"text\"])\n",
    "\n",
    "lm_data_collator = DataCollatorForLanguageModeling(tokenizer=gpt_tokenizer, mlm=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "88e31bf3",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "gpt_model = AutoModelForCausalLM.from_pretrained(gpt_model_name)\n",
    "gpt_model.config.pad_token_id = gpt_tokenizer.eos_token_id\n",
    "\n",
    "lora_config = LoraConfig(\n",
    "    task_type=TaskType.CAUSAL_LM,\n",
    "    r=4,\n",
    "    lora_alpha=16,\n",
    "    lora_dropout=0.1,\n",
    "    target_modules=[\"c_attn\"],\n",
    ")\n",
    "\n",
    "gpt_model = get_peft_model(gpt_model, lora_config)\n",
    "gpt_model.print_trainable_parameters()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4784fb09",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "gpt_args = TrainingArguments(\n",
    "    output_dir=\"tiny_gpt2_lora_ag_news\",\n",
    "    overwrite_output_dir=True,\n",
    "    max_steps=GPT_MAX_STEPS,\n",
    "    per_device_train_batch_size=8 if device == \"cuda\" else 2,\n",
    "    per_device_eval_batch_size=8 if device == \"cuda\" else 2,\n",
    "    gradient_accumulation_steps=2 if device == \"cuda\" else 4,\n",
    "    learning_rate=5e-4,\n",
    "    weight_decay=0.01,\n",
    "    evaluation_strategy=\"no\",\n",
    "    save_strategy=\"no\",\n",
    "    logging_strategy=\"steps\",\n",
    "    logging_steps=10,\n",
    "    fp16=(device == \"cuda\"),\n",
    "    report_to=\"none\",\n",
    ")\n",
    "\n",
    "gpt_trainer = Trainer(\n",
    "    model=gpt_model,\n",
    "    args=gpt_args,\n",
    "    train_dataset=tokenized_lm_train,\n",
    "    eval_dataset=tokenized_lm_valid,\n",
    "    tokenizer=gpt_tokenizer,\n",
    "    data_collator=lm_data_collator,\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6a195365",
   "metadata": {},
   "source": [
    "## 3.1 Train the GPT-style model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ce1ce122",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "gpt_train_result = gpt_trainer.train()\n",
    "print(\"Finished GPT-style fine-tuning.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "37d390e3",
   "metadata": {},
   "source": [
    "## 3.2 Generate text\n",
    "Use the trained model to generate short article continuations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b63759d2",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "gpt_model.eval()\n",
    "\n",
    "prompts = [\n",
    "    \"Category: Sci/Tech\\nArticle:\",\n",
    "    \"Category: Business\\nArticle:\",\n",
    "    \"Category: Sports\\nArticle:\",\n",
    "]\n",
    "\n",
    "for prompt in prompts:\n",
    "    print(\"=\" * 80)\n",
    "    print(\"PROMPT:\")\n",
    "    print(prompt)\n",
    "\n",
    "    inputs = gpt_tokenizer(prompt, return_tensors=\"pt\").to(gpt_model.device)\n",
    "    with torch.no_grad():\n",
    "        output_ids = gpt_model.generate(\n",
    "            **inputs,\n",
    "            max_new_tokens=50,\n",
    "            do_sample=True,\n",
    "            temperature=0.8,\n",
    "            top_p=0.95,\n",
    "            pad_token_id=gpt_tokenizer.eos_token_id,\n",
    "        )\n",
    "    print(\"\\nGENERATED TEXT:\")\n",
    "    print(gpt_tokenizer.decode(output_ids[0], skip_special_tokens=True))\n",
    "    print()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "161aecc6",
   "metadata": {},
   "source": [
    "## Exercise 3A — Prompt engineering and generation settings\n",
    "\n",
    "Try changing one thing at a time:\n",
    "\n",
    "1. Change the category prompt.\n",
    "2. Change `temperature`.\n",
    "3. Change `top_p`.\n",
    "4. Change `max_new_tokens`.\n",
    "\n",
    "**Questions**\n",
    "- Which settings make the output more repetitive?\n",
    "- Which settings make it more diverse?\n",
    "- When does the model start to drift off-topic?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0310a257",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# TODO: edit and re-run\n",
    "custom_prompt = \"Category: World\\nArticle:\"\n",
    "custom_temperature = 0.7\n",
    "custom_top_p = 0.9\n",
    "custom_max_new_tokens = 60\n",
    "\n",
    "inputs = gpt_tokenizer(custom_prompt, return_tensors=\"pt\").to(gpt_model.device)\n",
    "\n",
    "with torch.no_grad():\n",
    "    output_ids = gpt_model.generate(\n",
    "        **inputs,\n",
    "        max_new_tokens=custom_max_new_tokens,\n",
    "        do_sample=True,\n",
    "        temperature=custom_temperature,\n",
    "        top_p=custom_top_p,\n",
    "        pad_token_id=gpt_tokenizer.eos_token_id,\n",
    "    )\n",
    "\n",
    "print(gpt_tokenizer.decode(output_ids[0], skip_special_tokens=True))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "31bb17eb",
   "metadata": {},
   "source": [
    "## Exercise 3B — Compare BERT and GPT-style fine-tuning\n",
    "\n",
    "Answer in a text cell:\n",
    "\n",
    "1. What is the **task difference** between the BERT model and the GPT-style model in this notebook?\n",
    "2. Why is BERT well suited to classification?\n",
    "3. Why is a GPT-style model well suited to generation?\n",
    "4. Which model would you trust more for a factual downstream task, and why?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "76b47b41",
   "metadata": {},
   "source": [
    "---\n",
    "# Section 4 — Wrap-up\n",
    "\n",
    "## What you learned\n",
    "You have now:\n",
    "- released **DP-noisy dataset statistics**,\n",
    "- fine-tuned a **tiny BERT** model for classification,\n",
    "- fine-tuned a **tiny GPT-style** model for generation,\n",
    "- reflected on why **DP data release** and **DP model training** are related but different problems.\n",
    "\n",
    "## Key distinction\n",
    "- **Section 1:** privacy was applied to the **released dataset statistics**.\n",
    "- **Sections 2–3:** models were fine-tuned **without** a formal differential privacy guarantee.\n",
    "\n",
    "## Optional extension ideas\n",
    "If you have more time and a stronger runtime, try one of these:\n",
    "\n",
    "1. Use a larger dataset slice and compare performance vs runtime.\n",
    "2. Replace `bert-tiny` with a DistilBERT model.\n",
    "3. Replace `tiny-gpt2` with `distilgpt2` and keep the run short.\n",
    "4. Research how **DP-SGD** would change the training process.\n",
    "5. Add a small report section discussing trade-offs among privacy, accuracy, fluency, runtime, and memory.\n",
    "\n",
    "## Submission suggestion\n",
    "For students, a good submission could include:\n",
    "- screenshots or saved outputs from each main section,\n",
    "- short written answers to the exercise prompts,\n",
    "- one paragraph comparing privacy-preserving data release with private model training."
   ]
  }
 ],
 "metadata": {
  "colab": {
   "include_colab_link": true,
   "name": "dp_bert_gpt_colab_lab.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
