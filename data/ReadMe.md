# **Loan Word Dataset Overview**

This dataset contains **loan words and their linguistic attributes** across multiple languages. It is structured into various folders, each serving a specific purpose in loan word classification.

## **1. Dataset Structure**

```
dataset/ 
│── hard-negatives/ 
│── loans/ 
│── production_train_test/ 
│── randoms/ 
│── synonyms/ 
```

## **2. Folder Descriptions**

### **📌 hard-negatives/**
This folder contains **false loan words (hard negatives)** with various phonetic and linguistic features.  
**Columns:**
- `loan_word, original_word, loan_word_epitran, original_word_epitran, loan_english, original_english`
- `Fast Levenshtein Distance Div Maxlen, Dolgo Prime Distance Div Maxlen`
- `Feature Edit Distance Div Maxlen, Hamming Feature Distance Div Maxlen`
- `Weighted Feature Distance Div Maxlen, Partial Hamming Feature Distance Div Maxlen`
- `plain Levenshtein, loan_unicode, original_unicode`
- `label`

### **📌 loans/**
This folder contains **actual loan words** with their respective linguistic features.  
**Columns (same as `hard-negatives`):**
- `loan_word, original_word, loan_word_epitran, original_word_epitran, loan_english, original_english`
- `Fast Levenshtein Distance Div Maxlen, Dolgo Prime Distance Div Maxlen`
- `Feature Edit Distance Div Maxlen, Hamming Feature Distance Div Maxlen`
- `Weighted Feature Distance Div Maxlen, Partial Hamming Feature Distance Div Maxlen`
- `plain Levenshtein, loan_unicode, original_unicode`
- `label`

### **📌 production_train_test/**
This folder contains **training and test data** used for the classification model.
- Organized into **two different language folders**.
- Each folder has three subsets:
  - `alldata/`
  - `balanced/`
  - `realdist/`
- **Columns:**
  - `Unnamed: 0, loan_word, original_word, loan_word_epitran, original_word_epitran, loan_english, original_english`
  - `Fast Levenshtein Distance Div Maxlen, Dolgo Prime Distance Div Maxlen`
  - `Feature Edit Distance Div Maxlen, Hamming Feature Distance Div Maxlen`
  - `Weighted Feature Distance Div Maxlen, Partial Hamming Feature Distance Div Maxlen`
  - `plain Levenshtein, loan_unicode, original_unicode`
  - `label, label_bin, DNN_logits, MBERT_cos_sim, XLM_cos_sim`

### **📌 randoms/**
Contains **randomly selected loan words** with phonetic and linguistic features.  
**Columns (same as `hard-negatives` & `loans`)**.


---

## **3. Purpose of This Dataset**
This dataset is designed to **train and evaluate** models for **loan word detection and classification**.  
- It includes **both false loan words and true loan words**.  
- Used for **fine-tuning BERT** to classify loan words in various languages.  
- Helps in **linguistic analysis, translation systems, and language learning applications**.

This dataset is an essential component of the **Loan Word Classifier**, where the trained model detects loan words in **sentences and paragraphs**.

---

## **4. How This Data Is Used**
- **Preprocessing**: Data is cleaned and merged into two classes (`False Loan Word (1)` and `Loan Word (0)`).  
- **Fine-tuning mBERT**: Word pairs are used to train a **multilingual BERT model** (`training_bert.ipynb`).  
- **Loan Word Classification**: The classifier is trained using fine-tuned BERT to detect **loan words in text** (`training_classifier.ipynb`).  

---

This dataset plays a crucial role in **enhancing NLP models for cross-linguistic loan word detection**.  
🚀 **For implementation details, refer to `README.md` in the main project folder!**  
