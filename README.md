# **Loan Word Classification Using Fine-Tuned BERT**

## **1. Project Overview**
Loan words are words borrowed from one language and incorporated into another. The main goal of this project is to **detect and classify loan words** using a **fine-tuned multilingual BERT (mBERT) model**.

By leveraging **natural language processing (NLP)** and **deep learning**, we:  
- Train a model to differentiate between loan words and native words.  
- Fine-tune BERT on word pairs from multiple languages.  
- Develop a **Loan Word Classifier** that detects loan words in text.  

## **2. Data Understanding**
The dataset consists of **loan words from various language pairs**, categorized into:  
- **Hard negatives (False loan words)**  
- **Loan words**  
- **Synonyms**  

To simplify classification, we **merge** these into two classes:  
- **False Loan Word (1):** Hard negatives  
- **Loan Word (0):** Loan words, synonyms, and actual words  

## **3. Model Explanation**
### **Fine-Tuning mBERT**
- The fine-tuning is performed in **`training_bert.ipynb`**.  
- We train on **word pairs** (loan word and actual word) across different **language pairs**.  
- Performance is evaluated using **Accuracy & F1-score**.  
- The results of accuracy and F1-score for different languages are stored in **`bert_result.ipynb`**.  
- The CSV file is also stored by name **bert_result.csv**
The model is trained on the following **language pairs**:  

| **Language Pair**           | **Accuracy** | **F1-score** |
|-----------------------------|-------------|--------------|
| Azerbaijani - Arabic        | 97.32%      | 0.9722       |
| Catalan - Arabic            | 94.44%      | 0.9175       |
| Chinese - English           | 94.21%      | 0.9329       |
| English - French            | 92.02%      | 0.9062       |
| English - German            | 91.16%      | 0.9116       |
| Finnish - Swedish           | 92.80%      | 0.9186       |
| German - French             | 91.54%      | 0.9074       |
| German - Italian            | 88.00%      | 0.8800       |
| Hindi - Persian             | 91.57%      | 0.9061       |
| Hungarian - German          | 90.74%      | 0.9136       |
| Indonesian - Dutch          | 90.26%      | 0.8929       |
| Kazakh - Russian            | 94.78%      | 0.9410       |
| Persian - Arabic            | 92.79%      | 0.9155       |
| Polish - French             | 93.66%      | 0.9282       |
| Romanian - French           | 92.98%      | 0.9208       |
| Romanian - Hungarian        | 94.86%      | 0.9441       |


### **Loan Word Classifier**
- After fine-tuning, we **train a classifier** in **`training_classifier.ipynb`**.  
- The classifier identifies loan words in a given **sentence or paragraph**.  

## **4. Training and Evaluation**
- The classifier model is trained using **loan words and actual words**.  
- Validation and training accuracy are recorded.  
- Performance is assessed using **classification metrics** (Accuracy & F1-score).  
- Given a **string of words**, the model predicts which words are **loan words**.  

### **Example Prediction**

```python
sentence = "The government governed a new abordage policy."

The        tensor([[0.8167, 0.1833]], device='cuda:0')
government tensor([[0.5134, 0.4866]], device='cuda:0')
governed   tensor([[0.7013, 0.2987]], device='cuda:0')
a          tensor([[0.6716, 0.3284]], device='cuda:0')
new        tensor([[0.6311, 0.3689]], device='cuda:0')
abordage   tensor([[0.4892, 0.5108]], device='cuda:0')
policy.    tensor([[0.7452, 0.2548]], device='cuda:0')


false loan word is ['abordage']
```

### Files in the Repository:
- `training_bert.ipynb`: Fine-tuning mBERT.
- `training_classifier.ipynb`: Training and validating the Loan Word Classifier.

## 5. How to Use the Classifier for False Loan Word Detection

### Step 1: Download the Model

- Download the pre-trained model from Google Drive: [Link Here] (replace with actual link).

- Save it in the appropriate directory.

### Step 2: Run the Classifier
- Open ```classifier.ipynb.```
- Input a sentence.
- The model will predict which words are **false loan words**.

## 6. Usage
- Use the fine-tuned mBERT model for loan word detection.
- Apply the Loan Word Classifier to identify loan words in given texts.

---
This project helps in **loan word identification** across multiple languages by leveraging **multilingual BERT** and a custom classification model.


