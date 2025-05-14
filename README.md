# False Loan Word Detection

## **1. Project Overview**

This project focuses on detecting and classifying **false loan words** using a combination of fine-tuned transformer models, translation analysis, and similarity-based filtering. It leverages the power of **multilingual BERT (mBERT)** and **LLMs (Ollama)** to build a robust pipeline for identifying misinterpreted or falsely borrowed words across languages.

## **2. Objectives**

* Train a model to **differentiate loan words from native words**.
* Detect **false loan words** (i.e., words that appear to be borrowed but are not genuine loan words).
* Use **translation-based metrics** (BLEU, METEOR) to evaluate semantic drift.
* Use **embedding similarity across languages** to detect mismatches.

## **3. Datasets**

The dataset includes multilingual loan word pairs and their corresponding labels:

* **Loan Words (label: 0)**
* **False Loan Words / Hard Negatives (label: 1)**
* **Synonyms (label: 0)**

The dataset is preprocessed to merge the above into two classes for binary classification.

## **4. Training Details**

### **4.1 Fine-Tuning BERT**

* Model: `bert-base-multilingual-cased`
* Tokenization performed on the `loan_word_epitran` feature.
* Output files:

  * `training_bert.ipynb`: Fine-tunes mBERT on labeled pairs.
  * `bert_result.ipynb`: Stores and evaluates Accuracy and F1-Score per language pair.

| Language Pair        | Accuracy | F1-score |
| -------------------- | -------- | -------- |
| Azerbaijani - Arabic | 97.32%   | 0.9722   |
| English - German     | 91.16%   | 0.9116   |
| Romanian - Hungarian | 94.86%   | 0.9441   |

### **4.2 Classifier**

* File: `training_classifier.ipynb`
* Given a sentence, predicts each word’s probability of being a false loan word.

Example:

```python
sentence = "The government governed a new abordage policy."

# Output:
'abordage' is classified as the false loan word.
```

## **5. Translation-Based Validation**

File: `translation.ipynb`

* Generate a **German sentence** using the loan word.
* Translate it back to **English**.
* Use **BLEU** and **METEOR** scores to compare against the reference sentence.
* Metrics:

  * `bleu_score`
  * `meteor_score`

## **6. LLM-Based Detection**

File: `false_loanword_classifier.ipynb`

* Uses **Ollama LLM (LLaMA 3.2)** to predict false loan words from sentences.

Prompt example:

```
You are an expert linguistic model. 
        Given a sentence, identify if there is a false loanword (a word that seems borrowed but is wrongly used or misinterpreted).
        - If a false loanword is present, output only the false loanword (one word, no explanation).
        - If no false loanword is found, output exactly: no
        Sentence: {sentence}
        Output Response should must contain one single word only
        Output Format: Output_word
        Do donnott include any other thing just give one single word output!
```


## **7. False Loan Word Detection Results**

To further refine the detection of false loan words, we employed a post-classification filtering strategy based on:

* **Translation Quality Metrics (BLEU, METEOR)**
* **Embedding Similarity Scores (English vs. German)**
* **LLM-based Detection (Ollama with LLaMA-3)**

### **Lost in Translation Heuristic**

Using a heuristic that flags a word as a **false loan word** (i.e., "lost in translation") if:

* The **German embedding similarity** is higher than the English one.
* At least **3 out of 4** conditions were satisfied:

  * BLEU score < 0.5
  * METEOR score < 0.5
  * German embedding similarity > English similarity
  * The predicted word by the LLM is not `"false"`

We filtered the predictions accordingly:

```
final_score = 0.6579  # (i.e., ~65.79% accurate detection of lost in translation cases)
```

## **8. Key Finding**

> **German similarity dominance** in the embedding space strongly correlates with false loan word presence in translated sentences.
> Thus, a higher `sim_eng_ger[1]` (German) indicates that the German word is more semantically aligned, and the English term used might be a **false loan word**.

### **Conclusion**

With a final detection accuracy of **65.79%** under the proposed heuristic, this approach effectively isolates **false loan words** using a combination of **contextual translation metrics, embeddings, and LLM predictions**.

## **9. Final Files and Execution Order**

| File                              | Description                                      |
| --------------------------------- | ------------------------------------------------ |
| `training_bert.ipynb`             | Fine-tunes mBERT on labeled loan word pairs      |
| `bert_result.ipynb`               | Stores results across language pairs             |
| `training_classifier.ipynb`       | Trains sentence-level false loan word classifier |
| `translation.ipynb`               | Translates loan word sentences and scores them   |
| `false_loanword_classifier.ipynb` | Detects false loan words using mBERT + LLM       |

## **10. How to Use**

### **Step 1: Clone the Repo and Install Requirements**

```bash
pip install -r requirements.txt
```

### **Step 2: Fine-tune the Model**

Run `training_bert.ipynb` followed by `training_classifier.ipynb`

### **Step 3: Analyze with Translations**

Run `translation.ipynb` and generate BLEU/METEOR metrics.

### **Step 4: Run Classifier**

Use `false_loanword_classifier.ipynb` to test the model on new text.


### **Step 5: Filter Results**

Use `filtered_df` logic to shortlist true false loanwords.

## **11. Pretrained Model**

You can download pretrained models here:
[Google Drive Link](https://drive.google.com/drive/folders/10jFIIsZyGxEs9sq7v7rupOFzhzAJw55f?usp=sharing)

## **12. Future Work**

* Fine-tune deeper layers of BERT.
* Incorporate **XLM-RoBERTa**, **XLBERT**.
* Expand to more language pairs.
* Add **contextual paraphrase detection**.

---

This project builds a robust pipeline for multilingual loan word classification and lost-in-translation detection using a combination of machine learning and LLM reasoning. The integration of **translation metrics**, **embedding similarity**, and **LLM filtering** leads to strong precision in detecting semantic errors due to false loan words.
