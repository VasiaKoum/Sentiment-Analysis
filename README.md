# Sentiment-Analysis
#### Data Mining, Project 1:
 *Preprocessing(clean tweets), create model to predict sentiment-label(positive, negative or neutral) for tweets and model accuracy checking.*

Contributors
------------

* [Vasiliki Koumarela](https://github.com/VasiaKoum/ "Vasiliki Koumarela")
* [Lazaros Avgeridis](https://github.com/lazavgeridis/ "Lazaros Avgeridis")

System requirements
-------------------

* Python version 3.6
* NLTK
* Numpy

Run commands
------------
```bash
jupyter notebook
pip install --user -U nltk
pip install --user -U numpy
pip install vaderSentiment
```

Implementation
--------------
* Cleaning the data(process tweets from train.tsv and test.tsv using: Tokenization, StopWord filtering, Stemming)
* Make workclouds and matplots for the data
* Vectorization(using: BAG-OF-WORDS & TF-IDF)
* TSNE model(Word2vec)
* Classification: KNN , SVM
* Check the accuracy from label predictions

*Use f1_score to calculate the success rate (classification labels with the official labels of test tweets SemEval2017_task4_subtaskA_test_english_gold.txt)*

__Model Accuracy__: 0.59 success
