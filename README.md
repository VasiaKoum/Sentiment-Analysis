# Sentiment-Analysis
#### Data Mining, Project 1

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
Processing for tweets from train.tsv and test.tsv:
Tokenization, StopWord filtering, Stemming, Vectorization with BAG-OF-WORDS & TF-IDF

Make model with all tweets: Word2vec

Classification for tweets from test.tsv: KNN, SVM

Use f1_score to calculate the success rate (classification labels with the official labels of test tweets SemEval2017_task4_subtaskA_test_english_gold.txt) 

With this model: 0.59 success
