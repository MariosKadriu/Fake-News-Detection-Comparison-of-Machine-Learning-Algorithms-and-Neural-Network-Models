# Fake News Detection Comparison of Machine Learning Algorithms and Neural Network Models

## Table of Contents

[Description](https://github.com/MariosKadriu/Fake-News-Detection-Comparison-of-Machine-Learning-Algorithms-and-Neural-Network-Models#-description)
 
[Dataset](https://github.com/MariosKadriu/Fake-News-Detection-Comparison-of-Machine-Learning-Algorithms-and-Neural-Network-Models#-dataset)

[Installation](https://github.com/MariosKadriu/Fake-News-Detection-Comparison-of-Machine-Learning-Algorithms-and-Neural-Network-Models#%EF%B8%8F-installation)

[Exploratory Data Analysis](https://github.com/MariosKadriu/Fake-News-Detection-Comparison-of-Machine-Learning-Algorithms-and-Neural-Network-Models#-exploratory-data-analysis)

[Data Preprocessing](https://github.com/MariosKadriu/Fake-News-Detection-Comparison-of-Machine-Learning-Algorithms-and-Neural-Network-Models#-data-preprocessing)

[Modeling and Evaluation](https://github.com/MariosKadriu/Fake-News-Detection-Comparison-of-Machine-Learning-Algorithms-and-Neural-Network-Models#-modeling-and-evaluation)

[Conclusions](https://github.com/MariosKadriu/Fake-News-Detection-Comparison-of-Machine-Learning-Algorithms-and-Neural-Network-Models#-conclusions)

## 📝 Description

The above project was implemented in the context of the course Text Mining and Natural Language Processing during my postgraduate studies in the Data and Web Science program of the Aristotle University of Thessaloniki.    Its main objective was to compare machine learning algorithms and neural models such as LSTM and Bi-LSTM.

## 📚 Dataset

The dataset used can be found here: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## 🖥️ Installation

### 🛠️ Requirements
* Python >= 3.6
* NLTK
* NumPy
* Matplotlib
* Seaborn
* Wordcloud
* Spacy
* Beautiful Soup
* TextBlob
* Gensim
* TensorFlow
* Scikit-learn

### ⚙️ Setup

All of the above packages can be installed from the following commands.

```bash
pip install nltk
pip install numpy
pip install matplotlib
pip install seaborn
pip install wordcloud
pip install gensim
pip install tensorflow
pip install -U scikit-learn

pip install spacy
python -m spacy download en_core_web_sm
pip install beautifulsoup4==4.9.1
pip install textblob==0.15.3
pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force-reinstall

```

## 🔍 Exploratory Data Analysis

## Fake News

Counts of its subject on fake dataset with descending order

![image-1](https://user-images.githubusercontent.com/19438003/190975248-54f1f5a7-47c4-48f9-955c-6c326d15b6ee.png)

Fake text data visualization

![image-2](https://user-images.githubusercontent.com/19438003/190975991-9daaeb62-717d-4879-9842-44fdd0b0664f.png)


## Real News

Counts of its subject on real dataset with descending order

![image-3](https://user-images.githubusercontent.com/19438003/190975512-48c20732-9665-4ef0-9bbc-29342c78ab20.png)

Real text data visualization

![image-4](https://user-images.githubusercontent.com/19438003/190976298-65ffeda3-2476-432f-b009-950c44700294.png)


## ⏳ Data Preprocessing

* Remove reuters or tweets information from the text column
* Drop rows from fake/real dataset if their text column is empty
* Extract publisher information from the text data and save it to a new column
* Merge title and text columns of fake/real dataset on text column
* Create the class column for fake and real dataset
* Combine fake and real dataset together
* Remove special characters like (,.#@)
* Convert each word into a sequence of 100 vectors
* Convert each word of text data into a sequence like (1, 2, 3, 4, 5)
* Make every sequence at the length of 1000 words

## 🎯 Modeling and Evaluation

### Models

* LSTM (128 Units)
* Bi-LSTM (128 Units)
* Gaussian Naive Bayes
* Random Forest
* Decision Trees
* Logistic Regression
* Linear Support Vector Machines
* K-nearest Neighbors
* Ada Boost
* XGboost
* Voting Classifier (Random Forests, Logistic Regression, KNN)
* Voting Classifier (Random Forests, Decision Trees, Gaussian Naive Bayes)

### Evaluation Metrics

* Total Training Time
* Accuracy, Precision, Recall, F1-Score
* Confusion Matrix

### Results

#### Training Time
LSTM and Bi-LSTM were trained on RTX 3060 Laptop GPU

![image-5](https://user-images.githubusercontent.com/19438003/190997141-bd2f0817-8b44-4e6f-9c07-017ffe75ce41.jpg)

#### Scores
![image-6](https://user-images.githubusercontent.com/19438003/190997226-9fd01ebf-f9a6-4ecc-98b2-faa96354bd2a.jpg)


## 💡 Conclusions
* The Bidirectional LSTM and LSTM models fared the best
* XGBoost is a comparable decent choice, with minor classification sacrifices but faster training rates
* Despite its rapid learning rate, the Linear Support Vector Machines method performed moderately and poorly compared to the other models
* Both classes are classified more precisely by Bidirectional LSTM, LSTM, XGboost, and the Voting Classifier (Random Forest, Decision Trees, and Gaussian Naive Bayes). The rest of the models have some trouble discriminating the real from the fake news and vice versa.
