# Data Manipulation / Linear Algebra
import tokenize

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# NLP / Text Manipulation
import regex
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Machine Learning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, \
    RidgeClassifierCV, SGDOneClassSVM
from sklearn.model_selection import train_test_split
from fast_ml.model_development import train_valid_test_split
from nltk.tokenize.regexp import RegexpTokenizer

from sklearn.linear_model._logistic import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download("stopwords")
nltk.download("punkt")

data = pd.read_csv('dataset/imdb.csv')

print(data.head(10))

plt.hist(data[data.sentiment == 1].sentiment,
         bins=2, color='green', label='Positive')
plt.hist(data[data.sentiment == 0].sentiment,
         bins=2, color='blue', label='Negative')
plt.title('Classes distribution in the train data', fontsize=12)
plt.xticks([])
plt.xlim(-0.5, 2)
plt.legend()
plt.savefig('images/00_bnb_mnb_lr_Classes_distribution_in_the_train data.png')


def cleanText(text):
    # Remove HTML tags
    text = regex.sub(r"<[^<]+?>", "", text)

    # Remove Special chars
    text = regex.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Convet to LowerCase
    text = text.lower()

    return text


# Applying the function to data
data["review"] = data["review"].apply(cleanText)
#
# print(data.head(10))
#
# Get list of stopwords from nltk
stopword_list = stopwords.words('english')


#
#
# Tokenize and Remove StopWords
def remove_stopwords(text):
    tokens = [token.strip() for token in word_tokenize(text)]
    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text


#
#
# Applying the function to remove stopwords
data['review'] = data['review'].apply(remove_stopwords)
print(data.head(10))

# # PorterStemmer to Convert the words to base form
# def simple_stemmer(text):
#     ps = PorterStemmer()
#     text = ' '.join([ps.stem(word) for word in text.split()])
#     return text
#
#
# # Applying the function to stemm words
# data['review'] = data['review'].apply(simple_stemmer)
# print(data.head(10))

# data.head()
print(data.head())

# Seperating Features and Target Variable
X = data['review']
y = data['sentiment']

token = RegexpTokenizer(r'[^a-zA-Z0-9\s]+')
# Vectorizing the text
vect = CountVectorizer(ngram_range=(1, 1), max_features=20000)
X = vect.fit_transform(X)
print(X)

X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=0.25, random_state=42, stratify=y_main,
                                                  shuffle=True)

print("X_main shape: {}".format(X_train.shape))
print("y_main shape: {}".format(y_train.shape))
#

print("X_val shape: {}".format(X_val.shape))
print("y_val shape: {}".format(y_val.shape))


def classifier_testing(clf, X_train, X_val, y_train, y_val):
    # Training the Classifier
    clf.fit(X_train, y_train)

    # Getting Predictions
    y_pred = clf.predict(X_val)

    # Accuracy Score
    clf_accuracy_score = accuracy_score(y_val, y_pred)
    print("Accuracy Score:\n", clf_accuracy_score, "\n")

    # Classification Report
    class_rep = classification_report(y_val, y_pred)
    print("Classification Report:\n", class_rep, "\n")

    # Confusion Matrix
    conf_mtx = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix:\n", conf_mtx, "\n")


# GNB = GaussianNB()
# classifier_testing(GNB, X_train, X_test, y_train, y_test)

BNB = BernoulliNB(alpha=0, binarize=0.0, fit_prior=True, class_prior=None)
print('BernoulliNB')
classifier_testing(BNB, X_train, X_val, y_train, y_val)
print('---------------------------------------------------------------------------')

MNB = MultinomialNB(alpha=0, fit_prior=True, class_prior=None)
print('MultinomialNB')
classifier_testing(MNB, X_train, X_val, y_train, y_val)
print('---------------------------------------------------------------------------')

LR = LogisticRegression(random_state=42, max_iter=5)
print('LogisticRegression')
classifier_testing(LR, X_train, X_val, y_train, y_val)
print('---------------------------------------------------------------------------')

SGD = SGDClassifier(random_state=42, shuffle=True, max_iter=5)
print('SGDClassifier')
classifier_testing(SGD, X_train, X_val, y_train, y_val)
print('---------------------------------------------------------------------------')

SVM = SGDOneClassSVM(random_state=42, max_iter=5, shuffle=True)
print('SGDOneClassSVM')
classifier_testing(SVM, X_train, X_val, y_train, y_val)
print('---------------------------------------------------------------------------')

PAC = PassiveAggressiveClassifier(random_state=42, shuffle=True, max_iter=5)
print('PassiveAggressiveClassifier')
classifier_testing(SGD, X_train, X_val, y_train, y_val)
print('---------------------------------------------------------------------------')

PCT = Perceptron(random_state=42, shuffle=True, max_iter=5)
print('Perceptron')
classifier_testing(PCT, X_train, X_val, y_train, y_val)
print('---------------------------------------------------------------------------')

RC = RidgeClassifier(random_state=42, max_iter=5)
print('RidgeClassifier')
classifier_testing(RC, X_train, X_val, y_train, y_val)
print('---------------------------------------------------------------------------')
