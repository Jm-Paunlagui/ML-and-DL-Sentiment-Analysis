import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import nltk
import re
import regex
import io
import os
import nltk
from keras.optimizers import Adam

from wordcloud import WordCloud
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM, GlobalMaxPool1D, Dropout
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phrases

train_data = pd.read_csv('dataset/imdb.csv')

X = train_data['review']
y = train_data['sentiment']

X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y, shuffle=True)

print("X_main shape: {}".format(X_main.shape))
print("y_main shape: {}".format(y_main.shape))

print("X_val shape: {}".format(X_test.shape))
print("y_val shape: {}".format(y_test.shape))


plt.hist(train_data[train_data.sentiment == 1].sentiment,
         bins=2, color='green', label='Positive')
plt.hist(train_data[train_data.sentiment == 0].sentiment,
         bins=2, color='blue', label='Negative')
plt.title('Classes distribution in the train data', fontsize=12)
plt.xticks([])
plt.xlim(-0.5, 2)
plt.legend()
plt.show()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

REPLACE_WITH_SPACE = re.compile(r'[^A-Za-z\s]')
counter = 0


def clean_review(raw_review: str) -> str:
    # remove html
    review_text = re.sub(r'[^\w\s]', '', raw_review, re.UNICODE)

    # remove non-letters
    letters_only = REPLACE_WITH_SPACE.sub(" ", review_text)

    # 3. Convert to lower case
    lowercase_letters = letters_only.lower()
    return lowercase_letters


def lemmatize(tokens: list) -> list:
    tokens = list(map(lemmatizer.lemmatize, tokens))
    lemmatized_tokens = list(map(lambda x: lemmatizer.lemmatize(x, "v"), tokens))

    meaningful_words = list(filter(lambda x: not x in stop_words, lemmatized_tokens))
    return meaningful_words


def preprocess(review: str, total: int, show_progress: bool = True) -> list:
    if show_progress:
        global counter
        counter += 1
        print('Processing... %6i/%6i' % (counter, total), end='\r')
    # 1. Clean text
    review = clean_review(review)
    # 2. Split into individual words
    tokens = word_tokenize(review)
    # 3. Lemmatize
    lemmas = lemmatize(tokens)
    # 4. Join the words back into one string separated by space,
    # and return the result.
    return lemmas


train_data = np.array(list(map(lambda x: preprocess(x, len(train_data)), train_data)))
print(train_data)

X_train_data = train_data[:train_data.shape[0]]


train_data['review_lenght'] = np.array(list(map(len, X_train_data)))
median = train_data['review_lenght'].median()
mean = train_data['review_lenght'].mean()
mode = train_data['review_lenght'].mode()[0]

fig, ax = plt.subplots()
sns.distplot(train_data['review_lenght'], bins=train_data['review_lenght'].max(),
             hist_kws={"alpha": 0.9, "color": "blue"}, ax=ax,
             kde_kws={"color": "black", 'linewidth': 3})
ax.set_xlim(left=0, right=np.percentile(train_data['review_lenght'], 95))
ax.set_xlabel('Words in review')
ymax = 0.014
plt.ylim(0, ymax)
ax.plot([mode, mode], [0, ymax], '--', label=f'mode = {mode:.2f}', linewidth=4)
ax.plot([mean, mean], [0, ymax], '--', label=f'mean = {mean:.2f}', linewidth=4)
ax.plot([median, median], [0, ymax], '--',
        label=f'median = {median:.2f}', linewidth=4)
ax.set_title('Words per review distribution', fontsize=20)
plt.legend()
plt.show()
plt.savefig('images/words_per_review_distribution.png')

bigrams = Phrases(sentences=train_data)
trigrams = Phrases(sentences=bigrams[train_data])

trigrams = Phrases(sentences=bigrams[train_data])

embedding_vector_size = 256
trigrams_model = Word2Vec(
    sentences=trigrams[bigrams[train_data]],
    vector_size=embedding_vector_size,
    min_count=3, window=5, workers=4)

trigrams_model.save('00_bidi_lstm.model')
print("Vocabulary size:", len(trigrams_model.wv))

