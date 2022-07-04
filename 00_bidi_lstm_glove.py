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
import gensim.downloader as api
from keras.optimizers import Adam

from wordcloud import WordCloud
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM, GlobalMaxPool1D, Dropout, SpatialDropout1D
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.word2vec import Word2Vec

stopword_list = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# glove-twitter-200
# glove_gensim = api.load('glove-wiki-gigaword-300')
glove_gensim = api.load('glove-twitter-200')

# nltk.download("stopwords")
# nltk.download("punkt")

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
plt.show();
REPLACE_WITH_SPACE = re.compile(r'[^A-Za-z\s]')


def clean_text(text):
    # Remove HTML tags
    text = regex.sub(r"<[^<]+?>", "", text)

    # Remove Special chars
    text = regex.sub(r'[^a-zA-Z0-9\s]', "", text)

    # Convet to LowerCase
    text = text.lower()

    return text


# Applying the function to data
data["review"] = data["review"].apply(clean_text)

print('clean text', data.head())

# Tokenize and Remove StopWords
def remove_stopwords(text):
    tokens = [token.strip() for token in word_tokenize(text)]
    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text


# Applying the function to remove stopwords
# data['review'] = data['review'].apply(remove_stopwords)
print('removed stop words', data.head())

meanu = data['review'].apply(lambda x: len(x.split(" "))).mean()
print(meanu)

# X_train_data = data['review']
# Y_train_data = data.sentiment.values
#
# data['review_lenght'] = np.array(list(map(len, X_train_data)))
# median = data['review_lenght'].median()
# mean = data['review_lenght'].mean()
# mode = data['review_lenght'].mode()[0]
#
# fig, ax = plt.subplots()
# sns.distplot(data['review_lenght'], bins=data['review_lenght'].max(),
#              hist_kws={"alpha": 0.9, "color": "blue"}, ax=ax,
#              kde_kws={"color": "black", 'linewidth': 3})
# ax.set_xlim(left=0, right=np.percentile(data['review_lenght'], 95))
# ax.set_xlabel('Words in review')
# ymax = 0.014
# plt.ylim(0, ymax)
# ax.plot([mode, mode], [0, ymax], '--', label=f'mode = {mode:.2f}', linewidth=4)
# ax.plot([mean, mean], [0, ymax], '--', label=f'mean = {mean:.2f}', linewidth=4)
# ax.plot([median, median], [0, ymax], '--',
#         label=f'median = {median:.2f}', linewidth=4)
# ax.set_title('Words per review distribution', fontsize=20)
# plt.legend()
# plt.show()

X = data['review']
y = data['sentiment']
X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=0.25, random_state=42, stratify=y_main,
                                                  shuffle=True)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_val shape: {}".format(X_val.shape))
print("y_val shape: {}".format(y_val.shape))
print("X_test: {}".format(X_test.shape))
print("y_test: {}".format(y_test.shape))

vocab_size = 20000
tokenizer = Tokenizer(oov_token="<OOV>", num_words=vocab_size)

tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
print(word_index)
print('total word index', len(tokenizer.word_index))

X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)

# max_len = 838
max_len = 200
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_val = pad_sequences(X_val, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')


vector_size = 200
gensim_weight_matrix = np.zeros((vocab_size, vector_size))
print('gensim_weight_matrix: ', gensim_weight_matrix.shape)

for word, index in tokenizer.word_index.items():
    if index < vocab_size:
        if word in glove_gensim:
            gensim_weight_matrix[index] = glove_gensim[word]
        else:
            gensim_weight_matrix[index] = np.zeros(vector_size)

    # Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
    # Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
    # Bidirectional(LSTM(32, return_sequences=True, dropout=0.2)),
    # Bidirectional(LSTM(8)),


model = Sequential([
    tf.keras.layers.Embedding(vocab_size, vector_size, name="embedding", weights=[gensim_weight_matrix], trainable=False),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
    GlobalMaxPool1D(),
    Dense(20, activation="relu"),
    Dropout(0.05),
    Dense(1, activation="sigmoid")
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=7, verbose=1, validation_data=(X_val, y_val))
model.evaluate(X_test, y_test)
model.save('bidi_lstm_6/')


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

weights = model.get_layer('embedding').get_weights()[0]
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

out_v = io.open('vectors1.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata1.tsv', 'w', encoding='utf-8')

for word_num in range(1, vocab_size - 1):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

# Getting Predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Accuracy Score
clf_accuracy_score = accuracy_score(y_test, y_pred)
print("Accuracy Score:\n", clf_accuracy_score, "\n")

# Classification Report
class_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", class_rep, "\n")

# Confusion Matrix
conf_mtx = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_mtx, "\n")