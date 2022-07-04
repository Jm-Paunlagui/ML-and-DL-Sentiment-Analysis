import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import regex
import io
import os
import nltk
from nltk import WordNetLemmatizer

from wordcloud import WordCloud
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM, Dropout, GlobalMaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.word2vec import Word2Vec

data = pd.read_csv("dataset/imdb.csv")
print(data.head(10))
print(data.shape)
# stopword_list = set(stopwords.words("english"))
# lemmatizer = WordNetLemmatizer()


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
# def remove_stopwords(text):
#     tokens = [token.strip() for token in word_tokenize(text)]
#     filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
#     filtered_text = ' '.join(filtered_tokens)
#
#     return filtered_text


# Applying the function to remove stopwords
# data['review'] = data['review'].apply(remove_stopwords)
print('removed stop words', data.head())

# Seperating Features and Target Variable
X = data['review']
y = data['sentiment']

X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=0.25, random_state=42, stratify=y_main,
                                                  shuffle=True)

# word2vec
# corpus_text = '\n'.join(data['review'])
# sentences = corpus_text.split('\n')
# sentences = [line.lower().split(' ') for line in sentences]
#
#
# def clean(s):
#     return [w.strip(',."!?:;()<>\'') for w in s]
#
#
# sentences = [clean(s) for s in sentences if len(s) > 0]
#
# # WORD2VEC
# W2V_SIZE = 300
# W2V_WINDOW = 7
# W2V_EPOCH = 32
# W2V_MIN_COUNT = 10
#
# w2v_model = Word2Vec(sentences, vector_size=256, window=7, min_count=3, workers=7, epochs=32)
# w2v_model.build_vocab(sentences)
# w2v_model.train(sentences, total_examples=X_train.shape[0], epochs=W2V_EPOCH)
# w2v_model.save('w2vModel.model')

w2v_model = Word2Vec.load("w2vModel.model")
print("word2vec", len(w2v_model.wv))
# Print the sets data shapes

print("X_main shape: {}".format(X_train.shape))
print("y_main shape: {}".format(y_train.shape))

print("X_test: {}".format(X_test.shape))
print("y_test: {}".format(y_test.shape))

max_length = 200
vocab_size = 20000
tokenizer = Tokenizer(oov_token="<unk>", num_words=vocab_size)

tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
print(len(tokenizer.word_index))

train_seqs = tokenizer.texts_to_sequences(X_train)
test_seqs = tokenizer.texts_to_sequences(X_test)
val_seq = tokenizer.texts_to_sequences(X_val)

train_seqs = pad_sequences(train_seqs, padding='post', maxlen=max_length)
test_seqs = pad_sequences(test_seqs, padding='post', maxlen=max_length)
val_seq = pad_sequences(val_seq, padding='post', maxlen=max_length)

print(train_seqs)

plt.hist([len(x) for x in X_train], bins=700)
plt.show()

embedding_matrix = np.zeros((len(tokenizer.word_index), 256))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
print(embedding_matrix.shape)

# Embedding(vocab_size, 256, name="embedding", weights=[embedding_matrix], trainable=False),
# Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
# Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
# Bidirectional(LSTM(32, return_sequences=True, dropout=0.2)),
# Bidirectional(LSTM(8)),
# Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
# GlobalMaxPool1D(),
# Dense(20, activation="relu"),
# Dropout(0.05),

model = Sequential([
    Embedding(len(tokenizer.word_index), 256, name="embedding", weights=[embedding_matrix], trainable=False),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
    GlobalMaxPool1D(),
    Dense(20, activation="relu"),
    Dropout(0.05),
    Dense(1, activation="sigmoid")
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(train_seqs, y_train, epochs=5, verbose=1, validation_data=(val_seq, y_val))
model.evaluate(test_seqs, y_test)
model.save('99Acc3.h5')

predict_p = model.predict(test_seqs)
predict_p = predict_p.flatten()
print(predict_p.round(2))

pred = np.where(predict_p > 0.5, 1, 0)
print(pred)

classi = classification_report(y_test, pred)
confu = confusion_matrix(y_test, pred)
accu = accuracy_score(y_test, pred)

# Display the outcome of classification
print('Classification Report: \n', classi)
print('Confusion Matrix: \n', confu)
print('Accuracy Score: \n', accu)

weights = model.get_layer('embedding').get_weights()[0]
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for word_num in range(1, vocab_size - 1):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]

    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")

out_v.close()
out_m.close()


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

#
print('\nEpoch No.  Train Accuracy  Train Loss     Val Accuracy    Val Loss')
for i in range(12):
    print('{:8d} {:10f} \t {:10f} \t {:10f} \t {:10f}'.format(i + 1, history.history['accuracy'][i],
                                                              history.history['loss'][i],
                                                              history.history['val_accuracy'][i],
                                                              history.history['val_loss'][i]))
#
