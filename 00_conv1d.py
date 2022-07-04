import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import re
import regex
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM, GlobalMaxPool1D, Dropout, SpatialDropout1D, Conv1D, MaxPooling1D
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

# data = pd.read_csv('sentiment_tweets3.csv', engine='python')
data = pd.read_csv('dataset/imdb.csv', engine='python')

print(data.head(10))

data = data.sample(frac=1., random_state=14).reset_index(drop=True)

print(data['sentiment'].value_counts(0))

EPOCH = 5

stopword_list = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


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

max_length = 200
vocab_size = 20000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
print(len(tokenizer.word_index))

train_seqs = tokenizer.texts_to_sequences(X_train)
val_seqs = tokenizer.texts_to_sequences(X_val)
test_seqs = tokenizer.texts_to_sequences(X_test)

train_seqs = pad_sequences(train_seqs, padding='post', maxlen=max_length, truncating='post')
val_seqs = pad_sequences(val_seqs, padding='post', maxlen=max_length, truncating='post')
test_seqs = pad_sequences(test_seqs, padding='post', maxlen=max_length, truncating='post')

# AUTOTUNE = tf.data.AUTOTUNE
# train_seqs = train_seqs.cache().prefetch(buffer_size=AUTOTUNE)
# val_seqs = val_seqs.cache().prefetch(buffer_size=AUTOTUNE)
# test_seqs = test_seqs.cache().prefetch(buffer_size=AUTOTUNE)

model = Sequential([
    Embedding(vocab_size, 128, name="embedding"),
    Conv1D(64, 8, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(16, activation='relu'),
    GlobalMaxPool1D(),
    Dense(1, activation="sigmoid")
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(train_seqs, y_train, epochs=EPOCH, validation_data=(val_seqs, y_val), verbose=1)

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

model.save('99Acc2-qwerty.h5')


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

print('\nEpoch No.  Train Accuracy  Train Loss     Val Accuracy    Val Loss')
for i in range(EPOCH):
    print('{:8d} {:10f} \t {:10f} \t {:10f} \t {:10f}'.format(i + 1, history.history['accuracy'][i],
                                                              history.history['loss'][i],
                                                              history.history['val_accuracy'][i],
                                                              history.history['val_loss'][i]))

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
