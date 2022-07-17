import re

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.saving.save import load_model
from keras.utils import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

tokenizer = Tokenizer(oov_token="<OOV>")
# Use the model to predict a review
fake_reviews = ['I love this phone', 'I hate spaghetti',
                'Everything was cold',
                'Everything was hot exactly as I wanted',
                'Everything was green',
                'the host seated us immediately',
                'they gave us free chocolate cake',
                'not sure about the wilted flowers on the table',
                'only works when I stand on tippy toes',
                'does not work when I stand on my head',
                'they gave us free chocolate cake and did not charge us']

model = load_model('99Acc3.h5')

print(fake_reviews)
tokenizer.fit_on_texts(fake_reviews)
max_length = 50
padding_type = 'post'
# Create the sequences
padding_type = 'post'
sample_sequences = tokenizer.texts_to_sequences(fake_reviews)
fakes_padded = pad_sequences(sample_sequences, padding=padding_type, maxlen=max_length)

print('\nHOT OFF THE PRESS! HERE ARE SOME NEWLY MINTED, ABSOLUTELY GENUINE REVIEWS!\n')

classes = model.predict(fakes_padded)

# The closer the class is to 1, the more positive the review is deemed to be
for x in range(len(fake_reviews)):
    print(fake_reviews[x])
    print(classes[x])
    print('\n')

# Try adding reviews of your own
# Add some negative words (such as "not") to the good reviews and see what happens
# For example:
# they gave us free chocolate cake and did not charge us
