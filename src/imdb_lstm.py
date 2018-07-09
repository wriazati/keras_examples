'''Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 10000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 64


def encode_example(s):
	INDEX_FROM = 3  # word index offset

	word_to_id = imdb.get_word_index()
	word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
	word_to_id["<PAD>"] = 0
	word_to_id["<START>"] = 1
	word_to_id["<UNK>"] = 2
	enc = [word_to_id[w] for w in s.lower().split() if w in word_to_id]
	return [[enc]]

def decode_example(x):
	INDEX_FROM = 3  # word index offset

	word_to_id = imdb.get_word_index()
	word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
	word_to_id["<PAD>"] = 0
	word_to_id["<START>"] = 1
	word_to_id["<UNK>"] = 2

	id_to_word = {value: key for key, value in word_to_id.items()}
	print(' '.join(id_to_word[id] for id in x))


s = "This is the worst movie i have ever seen"
encode_example(s)


print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


# Ensure all examples are same length (maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()

# https://stackoverflow.com/questions/38189713/what-is-an-embedding-in-keras
# embedding size:  (vocab_size, embedding_dim)
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))

model.add(Dense(64, activation='relu'))

model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1))

model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

print(model.predict(encode_example(s)))



'''
Test score: 0.38382962569236756
Test accuracy: 0.8300399999809265
[[0.11731416]]


Test score: 0.38401358434677124
Test accuracy: 0.8331999999809265
[[0.07479599]]
'''