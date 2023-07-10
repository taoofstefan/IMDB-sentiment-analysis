import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

VOCAB_SIZE = 5000
INDEX_FROM = 3
MAX_SEQ_LEN = 128
EMBEDDING_DIM = 64

# Load the IMDB dataset and split it into training and testing sets
# First run will take a while until the dataset is loaded
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE, index_from=INDEX_FROM)

# Map word indices to their corresponding words
word_to_idx = imdb.get_word_index()
idx_to_word = {v+INDEX_FROM: k for k, v in word_to_idx.items()}
idx_to_word[0] = '<PAD>'
idx_to_word[1] = '<START>'
idx_to_word[2] = '<UNK>'

# Pad sequences to have a consistent length
X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQ_LEN)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQ_LEN)

# Define the model architecture
# Model 1
model1 = Sequential()
model1.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQ_LEN))
model1.add(LSTM(128))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2)

# Model 2
model2 = Sequential()
model2.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQ_LEN))
model2.add(LSTM(64, return_sequences=True))
model2.add(LSTM(64))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

# Model 3
model3 = Sequential()
model3.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQ_LEN))
model3.add(LSTM(256))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.fit(X_train, y_train, epochs=20, batch_size=256, validation_split=0.2)

# Create a TensorBoard callback to visualize training logs
tensorboard = keras.callbacks.TensorBoard(log_dir='sentiment_logs')

# Get the predicted probabilities for each review
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)


final_pred = (pred1 + pred2 + pred3) / 3
final_pred = np.round(final_pred).flatten().astype(int)

ensemble_accuracy = np.mean(final_pred == y_test)
print("Ensemble Accuracy: {:.4f}".format(ensemble_accuracy))



