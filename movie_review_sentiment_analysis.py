import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
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

# Print the first review in the training set as words instead of indices
# print(" ".join([idx_to_word[idx] for idx in X_train[0]]))

# Define the model architecture
model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQ_LEN))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with loss function, optimizer, and metrics
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print a summary of the model architecture
print(model.summary())

# Create a TensorBoard callback to visualize training logs
tensorboard = keras.callbacks.TensorBoard(log_dir='sentiment_logs')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=256, validation_split=0.2, callbacks=[tensorboard])

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test)
print('Accuracy: {0:.4f}'.format(score[1]))

# Get the predicted probabilities for each review
y_pred = model.predict(X_test)

# Threshold for classifying as negative sentiment
threshold = 0.5

# Iterate through the predicted probabilities and print negative reviews
for i in range(10):
    if y_pred[i] < threshold:
        review = " ".join([idx_to_word[idx] for idx in X_test[i] if idx != 0])
        print("Review: ", review)
        print("Predicted Sentiment: Negative")
        print("----------------------------------------")

