import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Load the IMDB dataset
vocab_size = 10000  # Number of words to consider as features
maxlen = 200  # Cut texts after this number of words

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to ensure uniform input size
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Define the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
batch_size = 64
epochs = 5

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')
# Save the model
model.save('sentiment_analysis_model.h5')

# Load the model
# from tensorflow.keras.models import load_model
# model = load_model('sentiment_analysis_model.h5')
# Function to decode the reviews
word_index = imdb.get_word_index()
index_word = {v: k for k, v in word_index.items()}

def decode_review(review):
    return ' '.join([index_word.get(i - 3, '?') for i in review])

# Make predictions on new data
sample_review = x_test[0]
predicted_sentiment = model.predict(tf.expand_dims(sample_review, axis=0))
print(f'Review: {decode_review(sample_review)}')
print(f'Predicted Sentiment: {"Positive" if predicted_sentiment > 0.5 else "Negative"}')
