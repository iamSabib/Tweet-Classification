# Import required libraries
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Load data from the current directory
df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Use only 'text' and 'target' columns for training data
df = df[['text', 'target']]

# Split into train, validation, and test sets (80-10-10)
train_val, test = train_test_split(df, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.1, random_state=42)  

# Tokenize text
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train['text'])

X_train = tokenizer.texts_to_sequences(train['text'])
X_val = tokenizer.texts_to_sequences(val['text'])
X_test = tokenizer.texts_to_sequences(test['text'])

X_train = pad_sequences(X_train, maxlen=max_len)
X_val = pad_sequences(X_val, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

y_train = train['target'].values
y_val = val['target'].values
y_test = test['target'].values

# Define an Adam optimizer with a learning rate of 0.0001
optimizer = Adam(learning_rate=0.0001)

# Build model with more regularization, dropout, and batch normalization
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    BatchNormalization(),  # Normalize the embeddings
    Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l2(0.02))),
    BatchNormalization(),  # Normalize GRU output
    Dropout(0.4),  # Increased dropout rate
    Bidirectional(GRU(32, kernel_regularizer=l2(0.01))),
    BatchNormalization(),  # Normalize GRU output
    Dropout(0.4),  # Increased dropout rate
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks: Early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1)

# Train model with printing after every epoch
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, lr_scheduler],
    verbose=2  # Verbose 2 prints progress after every epoch
)

# Print evaluation on validation set after training
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=2)
print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Evaluate on test set from train.csv
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Print test set evaluation metrics
print("\nResults on test set from train.csv:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_binary):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_binary):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_binary):.4f}")


# Print detailed training and validation metrics after each epoch
print("\nEpoch-wise training details:")
for epoch in range(len(history.history['accuracy'])):
    print(f"Epoch {epoch+1}:")
    print(f"Train Accuracy: {history.history['accuracy'][epoch]:.4f}, Train Loss: {history.history['loss'][epoch]:.4f}")
    print(f"Validation Accuracy: {history.history['val_accuracy'][epoch]:.4f}, Validation Loss: {history.history['val_loss'][epoch]:.4f}")
    print("-" * 50)

# Plot training history metrics
import matplotlib.pyplot as plt

# Accuracy plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
