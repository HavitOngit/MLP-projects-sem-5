import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# Reshape images to (28, 28, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Define the CNN model
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, batch_size=128, epochs=15, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Function to plot misclassified images
def plot_misclassified(X_test, y_test, y_pred_classes, num_images=10):
    misclassified = np.where(y_test != y_pred_classes)[0]
    num_plot = min(num_images, len(misclassified))
    
    plt.figure(figsize=(15, 2 * num_plot))
    for i, idx in enumerate(misclassified[:num_plot]):
        plt.subplot(num_plot, 5, i*5 + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_test[idx]}, Pred: {y_pred_classes[idx]}")
        plt.axis('off')
        
        for j in range(4):
            plt.subplot(num_plot, 5, i*5 + j + 2)
            plt.bar(range(10), y_pred[idx])
            plt.title(f"Top {j+1}: {np.argsort(y_pred[idx])[-j-1]}")
            plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

# Plot misclassified images
plot_misclassified(X_test, y_test, y_pred_classes)

# Function to predict new samples
def predict_digit(model, image):
    # Ensure the image is in the correct format (28x28 grayscale)
    image = image.reshape(1, 28, 28, 1).astype('float32') / 255
    
    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return predicted_class, confidence

# Example usage of prediction function
# Let's use a sample from the test set
sample_idx = np.random.randint(0, len(X_test))
sample_image = X_test[sample_idx]
true_label = y_test[sample_idx]

predicted_digit, confidence = predict_digit(model, sample_image)

plt.imshow(sample_image.reshape(28, 28), cmap='gray')
plt.title(f"True: {true_label}, Predicted: {predicted_digit}\nConfidence: {confidence:.4f}")
plt.axis('off')
plt.show()