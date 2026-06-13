import tensorflow as tf
import numpy as np
import pandas as pd
import os
import kagglehub 
import matplotlib.pyplot as plt
path = kagglehub.dataset_download("datamunge/sign-language-mnist")
train_df = pd.read_csv(os.path.join(path, "sign_mnist_train.csv"))
test_df = pd.read_csv(os.path.join(path, "sign_mnist_test.csv"))
train_images = train_df.drop("label", axis=1).values
train_labels = train_df["label"].values
test_images = test_df.drop("label", axis=1).values
test_labels = test_df["label"].values
train_images = train_images.reshape(-1, 28, 28) / 255.0      #784 to 28x28 and normalize pixel values and the 255 for simplification
test_images = test_images.reshape(-1, 28, 28) / 255.0
print("Training set shape:", train_images.shape, train_labels.shape)
print("Testing set shape:", test_images.shape, test_labels.shape)
label_to_letter = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}
model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(25, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=7)
metrics = model.evaluate(test_images, test_labels)
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
plt.figure(figsize=(20, 20))
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(test_images[i], cmap='gray')
    plt.text(1,2, f"True: {label_to_letter[test_labels[i]]}", color='red', fontsize=12)
    plt.text(1,26, f"Predicted: {label_to_letter[predicted_labels[i]]}", color='blue', fontsize=12)
    plt.axis('off')
plt.tight_layout()
plt.show()
model.save("sign_language_model.keras")