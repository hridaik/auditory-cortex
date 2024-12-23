import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers, models, utils, saving
import librosa
import matplotlib.pyplot as plt
import random
import shutil

@saving.register_keras_serializable()
class AuditoryCortexNet(models.Model):
    def __init__(self, input_shape, num_classes, **kwargs):
        """
        Initialize auditory cortex-inspired neural network
        
        Args:
            input_shape (tuple): Input cochleagram shape (time, frequency, channels)
            num_classes (int): Number of output classes
        """
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Primary Auditory Cortex (A1) Layers
        self.primary_layers = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, 
                          padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2))
        ])
        
        # Belt Regions Layers
        self.belt_layers = models.Sequential([
            layers.Conv2D(128, (5, 5), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (5, 5), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D()
        ])
        
        # Parabelt Regions (Dense Layers)
        self.parabelt_layers = models.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    def call(self, inputs):
        """
        Forward pass through the network
        
        Args:
            inputs (tensor): Input cochleagram
        
        Returns:
            tensor: Class probabilities
        """
        x = self.primary_layers(inputs)
        x = self.belt_layers(x)
        x = self.parabelt_layers(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

def create_model(input_shape, num_classes):
    """
    Convenience function to create and compile the model
    
    Args:
        input_shape (tuple): Input cochleagram shape
        num_classes (int): Number of output classes
    
    Returns:
        compiled keras model
    """
    model = AuditoryCortexNet(input_shape, num_classes)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_cochleagrams(dataset_dir, num_classes):
    """
    Load cochleagrams and their corresponding labels for training
    
    Args:
        dataset_dir (str): Path to cochleagram directory
        num_classes (int): Number of classes (speakers)
    
    Returns:
        tuple: (X, y) where X is the input data and y is the one-hot encoded labels
    """
    X = []
    y = []
    # Exclude background noise and other directories
    valid_labels = [label for label in os.listdir(dataset_dir) 
                    if label not in ['_background_noise_', 'other']]
    
    label_map = {label: idx for idx, label in enumerate(sorted(valid_labels))}
    
    for label in valid_labels:
        label_dir = os.path.join(dataset_dir, label)
        for file in os.listdir(label_dir):
            if file.endswith('.npy'):
                cochleagram = np.load(os.path.join(label_dir, file))
                
                if len(cochleagram.shape) == 2:
                    cochleagram = np.expand_dims(cochleagram, axis=-1)
                
                X.append(cochleagram)
                y.append(label_map[label])
    
    return np.array(X), utils.to_categorical(y, num_classes=num_classes)

def prepare_data(dataset_dir, num_classes, test_size=0.2, val_size=0.2):
    """
    Split data into training, validation, and test sets
    
    Args:
        dataset_dir (str): Path to cochleagram directory
        num_classes (int): Number of classes (speakers)
        test_size (float): Proportion of data for the test set
        val_size (float): Proportion of training data for the validation set
    
    Returns:
        tuple: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    X, y = load_cochleagrams(dataset_dir, num_classes)
    
    # Split into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    
    # Split training+validation into training and validation sets
    val_split = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_split, stratify=y_train_val)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

dataset_dir = "C:/Users/hridai/Desktop/DONN/datasets/cochleagrams"
num_classes = 5
(X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(dataset_dir, num_classes)

# Input shape for the model
input_shape = X_train.shape[1:]  # (time_steps, frequency_bins, channels)

# Create the model
model = create_model(input_shape, num_classes)

# Summary of the model
model.summary()

# Train the model
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the model to a directory
model_save_path = "C:/Users/hridai/Desktop/DONN/auditory_CNN_traditional.keras"
model.save(model_save_path)

print(f"Model saved at {model_save_path}")

def plot_feature_maps(model, input_sample, layer_names):
    intermediate_models = {name: models.Model(inputs=model.input, outputs=model.get_layer(name).output)
                           for name in layer_names}

    for name, intermediate_model in intermediate_models.items():
        feature_map = intermediate_model.predict(input_sample)

        # Plot feature maps
        num_filters = feature_map.shape[-1]
        size = feature_map.shape[1:3]  # Spatial dimensions

        plt.figure(figsize=(15, 15))
        for i in range(min(num_filters, 16)):  # Show up to 16 filters for clarity
            plt.subplot(4, 4, i + 1)
            plt.imshow(feature_map[0, :, :, i], cmap='viridis')
            plt.axis('off')
        plt.suptitle(f"Feature Maps from Layer: {name} (Shape: {size})", fontsize=16)
        plt.show()

def saliency_map(model, input_sample, class_index):
    """Compute the saliency map for a specific class prediction."""
    input_sample = tf.convert_to_tensor(input_sample, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_sample)
        predictions = model(input_sample)
        loss = predictions[0, class_index]
    grads = tape.gradient(loss, input_sample)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()[0]
    return saliency

# Prepare input sample
sample_index = 0
input_sample = X_test[sample_index]
input_sample = input_sample.reshape((1,) + input_shape)
true_label = np.argmax(y_test[sample_index])

# Define the layers to probe for feature and saliency maps
layer_names = [layer.name for layer in model.layers if isinstance(layer, (layers.Conv2D, layers.BatchNormalization))]

# Visualize Feature Maps
plot_feature_maps(model, input_sample, layer_names)

# Visualize Saliency Map
predicted_class = np.argmax(model.predict(input_sample))
saliency = saliency_map(model, input_sample, predicted_class)

plt.figure(figsize=(10, 5))
librosa.display.specshow(saliency, x_axis='time', y_axis='mel', cmap='hot', sr=16000)
plt.colorbar(format='%+2.0f')
plt.title(f"Saliency Map for Predicted Class {predicted_class} (True Class {true_label})")
plt.tight_layout()
plt.show()