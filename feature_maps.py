import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import librosa.display

def load_cochleagrams(dataset_dir, num_classes):
    X = []
    y = []
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
    
    return np.array(X), keras.utils.to_categorical(y, num_classes=num_classes)

def plot_feature_maps(model, input_sample, layer_names):
    """Visualize feature maps from specific layers of the model."""
    intermediate_models = {}
    for name in layer_names:
        try:
            # Get the layer
            layer = model.get_layer(name)
            
            # Create an intermediate model
            intermediate_model = keras.Model(
                inputs=model.input, 
                outputs=layer.output
            )
            intermediate_models[name] = intermediate_model
        except ValueError as e:
            print(f"Error processing layer {name}: {e}")

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

# Configuration
input_shape = (128, 32, 1)
num_classes = 5
model_save_path = "C:/Users/hridai/Desktop/DONN/auditory_CNN_traditional.keras"
test_dataset_dir = "C:/Users/hridai/Desktop/DONN/datasets/cochleagrams"

# Load the original saved model directly
model = keras.models.load_model(model_save_path)

# Load test data
X_test, y_test = load_cochleagrams(test_dataset_dir, num_classes)

# Prepare input sample
sample_index = 0
input_sample = X_test[sample_index]
input_sample = input_sample.reshape((1,) + input_shape)
true_label = np.argmax(y_test[sample_index])

# Define the layers to probe for feature and saliency maps
layer_names = [layer.name for layer in model.layers if isinstance(layer, (keras.layers.Conv2D, keras.layers.BatchNormalization))]

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