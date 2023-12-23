import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import load_img, img_to_array

# Define command-line arguments
parser = argparse.ArgumentParser(description='Predict hand sign from an image')
parser.add_argument('image_path', type=str, help='Path to the hand sign image file')

# Parse command-line arguments
args = parser.parse_args()
image_path = args.image_path

model_path = 'RPS_Resnet50Model.hdf5'
# Load the saved trained model
model = load_model(model_path)

class_names_path = 'class_names.json'
# Load the class names from the JSON file
with open(class_names_path, 'r') as f:
    class_names = json.load(f)
class_labels = list(class_names.keys())

# Load and preprocess the hand sign image
image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image /= 255.0

# Perform prediction
prediction = model.predict(image)
predicted_class_idx = np.argmax(prediction)
predicted_class = class_labels[predicted_class_idx]

print('Predicted hand sign: {}'.format(predicted_class))

# Visualisation of the supplied image with the prediction score and predicted label
predicted_label = predicted_class
prediction_score = np.max(prediction)

# Display the image with prediction info
plt.imshow(plt.imread(args.image_path))
plt.axis('off')
plt.title(f'Predicted Label: {str(predicted_label).capitalize()}\nPrediction Score: {prediction_score:.2f}')
plt.show()
