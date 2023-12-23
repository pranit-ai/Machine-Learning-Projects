import argparse
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Argument parser
parser = argparse.ArgumentParser(description='Predict hand signs')
parser.add_argument('image1_path', type=str, help='Path to the first hand sign image file')
parser.add_argument('image2_path', type=str, help='Path to the second hand sign image file')
args = parser.parse_args()

model_path = 'RPS_Resnet50Model.hdf5'

# Load the trained model
model = load_model(model_path)

class_names_path = 'class_names.json'

# Load the class names
with open(class_names_path, 'r') as f:
    class_names = json.load(f)
class_labels = list(class_names.keys())

# Load and preprocess the images
image1 = Image.open(args.image1_path).convert('RGB')  # Convert RGBA to RGB
image1 = image1.resize((224, 224))  # Resize to match input size of the model
image1 = np.array(image1)
image1 = image1 / 255.0  # Normalize pixel values to [0, 1]
image1 = np.expand_dims(image1, axis=0)  # Add batch dimension

image2 = Image.open(args.image2_path).convert('RGB')  # Convert RGBA to RGB
image2 = image2.resize((224, 224))  # Resize to match input size of the model
image2 = np.array(image2)
image2 = image2 / 255.0  # Normalize pixel values to [0, 1]
image2 = np.expand_dims(image2, axis=0)  # Add batch dimension

# Predict labels for the images
prediction1 = model.predict(image1)
predicted1_class_idx = np.argmax(prediction1)
label1 = class_labels[predicted1_class_idx]

prediction2 = model.predict(image2)
predicted2_class_idx = np.argmax(prediction2)
label2 = class_labels[predicted2_class_idx]

predicted_label1 = str(label1).capitalize()
predicted_label2 = str(label2).capitalize()

prediction_score1 = np.max(prediction1)
prediction_score2 = np.max(prediction2)

# Output the predicted labels
print("Prediction for image 1: {}".format(predicted_label1))
print("Prediction for image 2: {}".format(predicted_label2))

# Determine the winner of the rock, paper, scissor game
winner = None
if predicted_label1 == 'Rock':
    if predicted_label2 == 'Scissors':
        winner = 'Image 1: ' + predicted_label1
    elif predicted_label2 == 'Paper':
        winner = 'Image 2: ' + predicted_label2
elif predicted_label1 == 'Scissors':
    if predicted_label2 == 'Paper':
        winner = 'Image 1: ' + predicted_label1
    elif predicted_label2 == 'Rock':
        winner = 'Image 2: ' + predicted_label2
elif predicted_label1 == 'Paper':
    if predicted_label2 == 'Rock':
        winner = 'Image 1: ' + predicted_label2
    elif predicted_label2 == 'Scissors':
        winner = 'Image 2: ' + predicted_label2


# Plot the images side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(np.array(Image.open(args.image1_path)))
ax[0].set_title('Image 1\nPredicted Label: {}\nPrediction Score: {:.2f}'.format(predicted_label1, prediction_score1))
ax[0].axis('off')
ax[1].imshow(np.array(Image.open(args.image2_path)))
ax[1].set_title(
    'Image 2\nPredicted Label: {}\nPrediction Score: {:.2f}'.format(predicted_label2, prediction_score2))
ax[1].axis('off')

if winner:
    print("The winner is: {}".format(winner))
    plt.suptitle('Winner: \n{}'.format(winner), fontweight='bold')
    plt.show()  # Show the plot
else:
    print("The game is a tie")
