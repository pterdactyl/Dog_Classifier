import os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Load label mapping from the original training script
df = pd.read_csv('dogs.csv')
# Ensure label_mapping aligns with training indices
label_mapping = {idx: label for idx, label in enumerate(df['labels'].unique())}

print(label_mapping)

# Load the best model
model_path = 'best_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture
from torchvision import models
model = models.efficientnet_b0(pretrained=True)  # Ensure pretrained=True
num_features = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(num_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(256, len(label_mapping)),
    torch.nn.Softmax(dim=1)
)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Define data transformation (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    """Predict the label for a single image."""
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Map prediction to label
    label_idx = predicted.item()
    try:
        label = label_mapping[label_idx]
    except KeyError:
        label = f"Unknown class {label_idx}"
        print(f"Warning: Predicted class index {label_idx} not found in label_mapping.")

    return label

def visualize_prediction(image_path):
    """Visualize the image with the predicted label overlayed."""
    # Predict the label
    label = predict_image(image_path)

    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Create a figure
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f"Predicted Label: {label}", fontsize=16, color="blue")
    plt.axis('off')
    plt.show()

def predict_from_folder(folder_path):
    """Predict labels for all images in a folder and visualize them."""
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                visualize_prediction(file_path)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Example usage
if __name__ == "__main__":
    folder_path = 'input'  # Replace with your folder path
    predict_from_folder(folder_path)