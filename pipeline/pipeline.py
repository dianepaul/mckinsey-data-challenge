import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import os
import random
import pandas as pd
from sklearn.metrics import roc_auc_score

# Function to augment and save images
def augment_and_save_images(original_directory, output_directory):
    """
    Augments images from the original_directory and saves the augmented images to the output_directory.

    Args:
        original_directory (str): Path to the directory containing the original images.
        output_directory (str): Path to the directory where augmented images will be saved.

    """
    os.makedirs(output_directory, exist_ok=True)  # Create the output directory if it doesn't exist

    # List of transformation functions
    transformations = [
        lambda img: img,
        lambda img: img.rotate(90),
        lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)
    ]

    for filename in os.listdir(original_directory):
        if filename.endswith((".jpg", ".png", ".tif")):
            original_image_path = os.path.join(original_directory, filename)
            original_image = Image.open(original_image_path)

            # Apply transformations and save augmented images
            for i, transformation in enumerate(transformations):
                augmented_image = transformation(original_image)
                augmented_image.save(os.path.join(output_directory, f"augmented_{i}_{filename}"))

# Call the function to augment and save images for the "plume" class
original_directory_plume = "../cleanr/train data/images/plume"
output_directory_plume = "../data_augmented/plume"
augment_and_save_images(original_directory_plume, output_directory_plume)

# Call the function to augment and save images for the "no_plume" class
original_directory_no_plume = "../cleanr/train data/images/no_plume"
output_directory_no_plume = "../data_augmented/no_plume"
augment_and_save_images(original_directory_no_plume, output_directory_no_plume)


# Define a transform to preprocess the images (you can adjust the normalization values)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor()
])

# Path to your dataset folder
data_dir = '../data_augmented/'

# Create the ImageFolder dataset
def create_dataset(data_dir):
    """
    Creates an ImageFolder dataset from the specified directory.

    Args:
        data_dir (str): Path to the directory containing the dataset.

    Returns:
        dataset: An ImageFolder dataset object.
    """
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return dataset

# Define the CNN model
class PlumeCNN(nn.Module):
    def __init__(self):
        super(PlumeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.pool(self.bn1(nn.functional.relu(self.conv1(x))))
        x = self.pool(self.bn2(nn.functional.relu(self.conv2(x))))
        x = x.view(-1, 64 * 16 * 16)
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x

# Define a function to train the model
def train_model(data_dir, num_epochs=15, batch_size=32, learning_rate=0.001):
    """
    Trains a convolutional neural network model on the specified dataset.

    Args:
        data_dir (str): Path to the directory containing the dataset.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimization.

    Returns:
        model: The trained CNN model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = create_dataset(data_dir)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PlumeCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store training data predictions
    training_image_paths = []
    training_probabilities = []

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            labels = labels.view_as(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Collect training data predictions
            training_image_paths += [dataset.imgs[i][0] for i in range(len(images))]
            training_probabilities += outputs.cpu().tolist()

    # Save training data results to a CSV file
    training_df = pd.DataFrame({"path": training_image_paths, "proba": training_probabilities})
    training_df.to_csv("train_results.csv", index=False)

    return model

# Define a function to predict the ROC AUC of a new test image
def predict_new_image(model, image):
    """
    Predicts the probability of an image belonging to a class using a trained model.

    Args:
        model: Trained CNN model.
        image: Input image tensor.

    Returns:
        output (float): Probability of the image belonging to a class (0-1).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image).squeeze().item()
    return output

# Train the model
trained_model = train_model(data_dir)

# Test the model
test_data_dir = "../cleanr/test data/images"

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor()
    # Add any other necessary transformations and normalization steps here
])

def evaluate_test_images(trained_model, test_data_dir):
    """
    Evaluates the test images using a trained model and returns the image paths and probabilities of being a plume.

    Args:
        trained_model: Trained CNN model.
        test_data_dir (str): Path to the directory containing test images.

    Returns:
        image_paths (list): List of image file paths.
        probability (list): List of predicted probabilities.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)
    trained_model.eval()

    image_paths = []
    probability = []

    for filename in os.listdir(test_data_dir):
        if filename.endswith((".jpg", ".png", ".tif")):
            image_path = os.path.join(test_data_dir, filename)
            image = Image.open(image_path)
            image = test_transform(image)  # Apply the test transform
            image = image.unsqueeze(0).to(device)  # Convert to tensor and move to device
            label = predict_new_image(trained_model, image)
            image_paths.append(filename)
            probability.append(label)

    return image_paths, probability

image_paths, probability = evaluate_test_images(trained_model, test_data_dir)

df = pd.DataFrame({"path": image_paths, "proba": probability})

df.to_csv("test_results.csv", index=False)


