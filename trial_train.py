import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam
from PIL import Image
import os
from model import UNet
from depth_loss import *

# Define UNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
model = UNet(in_channels=3).to(device)

# Define optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Define dataset class
class DepthDataset(Dataset):
    def __init__(self, left_folder, right_folder, transform=None):
        self.left_folder = left_folder
        self.right_folder = right_folder
        self.transform = transform

        # Assuming images have the same name in both folders
        self.left_images = os.listdir(left_folder)
        self.right_images = os.listdir(right_folder)

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        left_path = os.path.join(self.left_folder, self.left_images[idx])
        right_path = os.path.join(self.right_folder, self.right_images[idx])

        left_image = Image.open(left_path).convert("RGB")
        right_image = Image.open(right_path).convert("RGB")

        if self.transform:
            left_image = transforms.Resize((640,480))(left_image)
            left_image = self.transform(left_image)
            right_image = transforms.Resize((292, 452))(right_image)
            right_image = self.transform(right_image)

        return left_image, right_image

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Specify your data folders
left_folder = ""
right_folder = ""

# Create dataset and dataloader
dataset = DepthDataset(left_folder, right_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

if __name__ == '__main__':
    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (left_data, right_data) in enumerate(dataloader):
            left_data, right_data = left_data.to(device), right_data.to(device)

            # Forward pass
            intermediate_disparity_1, intermediate_disparity_2, intermediate_disparity_3, final_disparity = model(left_data)

            # Compute loss
            loss = depth_loss(left_data, right_data, intermediate_disparity_1, intermediate_disparity_2, intermediate_disparity_3, final_disparity)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), "C:\\ML Projects\\ConsistentDepth\\Models\\test_model.pth")