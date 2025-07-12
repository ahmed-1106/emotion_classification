import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
import os

from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms 


class TrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['angry', 'happy', 'neutral']
        self.image_paths = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append((img_path, class_idx))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        
        # Open image and convert to RGB (even if grayscale)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label



class TestDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['angry', 'happy', 'neutral']
        self.image_paths = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for image_name in os.listdir(class_dir):
                if image_name.endswith('.jpg'):
                    image_path = os.path.join(class_dir, image_name)
                    self.image_paths.append((image_path, class_idx))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, class_idx = self.image_paths[idx]
        image = Image.open(image_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        return image, class_idx

data_transforms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),  # This converts PIL Image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet stats
                         std=[0.229, 0.224, 0.225]),
    transforms.Grayscale(num_output_channels=1)  # Add this if you want grayscale
])
##model design
class EmotionCNN(nn.Module):
     def __init__(self):
        super(EmotionCNN, self).__init__()  
        
        # First convo
        self.conv1 = nn.Conv2d(
            in_channels=1,  
            out_channels=32,  
            kernel_size=3,  
            padding=1  
        )
           
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(
            in_features=64*12*12,
            out_features=128
        )
        self.fc2 = nn.Linear(128, 3)

     def forward(self, x):
         x = self.pool(nn.functional.relu(self.conv1(x)))
         
         x = self.pool(nn.functional.relu(self.conv2(x)))

         x = x.view(-1, 64 * 12 * 12)

         x = self.dropout(x)

         x = nn.functional.relu(self.fc1(x))

         x = self.fc2(x)   

         return x 

##############################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionCNN().to(device)

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(train_loader, epochs=15):
    """Trains the model for specified number of epochs"""
    model.train()  # Set model to training mode
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for images, labels in train_loader:
            # Move data to GPU if available
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')





    ####################################################################
    
def real_time_detection(model, transform):
    """Performs real-time emotion detection using webcam"""
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Define emotion labels
    emotion_labels = ['Angry', 'Happy', 'Neutral']
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Convert to PIL Image and apply transforms
            face_img = Image.fromarray(face_roi)
            face_tensor = transform(face_img).unsqueeze(0).to(device)
            
            # Predict emotion
            with torch.no_grad():  # Disable gradient calculation
                outputs = model(face_tensor)
                _, predicted = torch.max(outputs, 1)
                emotion = emotion_labels[predicted.item()]
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame, 
                emotion, 
                (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 255, 0), 
                2
            )
        
        # Display result
        cv2.imshow('Real-Time Emotion Detection', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    # Initialize datasets
    train_dataset = TrainDataset(
        root_dir='C:/Users/Ahmed Sherif/Desktop/feelings/train',
        transform=data_transforms
    )
    
    test_dataset = TestDataset(
        root_dir='C:/Users/Ahmed Sherif/Desktop/feelings/test',
        transform=data_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32
    )
    
    # Train the model
    train_model(train_loader, epochs=15)
    
    # Save trained model
    torch.save(model.state_dict(), 'emotion_model.pth')
    
    # Run real-time detection
    real_time_detection(model, data_transforms)