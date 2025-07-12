import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Define the same model architecture as during training
class EmotionCNN(torch.nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.25)
        self.fc1 = torch.nn.Linear(64 * 12 * 12, 128)
        self.fc2 = torch.nn.Linear(128, 3)  # 3 classes: angry, happy, neutral

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 64 * 12 * 12)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
def load_model(model_path):
    model = EmotionCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Real-time detection
def run_detection(model):
    emotion_labels = ['Angry', 'Happy', 'Neutral']
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_img = Image.fromarray(face_roi)
            face_tensor = transform(face_img).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(face_tensor)
                _, predicted = torch.max(outputs, 1)
                emotion = emotion_labels[predicted.item()]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Load your trained model
    model = load_model('C:/Users/Ahmed Sherif/Desktop/feelings/emotion_model.pth')  
    
    # Run real-
    run_detection(model)