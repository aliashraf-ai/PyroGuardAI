"""
CNN Satellite Image Analyzer
Detects fires from aerial/satellite imagery using transfer learning
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split

class FireImageDataset(Dataset):
    """Custom dataset for fire detection"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Return blank image on error
            print(f"Error loading {img_path}: {e}")
            blank = Image.new('RGB', (224, 224), color='black')
            return self.transform(blank) if self.transform else blank, self.labels[idx]

class FireDetectorCNN:
    """CNN for fire detection using MobileNetV2 transfer learning"""

    def __init__(self):
        # 1. Setup Model Architecture
        print("🔥 Initializing MobileNetV2 for fire detection...")
        try:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            self.model = models.mobilenet_v2(weights=weights)
        except:
            self.model = models.mobilenet_v2(pretrained=True)

        # Freeze early layers
        for param in self.model.features[:10].parameters():
            param.requires_grad = False

        # Custom Classifier (Binary: Fire vs No Fire)
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),  # Increased Dropout to prevent overfitting
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

        # 2. Stronger Data Augmentation (Fixes "Red = Fire" bias)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),

            # Geometric Transformations
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),

            # Color Jitter (CRITICAL):
            # Randomly changes brightness/contrast so model can't rely on color alone
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Prediction Transform (No randomness)
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.classes = ['No Fire', 'Fire']

        # 3. Auto-load weights if they exist
        self.load_weights()

    def load_weights(self):
        possible_paths = [
            'models/fire_detector_cnn.pth',
            '../models/fire_detector_cnn.pth',
            'WILDFIRE_PROJECT_ML/models/fire_detector_cnn.pth'
        ]

        loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    state_dict = torch.load(path, map_location=torch.device('cpu'))
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    print(f"✅ SUCCESSFULLY LOADED WEIGHTS FROM: {path}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ Found file at {path} but failed to load: {e}")

        if not loaded:
            print("ℹ️ No trained model found. Ready to train.")

    def load_dataset(self, data_dir='../data/fire_data'):
        base_path = os.path.abspath(data_dir)
        print(f"\n📂 Loading images from {base_path}...")

        image_paths = []
        labels = []

        # 1. Fire Images (Label 1)
        fire_dir = os.path.join(data_dir, 'fire_images')
        if os.path.exists(fire_dir):
            files = [os.path.join(fire_dir, f) for f in os.listdir(fire_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_paths.extend(files)
            labels.extend([1] * len(files))
            print(f"   🔥 Fire images found: {len(files)}")

        # 2. Non-Fire Images (Label 0)
        no_fire_dir = os.path.join(data_dir, 'non_fire_images')
        if os.path.exists(no_fire_dir):
            files = [os.path.join(no_fire_dir, f) for f in os.listdir(no_fire_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_paths.extend(files)
            labels.extend([0] * len(files))
            print(f"   ✅ Non-fire images found: {len(files)}")

        if not image_paths:
            raise ValueError(f"❌ No images found in {base_path}. Check folder names!")

        return image_paths, labels

    def train(self, data_dir='../data/fire_data', epochs=10, batch_size=16):
        """Train the model"""
        print(f"\n🔥 STARTING TRAINING (Epochs: {epochs})")

        image_paths, labels = self.load_dataset(data_dir)

        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )

        train_ds = FireImageDataset(X_train, y_train, self.transform)
        test_ds = FireImageDataset(X_test, y_test, self.test_transform)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        best_acc = 0.0

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            self.model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

            train_acc = 100 * correct / total
            test_acc = 100 * test_correct / test_total

            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")

            if test_acc >= best_acc:
                best_acc = test_acc
                os.makedirs('models', exist_ok=True)
                torch.save(self.model.state_dict(), 'models/fire_detector_cnn.pth')
                if os.path.exists('../models'):
                    torch.save(self.model.state_dict(), '../models/fire_detector_cnn.pth')

        print(f"\n🏆 Training Complete. Best Accuracy: {best_acc:.1f}%")
        return best_acc

    def predict(self, image_path):
        """Run inference on a single image"""
        self.model.eval()

        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.test_transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probs = torch.softmax(outputs, dim=1).numpy()[0]

                # Explicit Mapping: 0=No Fire, 1=Fire
                score_no_fire = probs[0]
                score_fire = probs[1]

                if score_fire > score_no_fire:
                    return "Fire", score_fire * 100
                else:
                    return "No Fire", score_no_fire * 100

        except Exception as e:
            print(f"Prediction Error: {e}")
            return "Error", 0.0

    def test_predictions(self, data_dir='../data/fire_data', num_samples=5):
        """Sanity check"""
        print("\n🧪 Running Sanity Check...")
        try:
            paths, labels = self.load_dataset(data_dir)
            fire_indices = [i for i, x in enumerate(labels) if x == 1]
            if fire_indices:
                print("\nChecking FIRE images:")
                for i in np.random.choice(fire_indices, min(len(fire_indices), num_samples), replace=False):
                    res, conf = self.predict(paths[i])
                    status = "✅" if res == "Fire" else "❌"
                    print(f"  {status} Image: {os.path.basename(paths[i])} -> Predicted: {res} ({conf:.1f}%)")
        except: pass

if __name__ == '__main__':
    detector = FireDetectorCNN()

    # 1. Delete old weights if you want to force retrain, otherwise this detects them
    model_exists = os.path.exists('models/fire_detector_cnn.pth')

    # 2. Check for data
    data_path = None
    if os.path.exists('../data/fire_data'): data_path = '../data/fire_data'
    elif os.path.exists('data/fire_data'): data_path = 'data/fire_data'

    if data_path:
        # Force training if user deleted the file, otherwise skip
        if not model_exists:
            print("🚫 No model found. Training from scratch...")
            detector.train(data_dir=data_path, epochs=10) # 10 Epochs is the sweet spot

        detector.test_predictions(data_dir=data_path)
    else:
        print("❌ Error: Data directory not found.")