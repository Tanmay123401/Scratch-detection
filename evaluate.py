import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

DATA_DIR = "data"
MODEL_PATH = "model.pt"
IMG_SIZE = 224
BATCH_SIZE = 32

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

# Train/Val split (same as training)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 2)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to compute accuracy
def calc_accuracy(loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc, all_labels, all_preds


# TRAIN ACCURACY
train_acc, _, _ = calc_accuracy(train_loader)
print(f"Train Accuracy: {train_acc:.4f}")

# VALIDATION ACCURACY + REPORT
val_acc, val_labels, val_preds = calc_accuracy(val_loader)
print(f"Validation Accuracy: {val_acc:.4f}")

report = classification_report(val_labels, val_preds, target_names=["no_scratch", "scratch"])
print("\nClassification Report:\n")
print(report)

# Save report
with open("classification_report.txt", "w") as f:
    f.write(f"Train Accuracy: {train_acc:.4f}\n")
    f.write(f"Validation Accuracy: {val_acc:.4f}\n\n")
    f.write(report)