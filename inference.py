import torch
from PIL import Image
from torchvision import transforms, models

IMG_SIZE = 224
MODEL_PATH = "model.pt"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 2)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()
        return "scratch" if pred == 1 else "no_scratch"
print("Predictions:")
print("Prediction 1:")
print(predict("test1.jpg"))
print("Prediction 2:")
print(predict("test2.jpg"))
print("Prediction 3:")
print(predict("test3.jpg"))
print("Prediction 4:")
print(predict("test4.jpg"))
print("Prediction 5:")
print(predict("test5.jpg"))