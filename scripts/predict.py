import torch
from torchvision import transforms
from PIL import Image
from model import DeepfakeDetectorModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeDetectorModel().to(device)
model.load_state_dict(torch.load("models/deepfake_model.pth"))
model.eval()

def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    label = "Fake" if prob > 0.5 else "Real"
    print(f"Prediction: {label} ({prob*100:.2f}% confidence)")

# Test it:
predict_image("test.jpg")  # Replace with your test image path
