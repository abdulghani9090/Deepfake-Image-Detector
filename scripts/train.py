import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from model import DeepfakeDetectorModel
from tqdm import tqdm

class FaceDataset(Dataset):
    def __init__(self, root_dir):
        self.images = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        for label, folder in enumerate(["real", "fake"]):
            full_path = os.path.join(root_dir, folder)
            for img_file in os.listdir(full_path):
                self.images.append(os.path.join(full_path, img_file))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        return self.transform(image), self.labels[idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = FaceDataset("data")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = DeepfakeDetectorModel().to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


for epoch in range(5):
    total_loss = 0
    model.train()
    for inputs, labels in tqdm(dataloader):
        # 1. Prepare inputs (images)
        inputs = inputs.to(device)
        
        # 2. Prepare labels: Convert to float AND ensure shape is (BatchSize, 1)
        # The .unsqueeze(1) adds the required dimension.
        labels = labels.float().unsqueeze(1).to(device) # <--- FIXED LINE
        
        optimizer.zero_grad()
        
        # 3. Get outputs: Do NOT use .squeeze()
        outputs = model(inputs) # <--- FIXED LINE
        
        # Now, outputs.shape should be (BatchSize, 1) 
        # and labels.shape should be (BatchSize, 1). They match!
        
        loss = criterion(outputs, labels) # Now this works correctly
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader)}")


torch.save(model.state_dict(), "models/deepfake_model.pth")
print("Model saved.")
