import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image, ImageEnhance, ImageFile
import wandb
from model import HybridClassifier

ImageFile.LOAD_TRUNCATED_IMAGES = True

# WandB setup
wandb.init(project="plant-augment", config={
    "batch_size": 32,
    "epochs": 35,
    "learning_rate": 1e-5,
})
config = wandb.config

# Configurations
data_dir = "/kaggle/input/plant-augment/data"
output_dir = "/kaggle/working/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define preprocessing function
def preprocess_image(image):
    if random.random() < 0.5:
        crop_transform = T.CenterCrop((int(image.height * 0.8), int(image.width * 0.8)))
        image = crop_transform(image)

    resize_transform = T.Resize((224, 224))
    image = resize_transform(image)

    if random.random() < 0.5:
        angle = random.uniform(-30, 30)
        image = T.functional.rotate(image, angle)

    if random.random() < 0.5:
        if random.random() < 0.5:
            image = T.functional.hflip(image)
        else:
            image = T.functional.vflip(image)

    image_tensor = T.ToTensor()(image)
    if random.random() < 0.5:
        std_dev = random.uniform(5 / 255, 10 / 255)
        noise = torch.randn(image_tensor.size()) * std_dev
        image_tensor = torch.clamp(image_tensor + noise, 0, 1)

    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_tensor = normalize_transform(image_tensor)
    return image_tensor

def transform(image):
    return preprocess_image(image)

# Custom Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = transform(image)
        return image, label

# Load dataset
original_dataset = datasets.ImageFolder(root=data_dir)
train_idx, val_idx = train_test_split(
    list(range(len(original_dataset))), test_size=0.2, random_state=42, stratify=original_dataset.targets
)
print(len(original_dataset.classes))
train_dataset = CustomDataset(torch.utils.data.Subset(original_dataset, train_idx))
val_dataset = CustomDataset(torch.utils.data.Subset(original_dataset, val_idx))

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

model = HybridClassifier()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Resume from checkpoint if applicable
start_epoch = 0
best_val_accuracy = 0.0  # Track the best validation accuracy observed so far
checkpoint_path = "/kaggle/input/checkpoint-30/best_checkpoint_2.pth"  # Path to your checkpoint file

# Check if a checkpoint exists and load it
if os.path.exists(checkpoint_path):
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
    print(f"Resuming training from epoch {start_epoch}, Best Val Accuracy: {best_val_accuracy}")
else:
    print("No checkpoint found. Starting training from scratch.")

# Training loop
for epoch in range(start_epoch, config.epochs):
    model.train()
    total_loss = 0

    for step, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step % 10 == 0:
            wandb.log({"Training Loss": loss.item(), "Step": step})
            print(f"Training Loss:: {loss.item()}")

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_loss += criterion(val_outputs, val_labels).item()
            preds = torch.argmax(val_outputs, dim=1)
            correct += (preds == val_labels).sum().item()
            total += val_labels.size(0)

    val_accuracy = correct / total
    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch + 1}/{config.epochs}, Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    # Log metrics to WandB
    wandb.log({
        "Epoch": epoch + 1,
        "Average Training Loss": avg_train_loss,
        "Validation Loss": avg_val_loss,
        "Validation Accuracy": val_accuracy
    })

    # Save checkpoint if the current validation accuracy is the best
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_accuracy
        }
        best_checkpoint_path = os.path.join(output_dir, "best_checkpoint_3.pth")
        torch.save(checkpoint, best_checkpoint_path)
        print(f"Best checkpoint saved with accuracy: {best_val_accuracy:.4f}")

# Save final model
os.makedirs(output_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
print(f"Final model saved to {output_dir}")

# Classification report
print("Generating classification report...")
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

report = classification_report(all_labels, all_preds, target_names=original_dataset.classes)
print(report)

wandb.finish()