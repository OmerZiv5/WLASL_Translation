import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Dataset
# -----------------------------
class WLASLDataset(Dataset):
    def _init_(self, root_dir, transform=None, max_per_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []

        for word in sorted(os.listdir(root_dir)):
            word_path = os.path.join(root_dir, word)
            if not os.path.isdir(word_path):
                continue

            samples_for_word = []
            for sample in sorted(os.listdir(word_path)):
                sample_path = os.path.join(word_path, sample)
                if os.path.isdir(sample_path):
                    samples_for_word.append(sample_path)

            # Limit number of samples per category if requested
            if max_per_class is not None and len(samples_for_word) > max_per_class:
                samples_for_word = random.sample(samples_for_word, max_per_class)

            self.samples.extend(samples_for_word)
            self.labels.extend([word] * len(samples_for_word))

        self.le = LabelEncoder()
        self.labels_encoded = self.le.fit_transform(self.labels)

    def _len_(self):
        return len(self.samples)

    def _getitem_(self, idx):
        sample_path = self.samples[idx]
        frames = []
        for f in sorted(os.listdir(sample_path)):
            img_path = os.path.join(sample_path, f)
            if not os.path.isfile(img_path):
                continue
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                continue
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        if len(frames) == 0:
            return self._getitem_((idx + 1) % len(self.samples))

        frames = torch.stack(frames)  # (num_frames, C, H, W)
        label = self.labels_encoded[idx]
        return frames, label

# -----------------------------
# Transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -----------------------------
# Dataset creation
# -----------------------------

# in the following 2 commands, change to the difference folder's paths if necessary
train_dataset = WLASLDataset(r"D:\Isolated WLASL Files\Train Frames", transform=train_transform, max_per_class=30)
test_dataset = WLASLDataset(r"D:\Isolated WLASL Files\Test Frames", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

num_classes = len(train_dataset.le.classes_)

# -----------------------------
# Model
# -----------------------------
class WLASLTranslator(nn.Module):
    def _init_(self, num_classes, hidden_size=256, pretrained=True):
        super()._init_()
        backbone = models.resnet18(pretrained=pretrained)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_size = backbone.fc.in_features
        self.rnn = nn.GRU(input_size=self.feature_size, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(batch_size * num_frames, C, H, W)
        features = self.cnn(x)
        features = features.view(batch_size, num_frames, -1)
        _, h_n = self.rnn(features)
        h_n = h_n.squeeze(0)
        out = self.classifier(h_n)
        return out

# -----------------------------
# Training
# -----------------------------
if _name_ == '_main_':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WLASLTranslator(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (frames, labels) in enumerate(train_loader):
            frames = frames.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Per-batch print
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], "
                  f"Loss: {running_loss/(i+1):.4f}, Accuracy: {100*correct/total:.2f}%")

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for frames, labels in test_loader:
                frames = frames.to(device)
                labels = labels.to(device)
                outputs = model(frames)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss /= len(test_loader)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")
