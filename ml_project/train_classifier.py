# Preprocessing
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_dataset = CustomDataset(train_images, train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 5)  # 5 classes

# Training loop
for epoch in range(epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
