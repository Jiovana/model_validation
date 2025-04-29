import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import os
import json
from PIL import Image
from tqdm import tqdm


is_gpu = torch.cuda.is_available()
if is_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#model_loc = '/home/jiovana/Documents/nncodec/example/models/alexnet_rec.pth'
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=False).to(device)
#model = model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
#state_dict = torch.load(model_loc, map_location=device)
#model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode


print(model)

# Check if the model is loaded correctly
#for param_tensor in state_dict:
#    print(f"{param_tensor}: {state_dict[param_tensor].size()}")



# Assuming 'val' directory contains the ImageNet validation images
data_dir = '/media/jiovana/Data/ImageNet-Mini/images'
annotations_file =  '/media/jiovana/Data/ImageNet-Mini/imagenet_class_index.json'

# Define transformations for validation data
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class MiniImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform    
        self.labels = []
        self.data = []
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(root_dir)))}
        
        # Traverse the directories and collect image paths and labels
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                if file_path.lower().endswith(('jpg', 'jpeg', 'JPEG')):
                    self.data.append(file_path)
                    self.labels.append(class_idx)
     

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label  # Return the image and the label 

# Create the custom dataset instance
val_dataset = MiniImageNetDataset(data_dir, transform=val_transforms)

# Create DataLoader
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)  # num_workers can be adjusted

# Print a few samples
#for i in range(5):
#    image, label = val_dataset[i]
#    print(f"Image path: {val_dataset.data[i]}, Label: {label}")
    
    
# Validation loop
correct = 0
total = 0

# Read the categories
with open(annotations_file, "r") as f:
    idx_to_human = json.load(f)

with torch.no_grad():  # Disable gradient calculation during validation
    for images, labels in tqdm(val_loader, desc="Validating", unit="batch"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)


        # Map indices to human-readable labels
        predicted_labels = [idx_to_human[str(idx.item())][1] for idx in predicted]
        true_labels = [idx_to_human[str(idx.item())][1] for idx in labels]

        # Print predictions for debugging
       # for true_label, predicted_label in zip(true_labels, predicted_labels):
        #    print(f"True: {true_label}, Predicted: {predicted_label}")

        # Calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")