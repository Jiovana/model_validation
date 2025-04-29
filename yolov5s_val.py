import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
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

model_loc = '/home/jiovana/Documents/nncodec/example/models/yolov5_rec.pth'
#model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=False).to(device)
model = torch.hub.load('ultralytics/yolov5','yolov5s', pretrained=True)
#state_dict = torch.load(model_loc, map_location=device)
#model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode


#print(model)

# Check if the model is loaded correctly
#for param_tensor in state_dict:
#    print(f"{param_tensor}: {state_dict[param_tensor].size()}")



# Assuming 'val' directory contains the ImageNet validation images
data_dir = '/media/jiovana/Data1/coco2017/val2017'
annotations_file =  '/media/jiovana/Data1/coco2017/annotations/instances_val2017.json'


# Define transformations for validation data
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



def collate_fn(batch):
    """
    Custom collate function to handle variable-sized targets.
    """
    images, targets = zip(*batch)  # Unzip the batch into images and targets

    # Pad the targets to the maximum length in the batch
    max_len = max(len(t) for t in targets)
    padded_targets = [t + [{"bbox": [0, 0, 0, 0], "category_id": -1}] * (max_len - len(t)) for t in targets] 

    # Convert lists to tensors
    images = torch.stack(images) 
    # Create lists to hold bboxes and category_ids
    all_bboxes = []
    all_category_ids = []
    
    for target_list in padded_targets:  
        bboxes = []
        category_ids = []
        for target_dict in target_list:
            bboxes.append(target_dict['bbox'])
            category_ids.append(target_dict['category_id'])
            
        all_bboxes.append(torch.tensor(bboxes, dtype=torch.float32))  # Assuming bboxes are floats
        all_category_ids.append(torch.tensor(category_ids, dtype=torch.int64))  # Assuming category_ids are integers
    
    # Pad the bboxes and category_ids to the maximum length
    all_bboxes = pad_sequence(all_bboxes, batch_first=True, padding_value=0)
    all_category_ids = pad_sequence(all_category_ids, batch_first=True, padding_value=-1)
    
    return images, all_bboxes, all_category_ids

def extract_annotation_info(annotation):
  #Extracts bounding box and category ID from a COCO annotation.
  bbox = annotation['bbox']  # Get bounding box directly
  category_id = annotation['category_id']  # Get category ID directl
  return bbox, category_id

class CocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.images = self.annotations['images']
        self.annotations = self.annotations['annotations']
        self.image_id_to_annotations = {}
        for annotation in self.annotations:
          if annotation['image_id'] not in self.image_id_to_annotations:
            self.image_id_to_annotations[annotation['image_id']] = []
          self.image_id_to_annotations[annotation['image_id']].append(annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image_path = os.path.join(self.root_dir, img_info['file_name'])

        image = Image.open(image_path).convert("RGB")
        
        # Get annotations for this image
        annotations = self.image_id_to_annotations.get(img_info['id'], [])
        
        targets = []
        for annotation in annotations:
            print("Image id \n\n\n")
            print(annotation['image_id'])
            bbox, category_id = extract_annotation_info(annotation)
            target = {
                "bbox": bbox,
                "category_id": category_id
            }
            
            
            targets.append(target) 
        
        
        if self.transform:
            image = self.transform(image)
            
       
        print(f" targets: {targets}")
        return image, targets


dataset = CocoDataset(data_dir, annotations_file, transform=val_transforms)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2, collate_fn=collate_fn) # Adjust parameters

    
# Validation loop
correct = 0
total = 0

# Print a few samples
# for i in range(2):
#     print("New BATCH ------------------------ \n\n");
#     image, targets = dataset[i]
#     print(f" targets: {targets}")


with torch.no_grad():  # Disable gradient calculation during validation
    for images, bboxes, categories,  in tqdm(dataloader, desc="Validating", unit="batch"):
        images = images.to(device)
        results = model(images)

        # Process predictions for each image in the batch
        for i in range (images.shape[0]):  # Iterate through images in batch
            predicted_bboxes = results[:, :4].cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
            predicted_categories = results[:, -1].cpu().numpy().astype(int)  # Category IDs

            # Get ground truth for the current image
            ground_truth_bbox = bboxes[i]
            ground_truth_category = categories[i]

            # Compare predictions with ground truth (example - you'll need to define your criteria)
            # For example, check if IoU is above a threshold and category matches
            # ... (Your comparison logic here) ...
            
            # Example: Print predictions and ground truth for inspection
            print(f"Image {i}:")
            print("Predicted BBoxes:", predicted_bboxes)
            print("Predicted Categories:", predicted_categories)
            print("Ground Truth BBox:", ground_truth_bbox)
            print("Ground Truth Category:", ground_truth_category)
            print("-" * 20)

        # Calculate accuracy
        #total += labels.size(0)
        #correct += (predicted == labels).sum().item()

#accuracy = 100 * correct / total
#print(f"Validation Accuracy: {accuracy:.2f}%")

