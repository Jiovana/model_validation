import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_loc = '/home/jiovana/Documents/nncodec/example/models/yolov5_rec.pth'
#model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=False).to(device)
model = torch.hub.load('ultralytics/yolov5','yolov5s', pretrained=True).to(device)
#state_dict = torch.load(model_loc, map_location=device)
#model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode

# Check if the model is loaded correctly
#for param_tensor in state_dict:
#    print(f"{param_tensor}: {state_dict[param_tensor].size()}")

# Paths to dataset base and annotations
data_dir = '/media/jiovana/Data1/coco2017/val2017'
annotations_file =  '/media/jiovana/Data1/coco2017/annotations/instances_val2017.json'

# Load COCO ground truth
coco_gt = COCO(annotations_file)

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
    images, targets, image_ids = zip(*batch)  # Unzip the batch into images and targets

    # Pad the targets to the maximum length in the batch
    max_len = max(len(t) for t in targets)
    padded_targets = [t + [{"bbox": [0, 0, 0, 0], "category_id": -1}] * (max_len - len(t)) for t in targets]

    # Convert lists to tensors
    images = torch.stack(images)
    image_ids = torch.tensor(image_ids, dtype=torch.int64)
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
    
    return images, all_bboxes, all_category_ids, image_ids

def extract_annotation_info(annotation):
  #Extracts bounding box and category ID from a COCO annotation.
  image_id = annotation['image_id']
  bbox = annotation['bbox']  # Get bounding box directly
  category_id = annotation['category_id']  # Get category ID directl
  return image_id, bbox, category_id

class CocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = self.coco.getImgIds()
        """ with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.images = self.annotations['images']
        self.annotations = self.annotations['annotations']
        self.image_id_to_annotations = {}
        for annotation in self.annotations:
          if annotation['image_id'] not in self.image_id_to_annotations:
            self.image_id_to_annotations[annotation['image_id']] = []
          self.image_id_to_annotations[annotation['image_id']].append(annotation) """

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        targets = self.coco.loadAnns(ann_ids)

        return image, targets, image_id  # Return image_id
    """
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
            # print("Image id \n\n\n")
            # print(annotation['image_id'])
            image_id, bbox, category_id = extract_annotation_info(annotation)
            target = {
                "image_id": image_id,
                "bbox": bbox,
                "category_id": category_id
            }
            
            
            targets.append(target) 
        
        
        if self.transform:
            image = self.transform(image)
            
       
        # print(f" targets: {targets}")
        return image, targets """


dataset = CocoDataset(data_dir, annotations_file, transform=val_transforms)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=2, collate_fn=collate_fn) # Adjust parameters

    
# Validation loop
correct = 0
total = 0

# Print a few samples
# for i in range(2):
#     print("New BATCH ------------------------ \n\n");
#     image, targets = dataset[i]
#     print(f" targets: {targets}")

results = []
with torch.no_grad():  # Disable gradient calculation during validation
    for images, all_bboxes, all_category_ids, image_ids  in tqdm(dataloader, desc="Validating", unit="batch"):
        images = images.to(device)
        outputs = model(images)

        # Process YOLOv5 outputs and convert to COCO format
        for i in range(images.shape[0]):  # Iterate through images in the batch
            # Access image_id
            #print(f"\n\n\n\n image_ids: {all_image_ids}")
            image_id =  image_ids[i].item()
            # Get predictions for current image
            detections = outputs[i]  # Assuming each element in output is for one image

            # Iterate over the detections, handling potential empty cases
            if detections is not None and len(detections) > 0:
                for detection in detections:  # Extract the data from each detection
                    xyxy = detection[:4]  # Extract x1, y1, x2, y2 coordinates
                    conf = detection[4]  # Extract confidence score
                    cls = detection[5]  # Extract class ID
                    x1, y1, x2, y2 = map(float, xyxy)
                    width = x2 - x1
                    height = y2 - y1
                    category_id = int(cls)  # Convert to integer

                    results.append({
                        'image_id': image_id,
                        'category_id': category_id,
                        'bbox': [x1, y1, width, height],  # Assuming COCO format needs x, y, width, height
                        'score': float(conf)
                    })



# Evaluate using COCOeval
coco_dt = coco_gt.loadRes(results)
print("Passed coco_Dt")
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
print("Passed coco_Eval")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
           


