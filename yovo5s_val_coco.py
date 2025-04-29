import torch
from torch.utils.data import Dataset, DataLoader
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
    transforms.Resize((640,640)),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset class for COCO
class CocoDataset(Dataset):
    def __init__(self, coco, image_dir, transforms=None):
        self.coco = coco
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_ids = coco.getImgIds()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        # Load image info from COCO
        img_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        original_size = (img_info['width'], img_info['height'])

        if self.transforms:
            image = self.transforms(image)

        return image, image_id, original_size

# Create dataset and dataloader
dataset = CocoDataset(coco_gt, data_dir, transforms=val_transforms)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

# Perform inference
def perform_inference(model, dataloader, device):
    """
    Perform batch inference on the dataset and return predictions in COCO format.
    """
    results = []

    for images, image_ids, original_sizes in tqdm(dataloader, desc="Performing inference"):
        images = images.to(device)

        # Perform inference
        with torch.no_grad():
            preds = model(images)

        # Parse predictions for each image in the batch
        for i, pred in enumerate(preds.xyxy):  # `pred` contains predictions for one image
            img_id = image_ids[i].item()
            original_width, original_height = original_sizes[i]

            for detection in pred:
                x1, y1, x2, y2, conf, cls_id = detection[:6]
                # Convert to COCO format bbox: [x, y, width, height]
                bbox = [
                    (x1.item() / images.shape[-1]) * original_width,
                    (y1.item() / images.shape[-2]) * original_height,
                    ((x2.item() - x1.item()) / images.shape[-1]) * original_width,
                    ((y2.item() - y1.item()) / images.shape[-2]) * original_height,
                ]
                results.append({
                    'image_id': img_id,
                    'category_id': int(cls_id.item()),  # category is class ID
                    'bbox': [round(x, 2) for x in bbox],
                    'score': round(conf.item(), 3),
                })

    return results

# Run inference and collect predictions
results = perform_inference(model, dataloader, device)

# Fill missing ground truth images with dummy predictions for COCOeval
all_gt_image_ids = set(coco_gt.getImgIds())
predicted_image_ids = set(r['image_id'] for r in results)
missing_image_ids = all_gt_image_ids - predicted_image_ids

for img_id in missing_image_ids:
    results.append({'image_id': img_id, 'category_id': -1, 'bbox': [0, 0, 0, 0], 'score': 0})

# Evaluate using COCOeval
coco_dt = coco_gt.loadRes(results)

coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

           


