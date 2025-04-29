from ultralytics import YOLO

import torch

model = YOLO("yolo/yolo11s.pt")

state_dict = torch.load("yolo/yolo11s_rec.pt")

model.load_state_dict(state_dict)

model.save("yolo/yolo11s_rec_new.pt")

# com YOLO instalado chamar no terminal dentro da pasta de instalação do yolo
# > yolo val detect model=yolo11s_rec.pt data=coco.yaml
# dataset é baixado automatico caso não exista
# fica salvo em diretório ../datasets
