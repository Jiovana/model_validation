# THIS APPROACH DID NOT WORK.

import torch
from compressai.zoo import cheng2020_anchor

model_cheng = cheng2020_anchor(quality=2, pretrained=False).eval()

model_cheng.load_state_dict

state_dict = torch.load("/home/jiovana/Documents/model_validation/cheng_rec.pth", weights_only=True)
model_cheng.load_state_dict(state_dict)
torch.save(model_cheng, "/home/jiovana/Documents/model_validation/cheng_rec_new.pth")
print("done")


#!python -m compressai.utils.eval_model checkpoint /datasets/professional_test_2021/ -a cheng2020-anchor -p cheng_rec_new.pth  -o cheng_rec_output.json --per-image
# /datasets/