import torchvision
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import json
import os


hidden_dim_dict = {
    "resnet18": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "vit": 768,
    "mobilenet": 1280,
}

def get_model(model_name, output_dim, pretrained=True, get_last_dim=False):
    last_dim = hidden_dim_dict[model_name]
    if model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(last_dim, output_dim) if output_dim is not None else nn.Identity()
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(last_dim, output_dim) if output_dim is not None else nn.Identity()
    elif model_name == "resnet101":
        model = torchvision.models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(last_dim, output_dim) if output_dim is not None else nn.Identity()
    elif model_name == "vit":
        model = torchvision.models.vit_b_16(pretrained=pretrained)
        model.heads.head = nn.Linear(last_dim, output_dim) if output_dim is not None else nn.Identity()
    elif model_name == "mobilenet":
        model = torchvision.models.mobilenet_v3_large(pretrained=pretrained)
        model.classifier[3] = nn.Linear(last_dim, output_dim) if output_dim is not None else nn.Identity()

    if not get_last_dim:
        return model
    else:
        return model, last_dim


class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out

# Adapted from SINR: https://github.com/elijahcole/sinr
class GeoModel(nn.Module):

    def __init__(self, geo_model_weights, json_dir, num_classes=5547, num_inputs=4, num_filts=256, depth=4):
        super(GeoModel, self).__init__()

        dataset_json = os.path.join(json_dir, "val.json")
        inat2sci_path = "./assets/inat_id2scientific.json"

        self.inc_bias = False
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        layers = []
        layers.append(nn.Linear(num_inputs, num_filts))
        layers.append(nn.ReLU(inplace=True))
        for i in range(depth):
            layers.append(ResLayer(num_filts))
        self.feats = torch.nn.Sequential(*layers)

        self.eval()

        self.checkpoint = torch.load(geo_model_weights)
        self.load_state_dict(self.checkpoint["state_dict"])
        self.geo_class2taxa = self.checkpoint["params"]["class_to_taxa"]
        
        with open(dataset_json, "r") as f:
            val_data = json.load(f)
        with open(inat2sci_path, "r") as f:
            inat2sci = json.load(f)
            sci2inat = {v:int(k) for k, v in inat2sci.items()}
        
        inat2cls = {sci2inat[cat["name"]]:cat["id"] for cat in val_data["categories"] if cat["name"] in sci2inat}
        self.geo_pred_mask = torch.Tensor([
            i 
            for i, taxa in enumerate(self.geo_class2taxa) if taxa in inat2cls
        ]).to(torch.long)
        self.class_convert = torch.Tensor([
            inat2cls[taxa]
            for taxa in self.geo_class2taxa if taxa in inat2cls
        ]).to(torch.long)
        self.total_classes = len(inat2cls.keys())


    def forward(self, x):
        # x: B x 2
        # assumes x_i : [lat, lon] in range -1, 1
        assert ((x > 1) + (x < -1)).sum() == 0
        # print(x.shape)
        # change from [lat, lon] to [lon, lat]. That is the convention in geo model
        x = torch.flip(x, [-1])
        x_encode = torch.cat([
            torch.sin(np.pi*x), 
            torch.cos(np.pi*x)
        ], -1)
        loc_emb = self.feats(x_encode)
        pred = self.class_emb(loc_emb)
        pred = torch.sigmoid(pred)
        class_pred = torch.ones((x.shape[0], self.total_classes)).to(x.device)
        class_pred[:, self.class_convert] = pred[..., self.geo_pred_mask]
        return class_pred
