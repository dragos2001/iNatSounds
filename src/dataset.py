import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import json
import tqdm
import torchvision
from torchvision.transforms import v2
from torch.utils.data import default_collate


class CustomMixup():
    def __init__(self, num_classes, alpha=1.0, multilabel=False):
        self.num_classes = num_classes
        self.alpha = alpha
        self.dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def transform(self, inpt, lam):
        return inpt.roll(1, 0).mul_(1.0 - lam).add_(inpt.mul(lam))

    def __call__(self, images, labels):
        lam = self.dist.sample()
        mix_images = self.transform(images, lam)
        labels_oh = torch.nn.functional.one_hot(labels, num_classes=self.num_classes)
        mix_labels = self.transform(labels_oh.float(), lam)
        return mix_images, mix_labels, lam 



class InatJsonDataset(Dataset):
    def __init__(self, data_root, split_path, transforms, sound_aug=False, window_len=512, mode="test", test_stride=256, no_masking=False):

        self.transforms = transforms
        self.mode = mode
        self.sound_aug = sound_aug
        self.window_len = window_len
        self.test_stride = test_stride
        self.no_masking = no_masking
        
        with open(split_path, "r") as f:
            dataset = json.load(f)
            
        self.class_name2idx = {
            c["audio_dir_name"]:c["id"]
            for c in dataset["categories"]
        }
        self.class_idx2name = {
            c["id"]:c["audio_dir_name"]
            for c in dataset["categories"]
        }
        audio_id2path = {
            au["id"]:au["file_name"]
            for au in dataset["audio"]
        }
        self.datapoints = [
            {
                "path": os.path.join(data_root, audio_id2path[a["audio_id"]].replace(".wav", ".npy")),
                "class_name": self.class_idx2name[a["category_id"]],
                "name": audio_id2path[a["audio_id"]].replace(".wav", ""),
            }
            for a in dataset["annotations"]
        ]
        self.num_classes = len(list(self.class_name2idx.keys()))

            
        classes = [d["class_name"] for d in self.datapoints]
        classes = sorted(list(set(classes)))
        class_idx = [self.class_name2idx[c] for c in classes]
        self.present_classes_mask = torch.zeros(self.num_classes)
        for c in class_idx:
            self.present_classes_mask[c] = 1
        if self.no_masking:
            for c in range(self.num_classes):
                self.present_classes_mask[c] = 1


        audio_id2geo = {
            au["id"]: (au["latitude"], au["longitude"])
            for au in dataset["audio"]
        }
        scale = torch.tensor([90, 180])
        self.geo = [
            torch.Tensor(audio_id2geo[a["audio_id"]]) / scale
            for a in dataset["annotations"]
        ]
        self.num_geo_classes = 1

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):

        dp = self.datapoints[idx]
        img_path, class_name, img_name = dp["path"], dp["class_name"], dp["name"]

        label = self.class_name2idx[class_name] if class_name is not None else -1

        img = np.load(img_path)
        img = np.stack([img]*3)

        if label == -1:
            img = (img - img.min()) * 0.5 + img.min()

        if self.mode == "train":
            img = self.time_selection(img)
            if self.sound_aug:
                img = self.time_masking(self.freq_masking(img))
        else:
            img, num = self.dense_prediction(img)
            label = torch.Tensor([label]*num).to(torch.long)
            

        img = torch.Tensor(img).to(torch.float) / 255.0
        if self.transforms is not None:
            img = self.transforms(img)

        # geo = self.geo[idx] if self.geo is not None and idx < len(self.geo) else 0
        geo = self.geo[idx] if self.geo is not None and idx < len(self.geo) else torch.Tensor([0, 0])
        return img, label, geo, [img_name]

    def dense_prediction(self, img):
        time_len = img.shape[-1]
        if time_len <= self.window_len:
            pad = self.window_len - time_len
            img = np.pad(img, ((0, 0), (0, 0), (pad//2, pad - pad//2)), constant_values=0)
            time_len = img.shape[-1]

        num = (time_len - self.window_len + self.test_stride - 1) // self.test_stride
        pad = self.test_stride * num - time_len + self.window_len
        img = np.pad(img, ((0, 0), (0, 0), (pad//2, pad - pad//2)), constant_values=0)
        
        out_list = []
        for i in range(num+1):
            start = i * self.test_stride
            if start + self.window_len > img.shape[-1]: break
            out_list.append(img[..., start : start + self.window_len])
        
        return np.stack(out_list), len(out_list)

    def time_selection(self, img):
        time_len = img.shape[-1]
        if time_len <= self.window_len:
            pad = self.window_len - time_len
            img = np.pad(img, ((0, 0), (0, 0), (pad//2, pad - pad//2)), constant_values=0)
            start = 0
        else:
            start = np.random.randint(0, time_len - self.window_len)
        return img[..., start : start + self.window_len]


    def freq_masking(self, img, mask_len=15):
        factor = np.random.RandomState().rand()
        freq_len = img.shape[-2]
        start = np.random.randint(0, freq_len - mask_len)
        interval = np.random.randint(0, mask_len)
        img[..., start : start + interval, :] = 0
        return img

    def time_masking(self, img, mask_len=15):
        time_len = img.shape[-1]
        start = np.random.randint(0, time_len - mask_len)
        interval = np.random.randint(0, mask_len)
        img[..., start : start + interval] = 0
        return img


            
        

def get_dataloaders(args):
    standard_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Normalize(
            (0.6569, 0.6569, 0.6569), (0.1786, 0.1786, 0.1786)
        ),
    ])
    if hasattr(args, "noresize") and args.noresize:
        standard_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 512)),
            torchvision.transforms.Normalize(
                (0.6569, 0.6569, 0.6569), (0.1786, 0.1786, 0.1786)
            ),
        ])

    train_transforms = standard_transforms    
    sound_aug = True if hasattr(args, "sound_aug") and args.sound_aug else False

    
    json_path_format = os.path.join(args.json_dir, "{}.json")

    data = InatJsonDataset(
        data_root=args.spectrogram_dir,
        split_path=json_path_format.format("train"),
        transforms = train_transforms,
        sound_aug=sound_aug,
        mode="train",
        no_masking=args.no_masking,
    )
    val_data = InatJsonDataset(
        data_root=args.spectrogram_dir,
        split_path=json_path_format.format("val"),
        transforms = standard_transforms,
        sound_aug=False,
        mode="test",
        no_masking=args.no_masking,
    )
    test_data = InatJsonDataset(
        data_root=args.spectrogram_dir,
        split_path=json_path_format.format("test"),
        transforms = standard_transforms,
        sound_aug=False,
        mode="test",
        no_masking=args.no_masking,
    )

    num_classes = data.num_classes
    num_geo_classes = data.num_geo_classes
    
    collate_fn = default_collate
    if hasattr(args, "mixup") and args.mixup:
        mixup = CustomMixup(num_classes=num_classes)
        def mixup_collate(batch):
            collated_batch = default_collate(batch)
            images, labels, geo = collated_batch[:3]
            images_orig = images.clone().detach()
            images = torch.exp(images - 10)
            images_mix, labels_mix, lam = mixup(images, labels)
            images_mix = torch.log(images_mix) + 10

            mixed_batch = (images_mix, labels_mix, geo, *collated_batch[3:])
            return mixed_batch
        collate_fn = mixup_collate
    
    def test_collate(batch):

        collated_batch = default_collate([b[4:] for b in batch])
        images = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        geo = [b[2] for b in batch]
        img_names = [b[3] for b in batch]

        all_img_names = []
        for i, n in enumerate(img_names):
            all_img_names.extend(n*images[i].shape[0])
        all_geo = []
        for i, g in enumerate(geo):
            all_geo.extend([g]*images[i].shape[0])
        images = torch.cat(images, 0)
        labels = torch.cat(labels, 0)
        geo = torch.stack(all_geo, 0)
        # geo =  torch.Tensor(all_geo).to(torch.long)
        full_batch = (images, labels, geo, all_img_names, *collated_batch)
        return full_batch
        
    sampler = torch.utils.data.sampler.RandomSampler(data)

    train_dataloader = DataLoader(data, batch_size=args.batch_size, collate_fn=collate_fn, sampler=sampler, num_workers=4)
    val_dataloader = DataLoader(val_data, batch_size=8, shuffle=False, collate_fn=test_collate, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False, collate_fn=test_collate, num_workers=4)

    train_dataloader.num_classes = num_classes
    val_dataloader.num_classes = num_classes
    test_dataloader.num_classes = num_classes

    train_dataloader.num_geo_classes = num_geo_classes
    val_dataloader.num_geo_classes = num_geo_classes
    test_dataloader.num_geo_classes = num_geo_classes

    train_dataloader.present_classes_mask = data.present_classes_mask
    val_dataloader.present_classes_mask = val_data.present_classes_mask
    test_dataloader.present_classes_mask = test_data.present_classes_mask

    
    return train_dataloader, val_dataloader, test_dataloader
