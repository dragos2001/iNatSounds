
import src.utils as utils

import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import json

def update_ema_variables(model, ema_model, ema_decay=0.999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(ema_decay).add_(1 - ema_decay, param.data)


class BCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCELoss(reduction="none")
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, label):
        batch_size, num_classes = pred.shape
        if len(label.shape) == len(pred.shape):
            label_oh = (label > 0).to(torch.float)
        else:
            not_background = 1.0*(label != -1)
            label_oh = nn.functional.one_hot((not_background * label).to(torch.long), num_classes=num_classes).to(torch.float)
            label_oh = not_background.unsqueeze(-1) * label_oh
        losses = self.loss_fn(self.sigmoid(pred), label_oh)
        return losses.sum() / batch_size

def mask_prediction(prediction, mask):
    mask = 1.0 * mask
    return prediction * mask + prediction.min().item() * (1 - mask)

def run_loop(args, dataloader, model, mode="train", optimizer=None, use_cuda=True, save_dir=None, ema_model=None, epoch=50, geo_model=None, test_geo_mask=False):
    num_classes = dataloader.num_classes
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=-1)
    if mode == "train":
        model.train()
        if ema_model is not None:
            ema_model.train()
    else:
        model.eval()
        output_dim = num_classes
        metrics = utils.SingleLabelMetrics(args.json_dir, output_dim, dataloader.present_classes_mask, save_dir, multilabel=args.multilabel)
    augment = transforms.RandomErasing(p=1.0, scale=(0.1, 0.33))

    if use_cuda:
        dataloader.present_classes_mask = dataloader.present_classes_mask.cuda()

    if args.multilabel:
        loss_fn = BCE()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    avg_loss = []
    avg_acc = [] 
    batch_pbar = tqdm.tqdm(total=len(dataloader), desc=mode + " batches", leave=False)
    for batch_id, batch in enumerate(dataloader):

        img, label, geo, img_name = batch
        
        if use_cuda:
            img = img.cuda()
            label = label.cuda()
            geo = geo.cuda()

        if geo_model is not None:

            geo_prior = geo_model(geo)
            geo_prior = geo_prior > args.geo_threshold
            geo_prior = 1.0 * geo_prior
        else:
            geo_prior = None

        if mode == "train":
            optimizer.zero_grad()

            pred = model(img)

            if args.mean_teacher:
                with torch.no_grad():
                    if args.clean_teacher:
                        teacher_img, lam = batch[-2:]
                        if use_cuda:
                            teacher_img = teacher_img.cuda()
                            lam = lam.cuda()
                    else:
                        teacher_img = img
                    if args.ema_augment:
                        pseudo_label = ema_model(augment(teacher_img))
                    else:
                        pseudo_label = ema_model(img)
                    pseudo_label = pseudo_label.detach().clone()
                    
                    if args.clean_teacher:
                        pseudo_label = torch.maximum(torch.roll(pseudo_label, 1, 0), pseudo_label)
                    


                    if args.pseudo_geo:
                        pseudo_label = mask_prediction(pseudo_label, geo_prior)

            pred_cl = pred[..., :num_classes]

        else:
            if len(img.shape) > 4:
                img = img.squeeze(0)
            bsz = img.shape[0]
            img_batches = [
                img[args.batch_size * i: min(args.batch_size * (i+1), bsz), ...]
                for i in range((bsz + args.batch_size - 1)// args.batch_size)
            ]
            with torch.no_grad():
                pred = torch.cat([
                    model(img_batch).detach().clone()
                    for img_batch in img_batches
                ], 0)

            pred_cl = pred[..., :num_classes]
            pred_cl = mask_prediction(pred_cl, dataloader.present_classes_mask)

            if test_geo_mask and geo_prior is not None:
                pred_cl = mask_prediction(pred_cl, geo_prior)

        

        if args.multilabel and args.geo_train and not args.pseudo_geo:
            loss = loss_fn(pred_cl, label, geo_prior)
        else:
            loss = loss_fn(pred_cl, label)
       
        if mode == "train" and args.mean_teacher:
           
            ramp = 1
            teacher_loss = 0
            if args.teacher_loss == "pred_mse":
                act = sigmoid if args.multilabel else softmax
                ps_act = softmax if args.single_teacher else act
                teacher_loss = mse_loss(act(pred_cl), ps_act(pseudo_label))
            elif args.teacher_loss == "logit_mse":
                teacher_loss = mse_loss(pred_cl, pseudo_label)
            elif args.teacher_loss == "bce":
                teacher_loss = loss_fn(pred_cl, sigmoid(pseudo_label))
            loss = loss + ramp * args.ema_weight * teacher_loss
            # loss += ramp * args.ema_weight * loss_fn(pred_cl, pseudo_label)


        if args.mixup and mode == "train":
            acc = (1.0*(pred_cl.argmax(-1) == label.argmax(-1))).mean()
        else:
            acc = (1.0*(pred_cl.argmax(-1) == label)).mean()

        if mode == "train":
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()
            
            if args.mean_teacher:
                update_ema_variables(model, ema_model, ema_decay=args.ema_decay)
        else:
            metrics.update(
                label.cpu(), pred_cl.cpu(), img_name
            )
        avg_loss.append(loss.item())
        avg_acc.append(acc.item())
        batch_pbar.update(1)
        batch_pbar.set_description("Batches {} | Loss: {:.4f} ACC: {:.4f}".format(mode, loss.item(), acc))
        # if (batch_id+1)%1000 == 0:
        #     print(metrics.get_values())
    avg_loss = sum(avg_loss) / len(avg_loss) if len(avg_loss) else 0
    avg_acc = sum(avg_acc) / len(avg_acc) if len(avg_acc) else 0

    if mode == "train":
        return avg_loss, avg_acc

    metric_dict = metrics.get_metric_str()
    return avg_loss, avg_acc, metric_dict
