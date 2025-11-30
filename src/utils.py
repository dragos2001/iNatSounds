import os
import logging
import shutil
import torch
import json
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def metric_str2acc(m_str):
    m = m_str.replace(" ", "").split("\n")
    m = [l for l in m if l!= ""]
    m = m[1].split(":")[1].replace("%", "")
    return float(m)


def save_model(model, save_path):
    state_dict = model.state_dict()
    torch.save(state_dict, save_path)

def save_plots(plot_dict, save_path):
    for k in plot_dict.keys():
        x = list(plot_dict[k].keys())
        y = list(plot_dict[k].values())
        if len(x) == 0: continue
        plt.plot(x, y, label=k)
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

def dict_to_str(d):
    return " | ".join(["{}: {:.2f}".format(k, 100*v) for k, v in d.items()])


def setup_logging(args, mode="train"):
    exp_name = "{}/{}_b{}_lr{}_wd{}/{}".format(args.model, args.optim, args.batch_size, args.lr, args.wd, mode) 
    
    if args.no_masking:
        exp_name = "no_masking/" + exp_name
    if args.sound_aug:
        exp_name = "sound_aug/" + exp_name
    if args.pretrained:
        exp_name = "pretrained/" + exp_name
    if args.mixup:
        exp_name = "mixup/" + exp_name

    if args.loss == "geographic":
        exp_name = "geographic_loss/" + exp_name
    if args.multilabel:
        exp_name = "multilabel/" + exp_name
    if args.mean_teacher:
        exp_name = "mean_teacher/" + exp_name

    
    exp_name = exp_name if args.exp_name == "" else args.exp_name + "/" + exp_name

    # get unique number of experiment
    log_root = os.path.join(args.log_dir, exp_name)
    if os.path.exists(log_root):
        avail_nums = os.listdir(log_root)
        avail_nums = [-1] + [int(d) for d in avail_nums if d.isdigit()]
        log_num = max(avail_nums) + 1
    else:
        log_num = 0
    log_num = str(log_num)
    print("Logging in exp {}, number {}".format(exp_name, log_num))

    # get log directories and setup logger
    weight_dir = os.path.join(args.log_dir, exp_name, log_num, "checkpoints")
    plot_dir = os.path.join(args.log_dir, exp_name, log_num, "plots")
    pred_dir = os.path.join(args.log_dir, exp_name, log_num, "preds")
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, exp_name, log_num, "log.txt")
    logging.basicConfig(filename=log_path,
                    filemode='a',
                    format='%(asctime)s | %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO, force=True)
    logging.info("starting new experiment")

    # Copying the source python files
    src_dir = os.path.join(args.log_dir, exp_name, log_num, "src")
    os.makedirs(src_dir, exist_ok=True)
    cwd = os.getcwd()
    def copy_file(rel_path):
        src = os.path.join(cwd, rel_path)
        dst = os.path.join(src_dir, rel_path)
        shutil.copy(src, dst)
    for fname in os.listdir(cwd):
        if os.path.isdir(os.path.join(cwd, fname)):
            os.makedirs(os.path.join(src_dir, fname), exist_ok=True)
            for fname2 in os.listdir(os.path.join(cwd, fname)):
                if fname2.endswith(".py"):
                    copy_file(os.path.join(fname, fname2))
        if fname.endswith(".py"):
            copy_file(fname)

    return weight_dir, plot_dir, pred_dir

    

def list2str(lst):
    if type(lst) == torch.Tensor:
        lst = lst.tolist()
    return "_".join(["{:.2f}".format(l) for l in lst])

def list_average(lst):
    return sum(lst) / len(lst) if len(lst) else 0

def add_lists(lst):
    new_list = []
    for l in lst:
        new_list.extend(l)
    return new_list

class SingleLabelMetrics():
    def __init__(self, dataset_json_dir, num_classes, present_classes_mask, save_dir=None, multilabel=False):
        self.present_classes_mask = present_classes_mask
        self.present_classes_idx = [i for i in range(num_classes) if present_classes_mask[i]]
        self.present_classes_idx = torch.Tensor(self.present_classes_idx).to(torch.long)

        self.num_classes = num_classes
        self.all_classes = list(range(num_classes))
        self.metric_scales = ["sound_avg", "sound_max", "window"]
        self.top1_sum = {s:{c:0 for c in self.all_classes} for s in self.metric_scales}
        self.top5_sum = {s:{c:0 for c in self.all_classes} for s in self.metric_scales}
        self.num = {s:{c:0 for c in self.all_classes} for s in self.metric_scales}

        self.taxa_levels = ["kingdom", "phylum", "class", "order", "family", "genus", "specific_epithet"]
        self.taxa_top1_sum = {s:{l:{} for l in self.taxa_levels} for s in self.metric_scales}
        self.taxa_num =  {s:{l:{} for l in self.taxa_levels} for s in self.metric_scales}

        self.save_dir = save_dir
        self.save_paths = {}
        if self.save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            for scale in self.metric_scales:
                self.save_paths[scale] = os.path.join(save_dir, "{}_preds.csv".format(scale))
                with open(self.save_paths[scale], "w") as f:
                    f.write("img,gt,pred\n")

        self.softmax = torch.nn.Softmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()
        self.multilabel = multilabel

        dataset = read_json(os.path.join(dataset_json_dir, "test.json"))
        self.class_id2taxa_list = {}
        for cl_dict in dataset["categories"]:
            self.class_id2taxa_list[cl_dict["id"]] = [
                cl_dict[k] for k in self.taxa_levels
            ]

        self.all_preds = []
        self.all_gts = []

        self.all_preds_max = []
        self.all_gts_max = []



    def get_updates(self, preds, labels):
        pred = preds.argmax(-1).item()             
        _, top_5 = torch.topk(preds, k=5)
        top_5_correct = int(labels in top_5)
        top_1_correct = int(labels == pred)
        return top_1_correct, top_5_correct

    def update(self, labels, pred_logits, image_names):
        img_name2predgt = {}
        for i in range(len(image_names)):
            img_name = image_names[i]
            if img_name not in img_name2predgt:
                img_name2predgt[img_name] = [[], []]
            # print(pred_logits.shape, labels.shape, len(image_names))
            img_name2predgt[img_name][0].append(pred_logits[i])
            img_name2predgt[img_name][1].append(labels[i])
        for img_name in img_name2predgt:
            l = torch.stack(img_name2predgt[img_name][1])
            p = torch.stack(img_name2predgt[img_name][0])
            # print(l.shape, p.shape, img_name2predgt[img_name][0][0].shape)
            self.update_single(l, p, img_name)


    def update_single(self, label, pred_logits, image_name):
        # label: N, pred_scores: N x C
        # This is for all the frames in a single sound file
        pred_scores = self.softmax(pred_logits)

        assert pred_scores.shape[-1] == self.num_classes
        N = pred_scores.shape[0]


        # updating acc and top-5
        gt = label[0].item()
        for l in label: assert gt == l

        pred_gts_dict = {}

        # get window-level performances
        pred_gts_dict["window"] = []
        for i in range(N):
            pred_gts_dict["window"].append((pred_scores[i], label[i]))
            top_1, top_5 = self.get_updates(pred_scores[i], label[i])

            self.top1_sum["window"][gt] += top_1
            self.top5_sum["window"][gt] += top_5
            self.num["window"][gt] += 1

        # get average sound-level performances
        scale_pred = torch.mean(pred_scores, 0)
        scale_label = label[0]
        pred_gts_dict["sound_avg"] = [(scale_pred, scale_label)]
        top_1, top_5 = self.get_updates(scale_pred, scale_label)

        self.top1_sum["sound_avg"][gt] += top_1
        self.top5_sum["sound_avg"][gt] += top_5
        self.num["sound_avg"][gt] += 1

        assert scale_pred.shape[0] == self.num_classes
        assert scale_label in self.present_classes_idx
        filt_pred = scale_pred[self.present_classes_idx].numpy().tolist()
        filt_gt_oh = [1 if c == scale_label else 0 for c in self.present_classes_idx]
        self.all_preds.append(filt_pred)
        self.all_gts.append(filt_gt_oh)



        # get max sound-level performances
        scale_pred, _ = torch.max(pred_scores, dim=0)
        scale_label = label[0]
        pred_gts_dict["sound_max"] = [(scale_pred, scale_label)]
        top_1, top_5 = self.get_updates(scale_pred, scale_label)

        self.top1_sum["sound_max"][gt] += top_1
        self.top5_sum["sound_max"][gt] += top_5
        self.num["sound_max"][gt] += 1


        filt_pred = scale_pred[self.present_classes_idx].numpy().tolist()
        filt_gt_oh = [1 if c == scale_label else 0 for c in self.present_classes_idx]
        self.all_preds_max.append(filt_pred)
        self.all_gts_max.append(filt_gt_oh)


        # accuracies at different levels of taxonomy
        def class_id2taxa(class_id, i):
            taxa_list = self.class_id2taxa_list[class_id]
            taxa = "_".join(taxa_list[:i+1])
            return taxa

        # using average sound prediction for taxonomy-level scores
        for scale in self.metric_scales:
            for scale_pred_score, scale_gt in pred_gts_dict[scale]:
                pred = scale_pred_score.argmax(-1).item() 
                gt = scale_gt.item()
                for i, level in enumerate(self.taxa_levels):
                    gt_taxa = class_id2taxa(gt, i)
                    pred_taxa = class_id2taxa(pred, i)

                    if gt_taxa not in self.taxa_top1_sum[scale][level]:
                        self.taxa_top1_sum[scale][level][gt_taxa] = 0
                        self.taxa_num[scale][level][gt_taxa] = 0
                    self.taxa_top1_sum[scale][level][gt_taxa] += int(gt_taxa == pred_taxa)
                    self.taxa_num[scale][level][gt_taxa] += 1

        if self.save_dir is not None:
            # save the predictions and labels
            for scale in self.metric_scales:
                # if scale != "sound_avg": continue
                with open(self.save_paths[scale], "a") as f:
                    for p, g in pred_gts_dict[scale]:
                        f.write("{},{},{}\n".format(
                            image_name, 
                            g, 
                            list2str(p)
                        ))
        self.first = False

    def get_values(self):
        metric_dict = {}


        metric_dict["sound_avg"] = {}

        roc_auc = roc_auc_score(
            self.all_gts, self.all_preds, 
            average=None
        )    
        roc_auc = sum(roc_auc)/len(roc_auc)
    

        ap_per_class = average_precision_score(self.all_gts, self.all_preds, average=None)
        m_ap = sum(ap_per_class)/len(ap_per_class)

        def get_best_f1(g, p):
            precision, recall, _ = precision_recall_curve(g, p)
            f1s = 2*precision*recall/(precision+recall+1e-10)
            return f1s.max()

        f1s = [
            get_best_f1([g[c] for g in self.all_gts], [p[c] for p in self.all_preds])
            for c in range(len(self.all_gts[0]))
        ]
        macro_f1 = sum(f1s)/len(f1s)

        metric_dict["sound_avg"]["mAP"] = m_ap
        metric_dict["sound_avg"]["F1-max"] = macro_f1
        metric_dict["sound_avg"]["ROC-AUC"] = roc_auc









        metric_dict["sound_max"] = {}

        roc_auc = roc_auc_score(
            self.all_gts_max, self.all_preds_max, 
            average=None
        )
        roc_auc = sum(roc_auc)/len(roc_auc)

        ap_per_class = average_precision_score(self.all_gts_max, self.all_preds_max, average=None)
        m_ap = sum(ap_per_class)/len(ap_per_class)

        f1s = [
            get_best_f1([g[c] for g in self.all_gts_max], [p[c] for p in self.all_preds_max])
            for c in range(len(self.all_gts_max[0]))
        ]
        macro_f1 = sum(f1s)/len(f1s)


        metric_dict["sound_max"]["mAP"] = m_ap
        metric_dict["sound_max"]["F1-max"] = macro_f1
        metric_dict["sound_max"]["ROC-AUC"] = roc_auc


        for scale in self.metric_scales:
            if scale not in metric_dict:
                metric_dict[scale] = {}

            metric_dict[scale]["class-avg-top-1"] = list_average(
                [
                    self.top1_sum[scale][c] / self.num[scale][c] for c in self.all_classes if self.num[scale][c]
                ]
            )
            metric_dict[scale]["class-avg-top-5"] = list_average(
                [
                    self.top5_sum[scale][c] / self.num[scale][c] for c in self.all_classes if self.num[scale][c]
                ]
            )

            total_num = sum(self.num[scale].values())
            metric_dict[scale]["simple-top-1"] = sum(self.top1_sum[scale].values()) / total_num
            metric_dict[scale]["simple-top-5"] = sum(self.top5_sum[scale].values()) / total_num
            

            for l in self.taxa_levels:
                accs = [
                    self.taxa_top1_sum[scale][l][c] / self.taxa_num[scale][l][c] for c in self.taxa_top1_sum[scale][l]
                ]
                metric_dict[scale]["taxa_{}({})_acc".format(l, len(list(self.taxa_top1_sum[scale][l].keys())))] = sum(accs) / len(accs)
        return metric_dict

    def get_metric_str(self):
        metric_dict = self.get_values()
        s = "\n"
        for scale in metric_dict:
            s += "{}-level\n".format(scale)
            for k in metric_dict[scale]:
                if "taxa" in k: continue
                s += "|  {}: {:.2f}%\n".format(k, 100 * metric_dict[scale][k])
            s += "|  Taxonomy Levels\n"
            for k in metric_dict[scale]:
                if "taxa" not in k: continue
                s += "|  |  {}: {:.2f}%\n".format(k, 100 * metric_dict[scale][k])

        return s