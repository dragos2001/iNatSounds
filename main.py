import src
from src import dataset, models, utils, train_eval, params

import torch
import logging
import numpy as np
import random
import tqdm
import os
import copy

def run_train(args):
    weight_dir, plot_dir, pred_dir = utils.setup_logging(args, mode="train")

    use_cuda = torch.cuda.is_available()
    
    train_dataloader, val_dataloader, test_dataloader = dataset.get_dataloaders(args)
    num_classes = train_dataloader.num_classes
    
    print("Number of classes: " + str(num_classes))
    logging.info("Number of classes: " + str(num_classes))

    output_dim = num_classes 
    model = models.get_model(args.model, output_dim=output_dim, pretrained=args.pretrained)
    # model = nn.DataParallel(model)
    assert args.model_weight == "" or args.encoder_weight == ""
    if args.model_weight != "":
        weights = torch.load(args.model_weight)
        model.load_state_dict(weights)
    if args.encoder_weight != "":
        weights = torch.load(args.encoder_weight)
        weights = {k:v for k, v in weights.items() if not ("fc" in k or "heads" in k)}
        model.load_state_dict(weights, strict=False)
    model.train()
    if use_cuda:
        model = model.cuda()

    geo_model = models.GeoModel(
        geo_model_weights=args.geo_model_weights,
        json_dir=args.json_dir,
    )
    if use_cuda:
        geo_model = geo_model.cuda()


    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == "sgd_momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    elif args.optim == "nesterov":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True)
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise NotImplementedError


    BASE_EPOCHS = args.epochs
    # Warmup for 5 epochs and then cosine decay for args.epoch//2 - 5 epochs and then exponential decay rest
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=BASE_EPOCHS - 5, eta_min=0.1*args.lr)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[5])

    
    logging.info(str(model))
    logging.info(str(args))

    epoch2loss = {"train": {}, "val": {}}
    epoch2acc = {"train": {}, "val": {}}
    epoch2lr = {"train": {}}
    epoch_pbar = tqdm.tqdm(total=args.epochs, desc="epoch")
    LOG_FMT = "Epoch {:3d} | {} set | LR {:.4E} | Loss {:.4f} | Acc {:.2f}"
    val_acc = -1
    best_val_metrics = None
    best_weights = None
    best_epoch = None
    
    for epoch in range(args.epochs):

        cur_lr = lr_scheduler.get_last_lr()[-1]
        epoch2lr["train"][epoch] = cur_lr
        ## Train set
        train_loss, train_acc = train_eval.run_loop(
            args, train_dataloader, model, 
            mode="train", optimizer=optimizer, 
            use_cuda=use_cuda, 
            epoch=epoch,
            geo_model=geo_model,
        )
        logging.info(LOG_FMT.format(
            epoch, "train", cur_lr, 
            train_loss, 100*train_acc, 
        ))
        epoch2loss["train"][epoch] = train_loss
        epoch2acc["train"][epoch] = train_acc
        
        
        if (epoch+1)%args.eval_freq == 0:
            utils.save_model(model, os.path.join(weight_dir, "checkpoint_{}.pt".format(epoch)))
            
            ## Val set
            val_loss, val_acc, val_metrics = train_eval.run_loop(
                args, val_dataloader, model, 
                mode="eval",
                use_cuda=use_cuda,
                geo_model=geo_model,
                # save_dir=os.path.join(pred_dir, "val", "epoch_{}".format(epoch))
            )
            logging.info(LOG_FMT.format(
                epoch, "val", cur_lr, 
                val_loss, 100*val_acc, 
            ))
            logging.info(val_metrics)
            epoch2loss["val"][epoch] = val_loss
            epoch2acc["val"][epoch] = val_acc

            if best_val_metrics is None or utils.metric_str2acc(best_val_metrics) < utils.metric_str2acc(val_metrics):
                best_val_metrics = val_metrics
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch

            
        
        if epoch > BASE_EPOCHS:
            lr_scheduler = scheduler3
        
        lr_scheduler.step()

        epoch_pbar.update(1)
        epoch_pbar.set_description("Epochs | LR: {:.4E} Loss: {:.4f} Train Acc: {:.4f} Val Acc: {:.4f}".format(cur_lr, train_loss, train_acc, val_acc))


        ## Plots
        utils.save_plots(epoch2loss, os.path.join(plot_dir, "loss.png"))
        utils.save_plots(epoch2acc, os.path.join(plot_dir, "acc.png"))
        utils.save_plots(epoch2lr, os.path.join(plot_dir, "lr.png"))
    
    logging.info("Best Val: Epoch {}, Best metrics".format(best_epoch))
    logging.info(best_val_metrics)
    model.load_state_dict(best_weights)
    ## Test set
    test_loss, test_acc, test_metrics = train_eval.run_loop(
        args, test_dataloader, model, 
        mode="eval",
        use_cuda=use_cuda,
        geo_model=geo_model,
        # save_dir=os.path.join(pred_dir, "test", "epoch_{}".format(epoch))
    )
    logging.info(LOG_FMT.format(
        best_epoch, "test", cur_lr, 
        test_loss, 100*test_acc, 
    ))
    logging.info(test_metrics)


    ### With test-time geo-filtering
    ## Val set
    logging.info("With test-time geo-filtering!")
    val_loss, val_acc, val_metrics = train_eval.run_loop(
        args, val_dataloader, model, 
        mode="eval",
        use_cuda=use_cuda,
        geo_model=geo_model,
        test_geo_mask=True,
        # save_dir=os.path.join(pred_dir, "val", "epoch_{}".format(epoch))
    )
    logging.info(LOG_FMT.format(
        best_epoch, "val", cur_lr, 
        val_loss, 100*val_acc, 
    ))
    logging.info(val_metrics)

    ## Test set
    logging.info("With test-time geo-filtering!")
    test_loss, test_acc, test_metrics = train_eval.run_loop(
        args, test_dataloader, model, 
        mode="eval",
        use_cuda=use_cuda,
        geo_model=geo_model,
        test_geo_mask=True,
        # save_dir=os.path.join(pred_dir, "test", "epoch_{}".format(epoch))
    )
    logging.info(LOG_FMT.format(
        best_epoch, "test", cur_lr, 
        test_loss, 100*test_acc, 
    ))
    logging.info(test_metrics)



def run_eval(args):
    weight_dir, plot_dir, pred_dir = utils.setup_logging(args, mode="eval")

    use_cuda = torch.cuda.is_available()
    _, val_dataloader, test_dataloader = dataset.get_dataloaders(args)
    num_classes = val_dataloader.num_classes
    output_dim = num_classes 
    model = models.get_model(args.model, output_dim=output_dim, pretrained=args.pretrained)
    if args.model_weight != "":
        weights = torch.load(args.model_weight)
        model.load_state_dict(weights)
    if args.encoder_weight != "":
        weights = torch.load(args.encoder_weight)
        model.load_state_dict(weights, strict=True)

    model.eval()
    if use_cuda:
        model = model.cuda()

    geo_model = models.GeoModel(
        geo_model_weights=args.geo_model_weights,
        json_dir=args.json_dir,
    )
    if use_cuda:
        geo_model = geo_model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()

    
    logging.info(str(model))
    logging.info(str(args))

    LOG_FMT = "{} set | Loss {:.4f} | Acc {:.2f}"
        
    val_loss, val_acc, val_sound_metrics = train_eval.run_loop(
        args, val_dataloader, model, 
        mode="eval",
        use_cuda=use_cuda,
        # save_dir=os.path.join(pred_dir, "val"),
        geo_model=geo_model,
    )
    logging.info(LOG_FMT.format(
        "val", 
        val_loss, 100*val_acc, 
    ))
    logging.info(val_sound_metrics)
    print(val_sound_metrics)

    test_loss, test_acc, test_sound_metrics = train_eval.run_loop(
        args, test_dataloader, model, 
        mode="eval",
        use_cuda=use_cuda,
        # save_dir=os.path.join(pred_dir, "test"),
        geo_model=geo_model,
    )
    logging.info(LOG_FMT.format(
        "test", 
        test_loss, 100*test_acc, 
    ))
    logging.info(test_sound_metrics)
    print(test_sound_metrics)
        

    ## With test-time geo-filtering
    logging.info("With test-time geo-filtering!")
    val_loss, val_acc, val_sound_metrics = train_eval.run_loop(
        args, val_dataloader, model, 
        mode="eval",
        use_cuda=use_cuda,
        # save_dir=os.path.join(pred_dir, "geo_val"),
        geo_model=geo_model,
        test_geo_mask=True,
    )
    logging.info(LOG_FMT.format(
        "val", 
        val_loss, 100*val_acc, 
    ))
    logging.info(val_sound_metrics)
    print(val_sound_metrics)

    logging.info("With test-time geo-filtering!")
    test_loss, test_acc, test_sound_metrics = train_eval.run_loop(
        args, test_dataloader, model, 
        mode="eval",
        use_cuda=use_cuda,
        # save_dir=os.path.join(pred_dir, "geo_test"),
        geo_model=geo_model,
        test_geo_mask=True,
    )
    logging.info(LOG_FMT.format(
        "test", 
        test_loss, 100*test_acc, 
    ))
    logging.info(test_sound_metrics)
    print(test_sound_metrics)
    

if __name__=="__main__":
    args = params.get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.mode == "train":
        run_train(args)
    else:
        run_eval(args)