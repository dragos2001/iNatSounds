import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # Need changes
    parser.add_argument('--spectrogram_dir', type=str, default="")
    parser.add_argument('--json_dir', type=str, default="")
    parser.add_argument('--geo_model_weights', type=str, default="")
    parser.add_argument('--log_dir', type=str, default="./logs")

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--optim', type=str, default="nesterov")
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--model_weight', type=str, default="")
    parser.add_argument('--encoder_weight', type=str, default="")
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--mixup', default=False, action="store_true",)
    parser.add_argument('--pretrained', default=False, action="store_true",)
    parser.add_argument('--sound_aug', default=False, action="store_true",)
    parser.add_argument('--loss', type=str, default="ce")
    parser.add_argument('--multilabel', default=False, action="store_true",)
    parser.add_argument('--geo_model', default=False, action="store_true",)
    parser.add_argument('--geo_threshold', type=float, default=0.1)
    parser.add_argument('--no_masking', default=False, action="store_true",)
    parser.add_argument('--mean_teacher', default=False, action="store_true",)
    args = parser.parse_args()
    return args
