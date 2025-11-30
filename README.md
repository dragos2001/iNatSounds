# iNatSounds

[[Paper](https://openreview.net/forum?id=QCY01LvyKm)] [[Data](https://github.com/visipedia/inat_sounds)] [[Website](https://cvl-umass.github.io/iNatSounds/)] [[Model Weights](https://drive.google.com/drive/folders/1u8iqzP2WL2nkTMp9VZ5FCa3zveOrPj2X?usp=sharing)] 

## Sample Commands

### Training

```
python3 main.py --model mobilenet \
    --spectrogram_dir <> \          # Root for saved spectrogram npy
    --json_dir <> \                 # Root for dataset jsons 
    --geo_model_weights <> \        # Weights released
    --mixup                         # Whether to use mixup or not
```

Model hyperparameters and more instructions coming soon!