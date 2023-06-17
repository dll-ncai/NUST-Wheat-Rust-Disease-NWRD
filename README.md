# NUST Wheat Rust Disease (NWRD): Semantic Segmentation using Suppervised Deep Learning
Semantic segmentation of wheat yellow/stripe rust disease images to segment out rust and non-rust pixels using supervised deep learning.

This repo contains the source code for the study presented in the work:
`The NWRD Dataset: An Open-Source Annotated Segmentation Dataset of Diseased Wheat Crop`.

## Dataset
The NWRD dataset is a real-world segmentation dataset of wheat rust diseased and healthy leaf images specifically constructed for semantic segmentation of wheat rust disease.
The NWRD dataset consists of 100 images in total at this moment.

Sample images from The NWRD dataset; annotated images showing rust disease along with their binary masks:
![Sample images from The NWRD dataset; annotated images showing rust disease along with their binary masks](https://github.com/saadulkh/nwrd/assets/38633812/c8677336-82a0-4637-a3f8-61f5cedbad37)


Dataset is available at: https://dll.seecs.nust.edu.pk/downloads/

### Directory Structure
The NWRD dataset images are available in `.jpg` format and the annotated binary masks are available in `.png` format. Below is the directory structure of this dataset:
```
NWRD
├── test
│   ├── images
│   └── masks
└── train
    ├── images
    └── masks
```

### Data Splits
Here are the data splits of the NWRD dataset:
|   Split   |   Percentage  |
|-----------|-----------|
|   Train + Valid   |   90      |
|   Test    |   10      |
|   Total   |   100     |

The experimentation with 22 images was done with the following set of images:
|   Split   |   Images  |
|-----------|-----------|
|   Train + Valid   | 2, 7, 14, 30, 64, 83, 84, 90, 94, 95, 100, 118, 124, 125, 132, 133, 136, 137, 138, 146 |
|   Test    | 67, 123 |

## Requirements
* Python
* Pytorch
* Torchvision
* Numpy
* Tqdm
* Tensorboard
* Scikit-learn
* Pandas

## Usage
Different aspects of the training can be tunned from the `config.json` file.
Use `python train.py -c config.json` to run code.

### Config file format
Config files are in `.json` format:
```javascript
{
    "name": "WRS",                          // training session name
    "n_gpu": 1,                             // number of GPUs to use for training.

    "arch": {
        "type": "UNet",                     // name of model architecture to train
        "args": {
            "n_channels": 3,
            "n_classes": 2
        }                                   // pass arguments to the model
    },
    "data_loader": {
        "type": "PatchedDataLoader",        // selecting data loader
        "args":{
            "data_dir": "data/",            // dataset path
            "patch_size": 128,              // patch size
            "batch_size": 64,               // batch size
            "patch_stride": 32,             // patch overlapping stride
            "target_dist": 0.01,            // least percentage of rust pixels in a patch
            "shuffle": true,                // shuffle training data before
            "validation_split": 0.1,        // size of validation dataset. float(portion) or int(number of samples)
            "num_workers": 2                // number of cpu processes to be used for data loading
        }
    },
    "optimizer": {
        "type": "RMSprop",
        "args":{
            "lr": 1e-6,                     // learning rate
            "weight_decay": 0
        }
    },
    "loss": "focal_loss",                   // loss function
    "metrics": [                            // list of metrics to evaluate
        "precision",
        "recall",
        "f1_score"
    ],
    "lr_scheduler": {
        "type": "ExponentialLR",            // learning rate scheduler
        "args": {
            "gamma": 0.998
        }
    },
    "trainer": {
        "epochs": 500,                      // number of training epochs
        "adaptive_step": 5,                 // update dataset after every adaptive_step epochs

        "save_dir": "saved/",
        "save_period": 1,                   // save checkpoints every save_period epochs
        "verbosity": 2,                     // 0: quiet, 1: per epoch, 2: full
        
        "monitor": "min val_loss",          // mode and metric for model performance monitoring. set 'off' to disable.
        "early_stop": 50,                   // early stop

        "tensorboard": true                 // enable tensorboard visualization
    }
}
```

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --device 2,3 -c config.json
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```
## License
This project is licensed under the MIT License. See  [LICENSE](LICENSE) for more details

## Acknowledgements
* This research project was funded by German Academic Exchange Service (DAAD).
* This project follows the template provided by [victoresque](https://github.com/victoresque/pytorch-template)
