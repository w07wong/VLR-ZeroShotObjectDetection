# VLR-ZeroShotObjectDetection
##[Project Website](https://sites.google.com/andrew.cmu.edu/zeroshotobjectdetection/home?authuser=1)

## Approaches
Branches `approach_0`, `approach_1`, and `approach_2` contain the code to the corresponding approach described on the project website. The SIFT feature matching baseline can be found under `sift` and can be run with `eval_sift.py`.

## Process Dataset

Run from the repo's root directory, `python process_data.py <path/to/scene/data> <full/path/to/target/images> <path/to/data>` where `<path/to/scene/data>` is the YCB data directory, `<full/path/to/target/images>` is the generated target image directory, and `<path/to/data>` is the processed dataset that stores the images and bounding boxes.

Each scene image corresponds to 

1. scene image (e.g. `000000_scene.png`)
2. bounding box file (e.g. `000000.npy`)
3. ten target images (e.g. `000000_target_0000.png` - `000000_target_0009.png`)

`<path/to/data>/class.npy` contains the class label for each scene image. 

## To Train
1. Run `moments.py --data-dir=<path/to/data>` where `path/to/data` is where the images and bounding boxes are store. This script generates a moments.npz file used during training to normalize the images.
2. Run `train.py <path/to/data> ...` where `path/to/data` is where the images, bounding boxes and moments.npz file. Specify any other arguments, replacing `...` with them, which are defined in `train.py`. You can also specify other arguments in `constants.py` if you don't want to deal with command line arguments.
3. Training will produce tensorboard logs and save the model in `nets/run_id` where run_id is some timestamp of the experiment.

## Experimentation
You can modify the feature extractor network in `feature_net.py`. The bounding box network can be found in `bb_net.py` and predicts the bounding box from the combined features of the scene and target images.


## Data generation (target images)

```
python data_gen.py --model-dir ../YCB_Video_Models/models/002_master_chef_can --output-dir ../YCB_Video_Models/models/002_master_chef_can/target/ --image-size <width> <height>
```

## Download small dataset using gdown
```
 gdown https://drive.google.com/drive/folders/11QVBhEkmpfFzsDrs5Dj59f9HOijNtCnH -O <dataset-dir> --folder
```
