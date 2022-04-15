# VLR-ZeroShotObjectDetection

## To Train
1. Run `moments.py --data-dir=<path/to/data>` where `path/to/data` is where the images and bounding boxes are store. This script generates a moments.npz file used during training to normalize the images.
2. Run `train.py <path/to/data> ...` where `path/to/data` is where the images, bounding boxes and moments.npz file. Specify any other arguments, replacing `...` with them, which are defined in `train.py`.
3. Training will produce tensorboard logs and save the model in `nets/run_id` where run_id is some timestamp of the experiment.

## Experimentation
You can modify the feature extractor network in `feature_net.py`. The bounding box network can be found in `bb_net.py` and predicts the bounding box from the combined features of the scene and target images.

#### Download using gdown
```
 gdown https://drive.google.com/drive/folders/11QVBhEkmpfFzsDrs5Dj59f9HOijNtCnH -O <dataset-dir> --folder
```
