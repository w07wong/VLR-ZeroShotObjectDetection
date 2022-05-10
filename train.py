import argparse
from dataset import Dataset
from feature_net import FeatureNet
from bb_net import BoundingBoxNet
from trainer import Trainer
from constants import TrainingConstants, DeviceConstants


if __name__ == '__main__':
    # Parse args.
    parser = argparse.ArgumentParser(description='Train a network.')
    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to the training data.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=TrainingConstants.NUM_EPOCHS,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--total_size',
        type=float,
        default=TrainingConstants.TOTAL_SIZE,
        help='The proportion of the data to use.'
    )
    parser.add_argument(
        '--val_size',
        type=float,
        default=TrainingConstants.VAL_SIZE,
        help='The proportion of the data to use for validation.'
    )
    parser.add_argument(
        '--bsz',
        type=int,
        default=TrainingConstants.BSZ,
        help='Training batch size.'
    )
    parser.add_argument(
        '--feature_base_lr',
        type=float,
        default=TrainingConstants.FEATURE_BASE_LR,
        help='Feature net base learning rate.'
    )
    parser.add_argument(
        '--feature_lr_step_size',
        type=int,
        default=TrainingConstants.FEATURE_LR_STEP_SIZE,
        help='Step size for feature net learning rate in epochs.'
    )
    parser.add_argument(
        '--feature_lr_decay_rate',
        type=float,
        default=TrainingConstants.FEATURE_LR_DECAY_RATE,
        help='Decay rate for feature net learning rate at every --lr_step_size.'
    )
    parser.add_argument(
        '--bb_base_lr',
        type=float,
        default=TrainingConstants.BB_BASE_LR,
        help='Bounding box net base learning rate.'
    )
    parser.add_argument(
        '--bb_lr_step_size',
        type=int,
        default=TrainingConstants.BB_LR_STEP_SIZE,
        help='Step size for bounding box net learning rate in epochs.'
    )
    parser.add_argument(
        '--bb_lr_decay_rate',
        type=float,
        default=TrainingConstants.BB_LR_DECAY_RATE,
        help='Decay rate for bounding box net learning rate at every --lr_step_size.'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=TrainingConstants.LOG_INTERVAL,
        help='Log interval in batches.'
    )
    parser.add_argument(
        '--cuda',
        action='store_true',
        help='Enable CUDA support and utilize GPU devices.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=TrainingConstants.OUTPUT_DIR,
        help='Directory to output logs and trained model to.'
    )

    args = parser.parse_args()
    data_dir = args.data_dir
    num_epochs = args.num_epochs
    total_size = args.total_size
    val_size = args.val_size
    bsz = args.bsz

    feature_base_lr = args.feature_base_lr
    feature_lr_step_size = args.feature_lr_step_size
    feature_lr_decay_rate = args.feature_lr_decay_rate
    
    bb_base_lr = args.bb_base_lr
    bb_lr_step_size = args.bb_lr_step_size
    bb_lr_decay_rate = args.bb_lr_decay_rate

    log_interval = args.log_interval
    output_dir = args.output_dir

    cuda = args.cuda
    if cuda:
        device = DeviceConstants.CUDA
    else:
        device = DeviceConstants.CPU

    dataset = Dataset(data_dir, is_train=True)

    #feature_net = FeatureNet(dataset.scene_mean, dataset.scene_std, dataset.target_mean, dataset.target_std, name=TrainingConstants.FEATURE_NET_NAME)
    feature_net = FeatureNet(name=TrainingConstants.FEATURE_NET_NAME)
    bb_net = BoundingBoxNet(name=TrainingConstants.BOUNDING_BOX_NET_NAME)

    trainer = Trainer(feature_net,
                      bb_net,
                      dataset,
                      num_epochs,
                      total_size,
                      val_size,
                      bsz,
                      feature_base_lr,
                      feature_lr_step_size,
                      feature_lr_decay_rate,
                      bb_base_lr,
                      bb_lr_step_size,
                      bb_lr_decay_rate,
                      log_interval,
                      device,
                      output_dir)

    trainer.train()
