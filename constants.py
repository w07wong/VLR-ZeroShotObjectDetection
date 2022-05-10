class DeviceConstants(object):
    CUDA = 'cuda'
    CPU = 'cpu'

class TrainingConstants(object):
    NUM_EPOCHS = 100
    TOTAL_SIZE = 1.0
    VAL_SIZE = 0.2
    BSZ = 4

    LOG_INTERVAL = 1  # Batches
    DEVICE = DeviceConstants.CUDA
    OUTPUT_DIR = 'nets'
    FEATURE_NET_NAME = 'feature_net'
    BOUNDING_BOX_NET_NAME = 'bb_net'
    LOG_DIR = 'logs'
    FEATURE_NET_SAVE_FNAME = 'feature_net.pth'
    BOUNDING_BOX_NET_SAVE_FNAME = 'bb_net.pth'
    NET_SAVE_FREQUENCY =1  # Epochs
    
    FEATURE_BASE_LR = 1e-3
    FEATURE_LR_STEP_SIZE = 3
    FEATURE_LR_DECAY_RATE = 0.8

    BB_BASE_LR = 1e-3
    BB_LR_STEP_SIZE = 3
    BB_LR_DECAY_RATE = 0.7

    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
