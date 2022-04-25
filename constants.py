class DeviceConstants(object):
    CUDA = 'cuda'
    CPU = 'cpu'

class TrainingConstants(object):
    NUM_EPOCHS = 60
    TOTAL_SIZE = 1.0
    VAL_SIZE = 0.2
    BSZ = 20

    LOG_INTERVAL = 1  # Batches
    DEVICE = DeviceConstants.CUDA
    OUTPUT_DIR = 'nets'
    FEATURE_NET_NAME = 'feature_net'
    BOUNDING_BOX_NET_NAME = 'bb_net'
    LOG_DIR = 'logs'
    FEATURE_NET_SAVE_FNAME = 'feature_net.pth'
    BOUNDING_BOX_NET_SAVE_FNAME = 'bb_net.pth'
    NET_SAVE_FREQUENCY =1  # Epochs
    
    FEATURE_BASE_LR = 1e-2
    FEATURE_LR_STEP_SIZE = 3  # Epochs
    FEATURE_LR_DECAY_RATE = 0.7

    BB_BASE_LR = 5e-3
    BB_LR_STEP_SIZE = 1
    BB_LR_DECAY_RATE = 0.7

    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
