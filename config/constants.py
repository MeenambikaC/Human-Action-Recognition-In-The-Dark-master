from pathlib import Path
import os


DATA_DIR  = Path('EE6222_data')

MODEL_NAME = 'resnet18' # For CNN-RNN only
MODEL_TYPE = 'r2plus1d_18' # 'cnn-rnn', 'r3d_18', 'r2plus1d_18'
IMG_SHAPE = 'BCTHW' #'BCTHW', 'BTCHW'
#SAVED_WEIGHTS_DIR = '/kaggle/input/openpose-r21d-model-state-dict-221027/best_val_loss (1).pt'
PRETRAINED = Path('model/state_dict')
CHECKPOINT_DIR = PRETRAINED / 'checkpoint'
MODEL_STATE_DIR = PRETRAINED / 'bestloss'
OPENPOSE_STATE_DIR = PRETRAINED / 'openpose/body_pose_model.pth'
# D:\Fifth Semester\DSE Project\Human-Action-Recognition-In-The-Dark-master\model\state_dict\bestloss
SUBMISSION_DIR = Path('Submission')

TRAIN_VID_FOLDER = 'train_20'
TRAIN_IMG_FOLDER = 'train_20_img'
TRAIN_VID_DIR = DATA_DIR / TRAIN_VID_FOLDER
TRAIN_IMG_DIR = DATA_DIR / TRAIN_IMG_FOLDER
TRAIN_CSV = DATA_DIR / 'train_20.txt'

VAL_VID_FOLDER = 'validate_20'
VAL_IMG_FOLDER = 'validate_20_img'
VAL_VID_DIR = DATA_DIR / VAL_VID_FOLDER
VAL_IMG_DIR = DATA_DIR / VAL_IMG_FOLDER
VAL_CSV = DATA_DIR / 'validate_20.txt'

INF_VID_FOLDER = 'test_20'
INF_IMG_FOLDER = 'test_20_img'
INF_VID_DIR = DATA_DIR / INF_VID_FOLDER
INF_IMG_DIR = DATA_DIR / INF_IMG_FOLDER
INF_CSV = DATA_DIR / 'test_20.txt'

MAP_TABLE_DIR = DATA_DIR / 'mapping_table.txt'
NUM_CLASSES = 11

EXTENSION = ".mp4"
N_FRAMES = 16
INTERVAL = 5

SPLITS = 5

IMG_DIM = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB

DROPOUT = 0.3
RNN_HIDDEN_SIZE = 128
RNN_NUM_LAYERS = 1
ACCUM_ITER = 8

TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 1 * TRAIN_BATCH_SIZE
TEST_BATCH_SIZE = VAL_BATCH_SIZE

LR = 1e-4 # 1e-3
NUM_EPOCHS = 2#20
NUM_WARMUP_EPOCHS = 1#5
NUM_COS_CYCLE = 0.4 # 0.5

TOP_K = 5





