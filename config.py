from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.ARCH = "resnet50"
_C.MODEL.IMG_SIZE = 224
_C.MODEL.EMBEDDINGS = 512

# Train
_C.TRAIN = CN()
_C.TRAIN.OPT = "adam"
_C.TRAIN.WORKERS = 8
_C.TRAIN.LR = 0.001
_C.TRAIN.MILESTONES = [30, 40]
_C.TRAIN.MOMENTUM = 0.99
_C.TRAIN.WEIGHT_DECAY = 1e-5
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.EPOCHS = 30

# Test
_C.TEST = CN()
_C.TEST.WORKERS = 8
_C.TEST.BATCH_SIZE = 128

cfg = _C
