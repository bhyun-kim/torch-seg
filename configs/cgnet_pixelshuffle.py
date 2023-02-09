ROOT_DIR = '/home/user/server/05. Data/00. Benchmarks/01. cityscapes'

LOSS = dict(
    type='CrossEntropyLoss',
    weight=[
            2.5959933, 6.7415504, 3.5354059, 9.8663225, 9.690899, 9.369352,
            10.289121, 9.953208, 4.3097677, 9.490387, 7.674431, 9.396905,
            10.347791, 6.3927646, 10.226669, 10.241062, 10.280587,
            10.396974, 10.055647
            ]  
) 

MODEL = dict(
    encoder = dict(type='CGNet'),
    decoder = None,
    head = dict(
        type='PixelShuffle', 
        loss=LOSS,
        in_channels=256,
        num_classes=19,
        pixelshuffle_factor=2,
        scale_factor=8,
        mode='bilinear'
    )
)

CROP_SIZE = (512, 1024)
BATCH_SIZE = 8
MEAN = [72.39239876, 82.90891754, 73.15835921]

STD = [1, 1, 1]

TRAIN_PIPELINES =[
    dict(type='RandomRescale', output_range=(512, 2048)),
    dict(type='RandomCrop', output_size=CROP_SIZE),
    dict(type='RandomFlipLR'),
    dict(type='Normalization', mean=MEAN, std=STD),
    dict(type='ImgToTensor'),
    dict(type='SegToTensor')
]

VAL_PIPELINES = [
    dict(type='Rescale', output_size=CROP_SIZE),
    dict(type='Normalization', mean=MEAN, std=STD),
    dict(type='ImgToTensor'),
    dict(type='SegToTensor')
]

TEST_PIPELINES = [
    dict(type='Rescale', output_size=CROP_SIZE),
    dict(type='Normalization', mean=MEAN, std=STD),
    dict(type='ImgToTensor'),
    dict(type='SegToTensor')
]

DATA_LOADERS = dict(
    train=dict(
        dataset=dict(
            type='CityscapesDataset', 
            root=ROOT_DIR,
            split='train'), 
        pipelines=TRAIN_PIPELINES,
        loader=dict(
            shuffle=False,
            batch_size=BATCH_SIZE,
        )
    ),
    val = dict(
        dataset=dict(
            type='CityscapesDataset', 
            root=ROOT_DIR,
            split='val'), 
        pipelines=VAL_PIPELINES,
        loader=dict(
            shuffle=False,
            batch_size=1,
        )
    ),
    test=dict(
        dataset=dict(
            type='CityscapesDataset', 
            root=ROOT_DIR,
            split='test'), 
        pipelines=TEST_PIPELINES,
        loader=dict(
            shuffle=False,
            batch_size=1,
        )
    )
)
PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                [0, 80, 100], [0, 0, 230], [119, 11, 32]]

ITERATION = 60000
LR_CONFIG = dict(type='PolynomialLR', total_iters=ITERATION, power=0.9)
OPTIMIZER = dict(type='Adam', lr=0.001, eps=1e-08, weight_decay=0.0005)
RUNNER = dict(
    type='SupervisedLearner', 
    run_by='iteration',
    iteration=ITERATION, 
)

EVALUATION = dict(interval=10, metric='mIoU')
CHECKPOINT = dict(interval=10 )
LOGGER = dict(interval=1)
WORK_DIR = '/home/user/server/05. Data/03. Checkpoints/2023.02.09 CGNET_pixelshuffle'