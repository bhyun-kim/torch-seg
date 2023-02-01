ROOT_DIR = '/home/user/server/05. Data/00. Benchmarks/04. ImageNet2012'

LOSS = dict(
    type='CrossEntropyLoss',
    ignore_idx=-1,
) 

MODEL = dict(
    encoder = dict(type='resnet18'),
    decoder = None,
    head = dict(
        type='Classify', 
        loss=LOSS,
        num_classes=1000,
        in_channels=512,
    )
)

CROP_SIZE = (224, 224)
BATCH_SIZE = 256
MEAN = [123.675, 116.28, 103.53]

STD = [58.395, 57.12, 57.375]

TRAIN_PIPELINES =[
    dict(type='RandomRescale', output_range=(192, 256)),
    dict(type='RandomCrop', output_size=CROP_SIZE),
    dict(type='RandomFlipLR'),
    dict(type='Normalization', mean=MEAN, std=STD),
    dict(type='ImgToTensor')
]

VAL_PIPELINES = [
    dict(type='Rescale', output_size=CROP_SIZE),
    dict(type='Normalization', mean=MEAN, std=STD),
    dict(type='ImgToTensor')
]


DATA_LOADERS = dict(
    train=dict(
        dataset=dict(
            type='ImageNet', 
            root=ROOT_DIR,
            split='train'
            ), 
        pipelines=TRAIN_PIPELINES,
        loader=dict(
            shuffle=False,
            batch_size=BATCH_SIZE,
        )
    ),
    val = dict(
        dataset=dict(
            type='ImageNet', 
            root=ROOT_DIR,
            split='val'), 
        pipelines=VAL_PIPELINES,
        loader=dict(
            shuffle=False,
            batch_size=BATCH_SIZE,
        )
    )
)

ITERATION = 75000
LR_CONFIG = dict(type='PolynomialLR', total_iters=ITERATION, power=0.9)
OPTIMIZER = dict(type='Adam', lr=0.001, eps=1e-08, weight_decay=0.0005)
RUNNER = dict(
    type='SupervisedLearner', 
    run_by='iteration', 
    iteration=ITERATION,
    label_key='target'
)

LOAD_FROM = None
RESUME_FROM = None
EVALUATION = dict(interval=3000, metric='acc', label_key='target')
LOGGER = dict(interval=1)
WORK_DIR = '/home/user/#temp/imagenet'
CHECKPOINT = dict(interval=3000, work_dir=WORK_DIR)