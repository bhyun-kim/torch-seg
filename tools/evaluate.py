import torch 
import numpy as np

from tqdm import tqdm
from prettytable import PrettyTable


def evaluate(
    model,
    dataloader, 
    device, 
    metric='miou',
    logger=None, 
    ignore_idx=255.,
    img_key = 'image',
    label_key = 'segmap',
    interval = None, 
    ):
    """evaluate dataset
    Args: 
        model (torch.nn.Module)
        dataloader (torch.DataLoader)
        device (torch.device)
        metric (str): performance metric, default='miou'
        logger (logging.logger)
        ignore_idx (idx)
    
    """
    # no gradients needed
    supported_metrices = ['miou', 'acc']
    model.eval()

    if hasattr(dataloader.dataset, 'datasets'):
        # if dataset is concat, bring the num of classes of the first
        classes = dataloader.dataset.datasets[0].classes 
        num_classes = len(classes)
    else: 
        classes = dataloader.dataset.classes
        num_classes = len(classes)
    
    with torch.no_grad():
        preds = []
        gts = []
        i = 0
        # iter_dataloader = iter(dataloader)
        # while i < 5:
        #     i+= 1
        #     data = next(iter_dataloader)

        for data in tqdm(dataloader, desc='Evaluation:'):
            # print("you are evaluating")

            inputs, labels = data[img_key], data[label_key]
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            pred = torch.argmax(outputs, dim=1)

            pred = pred.flatten().detach().cpu().numpy()
            gt = labels.flatten().detach().cpu().numpy()

            preds.append(pred)
            gts.append(gt)
    
        if metric=='miou':
            cm = calculate_cm(preds, gts, num_classes, ignore_idx)
            calculate_miou(cm, logger=logger, classes=classes) 

        elif metric=='acc':
            calculate_acc(preds, gts)
        else: 
            print(f'The provided metric {metric} is not supported.')
            print(f'Choose one of {supported_metrices}')

    # gradients needed for training
    model.train()


def calculate_cm(preds, gts, num_classes, ignore_idx):
    """calculate confusion matrix
    Args:
        preds (list): list of predictions (np.array)
        gts (list): list of ground truths (np.array)
        num_classes (int)
        ignore_idx (int)

    Returns: 
        cm (np.array): confusion matrix of which shape is (num_classes x num_classes)

    """
    category_vectors = []
    for pred, gt in zip(preds, gts):
        
        pred = pred[gt != ignore_idx]
        gt = gt[gt != ignore_idx]
        
        category_vector = gt * num_classes + pred

        category_vectors.append(category_vector)

    category_vectors = np.concatenate(category_vectors)
    category_bincount = np.bincount(category_vectors, minlength = num_classes ** 2)

    cm = category_bincount.reshape(num_classes, num_classes)

    return cm

def calculate_miou(cm, logger=None, classes=None):
    """calculate miou
    Args: 
        cm (np.array): confusion matrix of which shape is (num_classes x num_classes)
        logger (logger): default = None, If logger is None, print mIoU on terminal 
    """
    assert cm.shape[0] == cm.shape[1], "The number of columns and rows of confusion matrix should be same."

    num_classes = cm.shape[0]

    if classes == None: 
        classes = [f'class_{i}' for i in range(num_classes)]

    ious = []
    for i in range(num_classes): 
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - intersection
        ious.append(intersection / union)
    
    miou = np.mean(ious)

    iou_table = PrettyTable(['Class', 'IoU (%)'])
    
    for idx, iou in enumerate(ious):
        iou_table.add_row([classes[idx], f'{iou*100:.2f}'])

    miou_table = PrettyTable(['Metric', '(%)' ])
    miou_table.add_row(['mIoU', f'{miou*100:.2f}'])


    if logger: 
        logger.info('\n'+iou_table.get_string())
        logger.info('\n'+miou_table.get_string())
    else: 
        print('\n'+iou_table.get_string())
        print('\n'+miou_table.get_string())
        


def calculate_acc(preds, gts, logger=None):
    
    pred_arr = np.asarray(preds)
    gt_arr = np.asarray(gts)        

    num_correct = np.sum(pred_arr == gt_arr)
    val_acc = num_correct / len(preds)

    eval_result_str = f'Evaluation Acc: {val_acc*100:.3f}'
    if logger: 
        logger.info(eval_result_str)
    else: 
        print(eval_result_str)
        

