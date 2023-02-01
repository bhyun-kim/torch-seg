import torch 

import os.path as osp

from evaluate import evaluate
from tools.library import RunnerRegistry
from tools.utils import logical_xor
from tools.early_stop import BasicStopper


@RunnerRegistry.register('SupervisedLearner')
class SupervisedLearner(object):
    def __init__(
        self, 
        run_by = 'iteration',
        iteration = None,
        epochs = None,
        img_key='image',    
        label_key='segmap'    
        ):

        assert logical_xor(iteration, epochs), \
            "Choose one of iteration and epoch."
        assert run_by in ['iteration', 'epoch'], \
            'Supervised Learner only supports: epoch, iteration'
    
        self.run_by = run_by
        self.epochs = epochs
        self.iteration = iteration 
        self.img_key = img_key
        self.label_key = label_key

    
    def train(
        self,
        model, 
        device, 
        logger,  # cfg['LOGGER']['interval']
        optimizer, 
        data_loaders,
        scheduler, # cfg['EPOCH'] * len(data_loaders['train']), cfg['ITERATION']
        eval_cfg, # cfg['EVALUATION']['interval'] # cfg['WORK_DIR'], "best_checkpoint.pt")
        checkpoint_cfg,
        is_dist = False,
        early_stop=False,
        start_iter = 0,
        ):

        is_main = True

        if is_dist == True:
            rank = model.device_ids[0]

            if rank != 0:
                is_main = False

        if early_stop:
            stopper = BasicStopper()
            stop_generator = iter(data_loaders['val'])
        
        img_key = self.img_key
        label_key = self.label_key


    
        if self.run_by == 'epoch':

            iteration = self.epochs * len(data_loaders['train'])
            epochs = self.epochs
        elif self.run_by == 'iteration':
            iteration = self.iteration 
            epochs = iteration // len(data_loaders['train']) + 1
        
        train_generator = iter(data_loaders['train'])

        running_loss = 0.

        i = start_iter
        while i <= iteration:
        # for i in range(iteration): # loop over the dataset multiple times
            # print(f'iteration {i}')
            # print(f'rank {rank}')
            optimizer.zero_grad()
            
            # print(f'iteration {i} rank {rank} You are here 1')
            try:
                train_data = next(train_generator)

            except StopIteration: 

                train_generator = iter(data_loaders['train'])
                train_data = next(train_generator)
            # print(f'iteration {i} rank {rank} You are here 2')

            inputs, labels = train_data[img_key], train_data[label_key]
            inputs, labels = inputs.to(device), labels.to(device)
            # print(f'iteration {i} rank {rank} You are here 3')

            loss = model(inputs, labels)
            loss.backward()
            running_loss += loss.item()
            # print(f'iteration {i} rank {rank} You are here 4')
            optimizer.step()
            scheduler.step()

            # print(f'iteration {i} rank {rank} You are here 6')
            pred = model(inputs)

            if is_main: 
                # print(f'iteration {i} rank {rank} You are here 7')

                if i % logger.interval == logger.interval-1:
                
                    pred = torch.argmax(pred, dim=1)
                    num_correct = torch.sum(pred == labels)
                    train_acc = num_correct / len(pred)

                    epoch = i // len(data_loaders['train']) + 1

                    logger.info(
                        f'[Epoch: {epoch:3d}/{epochs}][Iteration: {i:5d}/{iteration}] Train Loss: {running_loss / logger.interval:.3f}, Train Acc: {train_acc*100:.3f}%')
                    running_loss = 0. 

                if i % eval_cfg['interval'] == eval_cfg['interval']-1:
                    # print(f'iteration {i} rank {rank} You are here 8')
                    evaluate(
                        model.module, 
                        data_loaders['val'], 
                        device, 
                        logger=logger,
                        **eval_cfg
                        )  

                if i % checkpoint_cfg['interval'] == checkpoint_cfg['interval']-1:
                    # print(f'rank {rank} You are here 9')
                    save_path = osp.join(
                        checkpoint_cfg['work_dir'],
                        f'checkpoint_iter_{i}.pth'
                        )
                    torch.save(model.state_dict(), save_path)

                if early_stop:
                    # print(f'rank {rank} You are here 10')
                    try:
                        stop_data = next(stop_generator)

                    except StopIteration: 

                        stop_generator = iter(data_loaders['val'])
                        stop_data = next(stop_generator)
                    

                    inputs, labels = stop_data[img_key], stop_data[label_key]
                    inputs, labels = inputs.to(device), labels.to(device)

                    loss = model(inputs, labels)
                    loss.backward()

                    path = osp.join(checkpoint_cfg['work_dir'], 'best_checkpoint.pth')

                    stopper.early_stopping(loss.item(), model, path) 

<<<<<<< HEAD
            # print(f'rank {rank} You are here 11')
=======


        logger.info('Finished Training')



# @RunnerRegistry.register('SupervisedLearner')
# class SupervisedLearner(object):
#     def __init__(
#         self, 
#         run_by='epoch',
#         patience=None,
#         min_delta=0
#         ):
        
#         self.run_by = run_by
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_loss = None
#         self.early_stop = False
#         self.val_loss_min = 100000
    
#     def train(
#         self, 
#         cfg,
#         model, 
#         device, 
#         logger, 
#         optimizer, 
#         data_loaders,
#         scheduler,
#         is_dist = None
#         ):

#         """
#         Args: 
#             runner_pack (dict): 
#                 includes configuration, model, data_loaders, 
#                          device, logger
#         """

#         logger_interval = cfg['LOGGER']['interval']
#         eval_interval = cfg['EVALUATION']['interval']
#         checkpoint_interval = cfg['CHECKPOINT']['interval']

#         best_save_path = osp.join(cfg['WORK_DIR'], "best_checkpoint.pt")
#         graph_path = osp.join(cfg['WORK_DIR'], "runs")
    

#         if self.run_by == 'epoch':
#             iteration = cfg['EPOCH'] * len(data_loaders['train'])
#             print('iteration: ', iteration)
#         elif self.run_by == 'iteration':
#             iteration = cfg['ITERATION']
#         else:
#             print('supported run by option: epoch, iteration')
       

#         train_running_loss = 0.0
#         val_running_loss = 0.0
        
#         train_generator = iter(data_loaders['train'])
#         val_generator = iter(data_loaders['val'])


#         for i in range(iteration): # loop over the dataset multiple times
            
#             optimizer.zero_grad()
            
#             try:
#                 train_data = next(train_generator)

#             except StopIteration: 

#                 train_generator = iter(data_loaders['train'])
#                 train_data = next(train_generator)
>>>>>>> 18d2bde51025aa3659d76706d15a01ca2a39e2e7
            

            torch.distributed.barrier()
            # print(f'rank {rank} You are here 12')

            i += 1




<<<<<<< HEAD
        logger.info('Finished Training')
=======

>>>>>>> 18d2bde51025aa3659d76706d15a01ca2a39e2e7
