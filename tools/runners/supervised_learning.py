import torch 

import os.path as osp

from evaluate import evaluate
from library import RunnerRegistry

        
@RunnerRegistry.register('SupervisedLearner')
class SupervisedLearner(object):
    def __init__(self, run_by='epoch', patience=None, min_delta=0):
        
        self.run_by = run_by
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = 100000
    
    def train(self, 
             cfg,
             model, 
             device, 
             logger, 
             optimizer, 
             data_loaders,
             scheduler,
             is_dist = None):

        """
        Args: 
            runner_pack (dict): 
                includes configuration, model, data_loaders, 
                         device, logger
        """

        logger_interval = cfg['LOGGER']['interval']
        eval_interval = cfg['EVALUATION']['interval']
        checkpoint_interval = cfg['CHECKPOINT']['interval']

        best_save_path = osp.join(cfg['WORK_DIR'], "best_checkpoint.pt")
        graph_path = osp.join(cfg['WORK_DIR'], "runs")
    

        if self.run_by == 'epoch':
            iteration = cfg['EPOCH'] * len(data_loaders['train'])
            print('iteration: ', iteration)
        elif self.run_by == 'iteration':
            iteration = cfg['ITERATION']
        else:
            print('supported run by option: epoch, iteration')
       

        train_running_loss = 0.0
        val_running_loss = 0.0
        
        train_generator = iter(data_loaders['train'])
        val_generator = iter(data_loaders['val'])


        for i in range(iteration): # loop over the dataset multiple times
            
            optimizer.zero_grad()
            
            try:
                train_data = next(train_generator)

            except StopIteration: 

                train_generator = iter(data_loaders['train'])
                train_data = next(train_generator)
            

            inputs, labels = train_data['image'], train_data['segmap']
            inputs, labels = inputs.to(device), labels.to(device)

            train_loss = model(inputs, labels)
            train_loss.backward()
            train_running_loss += train_loss.item()

            optimizer.step()
            scheduler.step()


            model.eval()

            with torch.no_grad():
                try:
                    val_data = next(val_generator)

                except StopIteration:

                    val_generator = iter(data_loaders['val'])
                    val_data = next(val_generator)

                inputs, labels = val_data['image'], val_data['segmap']
                inputs, labels = inputs.to(device), labels.to(device)

                val_loss = model(inputs, labels)
                val_running_loss += val_loss.item()


            if i % logger_interval == logger_interval-1:
                logger.info(f'[Iteration: {i + 1:5d}] Train Loss: {train_running_loss / logger_interval:.3f}')
                logger.info(f'[Iteration: {i + 1:5d}] Val Loss: {val_running_loss / logger_interval:.3f}')

                if self.patience:
                    self.early_stopping(val_running_loss, model, best_save_path)

                    if self.early_stop:
                        print("Early stopping")
                        break

                train_running_loss = 0.0
                val_running_loss = 0.0


            if is_dist: 
                rank = model.device_ids[0]

                if rank == 0: 

                    if i % eval_interval == eval_interval-1:
                        evaluate(model, data_loaders['val'], device, logger = logger)  

                    if i % checkpoint_interval == checkpoint_interval-1:
                        save_path = osp.join(cfg['WORK_DIR'], f'checkpoint_iter_{i+1}.pth')
                        torch.save(model.state_dict(), save_path)

                # torch.distributed.barrier()
            else: 
                if i % eval_interval == eval_interval-1:
                    evaluate(model, data_loaders['val'], device, logger)  

                if i % checkpoint_interval == checkpoint_interval-1:
                    save_path = osp.join(cfg['WORK_DIR'], f'checkpoint_iter_{i+1}.pth')
                    torch.save(model.state_dict(), save_path)


        logger.info('Finished Training')


    def early_stopping(self, val_loss, model, path):

        """
        Early stopping to stop the training when the loss does not improve after certain epochs.

        References:
            [1] https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
            [2] https://quokkas.tistory.com/37
        """

        if self.best_loss == None:
            self.best_loss = val_loss
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
            torch.save(model.state_dict(), path)
            self.val_loss_min = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
            torch.save(model.state_dict(), path)
            self.val_loss_min = val_loss
            self.counter = 0  # reset counter if validation loss improves
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


    def eval(self,
             model, 
             device, 
             logger, 
             data_loaders,): 

        evaluate(model, data_loaders['val'], device, logger)  

