import os
import torch

class BasicStopper(object):
    def __init__(
        self, 
        patience=None,
        min_delta=0,
        ):

        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.patience = patience
        self.min_delta = min_delta
        self.val_loss_min = 100000

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

                os._exit