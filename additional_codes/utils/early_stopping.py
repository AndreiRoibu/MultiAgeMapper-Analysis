import numpy as np

class EarlyStopping:

    def __init__(self, patience=5, min_delta=0.0, best_score=None, counter=0):

        self.patience = patience
        self.counter = counter
        self.best_score = best_score
        self.early_stop = False
        self.min_delta = min_delta

    def __call__(self, validation_loss): 

        score = validation_loss

        if self.best_score is None:
            self.best_score = score
        
        elif np.greater_equal(self.min_delta, self.best_score - score):
            self.counter += 1
            print("Early Stopping Counter: {}/{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop, self.best_score, self.counter