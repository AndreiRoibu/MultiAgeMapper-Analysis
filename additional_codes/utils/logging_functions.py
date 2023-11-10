import os
import shutil
import logging
import numpy as np
# The SummaryWriter class provides a high-level API to create an event file in a given directory and add summaries and events to it.
# More here: https://tensorboardx.readthedocs.io/en/latest/tensorboard.html
from tensorboardX import SummaryWriter

class LogWriter():

    def __init__(self, number_of_classes, logs_directory, experiment_name, use_last_checkpoint=False, 
                    ):

        self.number_of_classes = number_of_classes
        training_logs_directory = os.path.join(logs_directory, experiment_name, "train")
        validation_logs_directory = os.path.join(logs_directory, experiment_name, "validation")

        # If the logs directory exist, we clear their contents to allow new logs to be created
        if not use_last_checkpoint:
            if os.path.exists(training_logs_directory):
                shutil.rmtree(training_logs_directory)
            if os.path.exists(validation_logs_directory):
                shutil.rmtree(validation_logs_directory)
                
        self.log_writer = {
            'train': SummaryWriter(logdir=training_logs_directory),
            'validation': SummaryWriter(logdir=validation_logs_directory)
        }

        self.current_iteration = 1
        self.logger = logging.getLogger()
        file_handler = logging.FileHandler("{}/{}.log".format(os.path.join(logs_directory, experiment_name), "console_logs"))
        self.logger.addHandler(file_handler)

    def log(self, message):
        self.logger.info(msg=message)

    def loss_per_iteration(self, loss_per_iteration, batch_index, iteration):
        print("Loss for Iteration {} is: {}".format(batch_index, loss_per_iteration))
        self.log_writer['train'].add_scalar('loss/iteration', loss_per_iteration, iteration)

    def loss_per_epoch(self, losses, phase, epoch, previous_loss=None):
        loss = np.mean(losses)
        if phase == 'train':
            print("Loss for Epoch {} of {} is: {}".format(epoch, phase, loss))
        elif phase == 'train_no_noise':
            print("No Noise Loss for Epoch {} of {} is: {}".format(epoch, phase, loss))
        else:
            if previous_loss == None:
                print("Loss for Epoch {} of {} is: {}".format(epoch, phase, loss))
            else:
                print("Loss for Epoch {} of {} is {} and Absolute Change is {}".format(epoch, phase, loss, previous_loss - loss))
        self.log_writer[phase].add_scalar('loss/epoch', loss, epoch)

    def age_delta_per_epoch(self, losses, phase, epoch, previous_loss=None):
        loss = np.mean(losses)
        if phase == 'train':
            print("Age Delta for Epoch {} of {} is: {}".format(epoch, phase, loss))
        elif phase == 'train_no_noise':
            print("No Noise Age Delta for Epoch {} of {} is: {}".format(epoch, phase, loss))
        else:
            if previous_loss == None:
                print("Age Delta for Epoch {} of {} is: {}".format(epoch, phase, loss))
            else:
                print("Age Delta for Epoch {} of {} is {} and Absolute Change is {}".format(epoch, phase, loss, previous_loss - loss))
        self.log_writer[phase].add_scalar('AgeDelta/epoch', loss, epoch)

    def learning_rate_per_epoch(self, lr, phase, epoch):
         self.log_writer[phase].add_scalar('LearningRate/epoch', lr, epoch)

    def learning_rate_per_iteration(self, lr, batch_index, iteration):
        self.log_writer['train'].add_scalar('LearningRate/iteration', lr, iteration)

    def close(self):
        self.log_writer['train'].close()
        self.log_writer['validation'].close()