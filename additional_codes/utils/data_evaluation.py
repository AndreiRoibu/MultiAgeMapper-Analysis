import os
import numpy as np
import torch
import torch.nn as nn
import logging
from utils.misc import create_folder
import pandas as pd
from collections import OrderedDict

log = logging.getLogger(__name__)
MSELoss = nn.MSELoss()

def evaluate_data(
                        model,
                        test_loader,
                        volumes_to_be_used, 
                        prediction_output_statistics_name, 
                        trained_model_path,
                        device,
                        prediction_output_path,
                        control,
                        fused_data_flag,
                        dataset_sex,
                        len_test_data,
                    ):

    trained_model = torch.load(trained_model_path,map_location=torch.device(device))

    if hasattr(trained_model, 'state_dict'):
        model_sd = trained_model.state_dict()
        if torch.cuda.device_count()>1 or device=='cpu':
            correct_state_dict = {}
            for key in model_sd.keys():
                if key.startswith('module.'):
                    new_key = key.replace('module.', "")
                    correct_state_dict[new_key] = model_sd[key]
                else:
                    correct_state_dict[key] = model_sd[key]
            correct_state_dict = OrderedDict(correct_state_dict)
            del model_sd
        model.load_state_dict(correct_state_dict)
    else:
        model.load_state_dict(trained_model)

    del trained_model

    if torch.cuda.is_available() == True and device!='cpu':
        model.cuda(device)
        cuda_available = torch.cuda.is_available()
    else:
        cuda_available = 'False'

    model.eval()

    # Create the prediction path folder if this is not available

    create_folder(prediction_output_path)

    # Initiate the evaluation

    log.info("Evaluation Started")

    if control == 'mean':
        prediction_output_statistics_name = "output_statistics_mean_target.csv"
    elif control == 'null':
        prediction_output_statistics_name = "output_statistics_null_target.csv"

    output_statistics = {}
    output_statistics_path = os.path.join(prediction_output_path, prediction_output_statistics_name)

    with torch.no_grad():

        for batch_index, sampled_batch in enumerate(test_loader):
            X = sampled_batch[0]
            y_age = sampled_batch[1].type(torch.FloatTensor)
            y_age = (y_age.cpu().numpy()).astype('float32')
            y_age = np.squeeze(y_age)
            subject = volumes_to_be_used[batch_index]

            number_of_modalities = len(X)

            # First, torch unsqueeze all dictionary values
            # Here, insert flag - if input_fusion flag, then concatenate all and proceed as single value
            # In the network, the fusion ones will expect a single value; others will expect a dict 
            # Try and load all values to cuda in the dict. 
            
            for idx in range(number_of_modalities):
                X[idx] = X[idx].type(torch.FloatTensor)
                if len(X[idx].size())<5:
                    X[idx] = torch.unsqueeze(X[idx], dim=1) # We add an extra dimension (~ number of channels) for the 3D convolutions.
                if fused_data_flag == False:
                    X[idx] = X[idx].cuda(device, non_blocking=True)
            
            if fused_data_flag == True:
                for idx in range(1, number_of_modalities):
                    if idx == 1:
                        Xnew = torch.cat((X[0], X[1]), dim=1)
                    else:
                        Xnew = torch.cat((Xnew, X[idx]), dim=1)
                X = torch.clone(Xnew)
                del Xnew
                if torch.cuda.is_available():
                    X = X.cuda(device, non_blocking=True)

            y_hat = model(X)   # Forward pass
            y_hat = (y_hat.cpu().numpy()).astype('float32')
            y_hat = np.squeeze(y_hat)

            if control == 'mean':
                target_age = _load_mean(dataset_sex)
            elif control == 'null':
                target_age = np.array(0.0)
            elif control == 'both':
                target_age_mean = _load_mean(dataset_sex)
                target_age_null = np.array(0.0)
                target_age = y_age
            else:
                target_age = y_age
            if control == 'both':
                age_delta, loss = _statistics_calculator(y_hat, target_age)
                age_delta_mean, loss_mean = _statistics_calculator(y_hat, target_age_mean)
                age_delta_null, loss_null = _statistics_calculator(y_hat, target_age_null)
                output_statistics[subject] = [target_age, y_hat, age_delta, loss, age_delta_mean, loss_mean, age_delta_null, loss_null]
            else:
                age_delta, loss = _statistics_calculator(y_hat, target_age)
                output_statistics[subject] = [target_age, y_hat, age_delta, loss]


            log.info("Processed: " + volumes_to_be_used[batch_index] + " " + str(batch_index + 1) + " out of " + str(len(volumes_to_be_used)))

            print("\r Processed {:.3f}%: {}/{} subjects".format((batch_index+1)/len_test_data * 100.0, batch_index+1, len_test_data), end='')

        if control == 'both':
            columns=['target_age', 'output_age', 'age_delta', 'loss', 'age_delta_mean', 'loss_mean', 'age_delta_null', 'loss_null']
            output_statistics_df = pd.DataFrame.from_dict(output_statistics, orient='index', columns=columns)
        else:
            output_statistics_df = pd.DataFrame.from_dict(output_statistics, orient='index', columns=['target_age', 'output_age', 'age_delta', 'loss'])     
        output_statistics_df.to_csv(output_statistics_path)

    log.info("Output Data Generation Complete")

def _load_mean(dataset_sex):

    if dataset_sex == 'male':
        mean_age = 64.64810970818492
    else:
        mean_age = 63.370316719492536

    mean_age = np.array(np.float32(mean_age))

    return mean_age 

def _statistics_calculator(output_age, target_age):

    output_age = np.array(output_age)
    target_age = np.array(target_age)
    age_delta = output_age - target_age
    loss = MSELoss(torch.from_numpy(output_age), torch.from_numpy(target_age)).numpy()

    return age_delta, loss
