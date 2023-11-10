import os
import numpy as np
import torch
import torch.nn  as nn
import torch.utils.data as data
import h5py
import pandas as pd
from fsl.data.image import Image
from fsl.utils.image.roi import roi
from scipy.stats import norm
import nibabel as nib


class DynamicDataMapper(data.Dataset):

    def __init__(self, X_paths, y_ages, modality_flag, scale_factor, resolution, t1t2ratio_flag=0, rsfmri_volume=None, shift_transformation=False, mirror_transformation=False):

        self.X_paths = X_paths # List where all the file paths are already in the deisred order
        self.X_paths_shape = len(X_paths) # Indicator of how many modalities we're dealing with it
        self.y_ages = y_ages
        self.modality_flag = list(modality_flag)
        self.scale_factor = scale_factor
        self.resolution = resolution
        self.shift_transformation = shift_transformation
        self.mirror_transformation = mirror_transformation
        self.rsfmri_volume = rsfmri_volume
        self.non_deterministic_modalities = ['T1_nonlinear', 'T1_linear', 'T2_nonlinear']

        self.t1t2ratio_flag = t1t2ratio_flag
        
    def preprocessInput(self, X_volume, idx):

        if self.resolution[idx] == '2mm':
            crop_values = [5, 85, 6, 102, 0, 80]
            # upsample_flag = True
        else:
            crop_values = [10, 170, 12, 204, 0, 160]
            # upsample_flag = False

        X_volume = X_volume[crop_values[0]:crop_values[1],
                            crop_values[2]:crop_values[3], 
                            crop_values[4]:crop_values[5]]

        if self.modality_flag[idx] in self.non_deterministic_modalities:
            X_volume = X_volume / X_volume.mean()

        X_volume = X_volume / self.scale_factor[idx]
        
        # if self.mirror_transformation==True:
        #     prob = np.random.rand(1)
        #     if prob < 0.5:
        #         X_volume = np.flip(X_volume,0)

        # if self.shift_transformation==True:
        #     x_shift, y_shift, z_shift = np.random.randint(-2,3,3)
        #     X_volume = np.roll(X_volume,x_shift,axis=0)
        #     X_volume = np.roll(X_volume,y_shift,axis=1)
        #     X_volume = np.roll(X_volume,z_shift,axis=2)
        #     if z_shift < 0:
        #         X_volume[:,:,z_shift:] = 0
            
        # X_volume = torch.from_numpy(X_volume)

        # if upsample_flag == True:
        #     UpsampleOp = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        #     X_volume = torch.squeeze(UpsampleOp(torch.unsqueeze(torch.unsqueeze(X_volume, dim=0),dim=0)))

        # Another option would be to have this code in solver.py after the data has been loaded to the GPU

        return X_volume

    def augmentInput(self, X_volume):

        if self.mirror_transformation==True:
            prob = np.random.rand(1)
            if prob < 0.5:
                X_volume = np.flip(X_volume,0)

        if self.shift_transformation==True:
            x_shift, y_shift, z_shift = np.random.randint(-2,3,3)
            X_volume = np.roll(X_volume,x_shift,axis=0)
            X_volume = np.roll(X_volume,y_shift,axis=1)
            X_volume = np.roll(X_volume,z_shift,axis=2)
            if z_shift < 0:
                X_volume[:,:,z_shift:] = 0

        return X_volume

    def numpyToTensor(self, X_volume, idx):
        
        if self.resolution[idx] == '2mm':
            upsample_flag = True
        else:
            upsample_flag = False

        X_volume = torch.from_numpy(X_volume)

        if upsample_flag == True:
            UpsampleOp = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            X_volume = torch.squeeze(UpsampleOp(torch.unsqueeze(torch.unsqueeze(X_volume, dim=0),dim=0)))

        # Another option would be to have this code in solver.py after the data has been loaded to the GPU

        return X_volume
        
    def __getitem__(self, index):

        X_volumes = {}

        # New code
        X_ratio = None
        #

        for idx in range(self.X_paths_shape):            
            if self.rsfmri_volume[idx] != None:
                X_volume = np.array(nib.load(self.X_paths[idx][index]).dataobj)[:,:,:,self.rsfmri_volume[idx]]
            else:
                X_volume = np.array(nib.load(self.X_paths[idx][index]).dataobj)                

            X_volume = self.preprocessInput(X_volume, idx)

            # New code
            if self.t1t2ratio_flag != 0: 
                if self.modality_flag[idx] == 'T1_nonlinear':
                    X_ratio = np.copy(X_volume)
                    X_volume = self.augmentInput(X_volume)
                    X_volume = self.numpyToTensor(X_volume, idx)
                    X_volumes[idx] = X_volume
                if self.modality_flag[idx] == 'T2_nonlinear':
                    
                    X_ratio = np.divide(X_ratio, X_volume, where=X_volume != 0)
                    X_ratio_scaling = np.percentile(np.abs(X_ratio.flatten()), 99)
                    X_ratio[X_ratio > X_ratio_scaling] = X_ratio_scaling
                    X_ratio[X_ratio < -X_ratio_scaling] = -X_ratio_scaling
                    X_ratio = X_ratio / X_ratio_scaling

                    
                    if self.t1t2ratio_flag == 2:

                        X_volume = self.augmentInput(X_volume)
                        X_volume = self.numpyToTensor(X_volume, idx)
                        X_volumes[idx] = X_volume

                        X_ratio = self.augmentInput(X_ratio)
                        X_ratio = self.numpyToTensor(X_ratio, idx)
                        X_volumes[idx+1] = X_ratio

                    else:
                        X_ratio = self.augmentInput(X_ratio)
                        X_ratio = self.numpyToTensor(X_ratio, idx)
                        X_volumes[idx] = X_ratio
            else:
                X_volume = self.augmentInput(X_volume)
                X_volume = self.numpyToTensor(X_volume, idx)
                X_volumes[idx] = X_volume

            #  X_volumes[idx] = X_volume

        y_age = np.array(self.y_ages[index])

        return X_volumes, y_age

    def __len__(self):
        return len(self.X_paths[0])


def select_datasets_path(data_parameters):
    dataset_sex = data_parameters['dataset_sex']
    dataset_size = data_parameters['dataset_size']
    data_folder_name = data_parameters['data_folder_name']

    X_train_list_path = data_parameters['male_train']
    y_train_ages_path = data_parameters['male_train_age']
    X_validation_list_path = data_parameters['male_validation']
    y_validation_ages_path = data_parameters['male_validation_age']

    if dataset_sex == 'female':
        X_train_list_path = "fe" + X_train_list_path
        y_train_ages_path = "fe" + y_train_ages_path
        X_validation_list_path = "fe" + X_validation_list_path
        y_validation_ages_path = "fe" + y_validation_ages_path

    if dataset_size == 'small':
        X_train_list_path += "_small.txt"
        y_train_ages_path += "_small.npy"
        X_validation_list_path += "_small.txt"
        y_validation_ages_path += "_small.npy"
    elif dataset_size == 'tiny':
        X_train_list_path += "_tiny.txt"
        y_train_ages_path += "_tiny.npy"
        X_validation_list_path += "_tiny.txt"
        y_validation_ages_path += "_tiny.npy"
    else:
        X_train_list_path += ".txt"
        y_train_ages_path += ".npy"
        X_validation_list_path += ".txt"
        y_validation_ages_path += ".npy"

    if dataset_size == 'han':
        X_train_list_path = "train_han.txt"
        y_train_ages_path = "train_age_han.npy"
        X_validation_list_path = "validation_han.txt"
        y_validation_ages_path = "validation_age_han.npy"

    if dataset_size == 'everything':
        X_train_list_path = "train_everything.txt"
        y_train_ages_path = "train_age_everything.npy"
        X_validation_list_path = "validation_everything.txt"
        y_validation_ages_path = "validation_age_everything.npy"

    if 'small' in dataset_size and dataset_size!='small':
        # ATTENTION! Cross Validation only enabled for male subjects at the moment!
        print('ATTENTION! CROSS VALIDATION DETECTED. This will only work for small male subject datasets ATM!')
        X_train_list_path = data_parameters['male_train'] + '_' + dataset_size + '.txt'
        y_train_ages_path = data_parameters['male_train_age'] + '_' + dataset_size + '.npy'
        X_validation_list_path = data_parameters['male_validation'] + '_' + dataset_size + '.txt'
        y_validation_ages_path = data_parameters['male_validation_age'] + '_' + dataset_size + '.npy'

    X_train_list_path = os.path.join(data_folder_name, X_train_list_path)
    y_train_ages_path = os.path.join(data_folder_name, y_train_ages_path)
    X_validation_list_path = os.path.join(data_folder_name, X_validation_list_path)
    y_validation_ages_path = os.path.join(data_folder_name, y_validation_ages_path)

    return X_train_list_path, y_train_ages_path, X_validation_list_path, y_validation_ages_path


def select_test_datasets_path(data_parameters, mapping_evaluation_parameters):
    dataset_sex = data_parameters['dataset_sex']
    dataset_size = data_parameters['dataset_size']

    prediction_output_statistics_name = mapping_evaluation_parameters['prediction_output_statistics_name']

    if mapping_evaluation_parameters['dataset_type'] == 'validation':
        X_test_list_path = mapping_evaluation_parameters['male_validation']
        y_test_ages_path = mapping_evaluation_parameters['male_validation_age']
        prediction_output_statistics_name += '_validation.csv'
    elif mapping_evaluation_parameters['dataset_type'] == 'train':
        X_test_list_path = data_parameters['male_train']
        y_test_ages_path = data_parameters['male_train_age']
        prediction_output_statistics_name += '_train.csv'
    else:
        X_test_list_path = mapping_evaluation_parameters['male_test']
        y_test_ages_path = mapping_evaluation_parameters['male_test_age']
        prediction_output_statistics_name += '_test.csv'

    if dataset_sex == 'female':
        X_test_list_path = "fe" + X_test_list_path
        y_test_ages_path = "fe" + y_test_ages_path

    if dataset_size == 'small':
        X_test_list_path += "_small.txt"
        y_test_ages_path += "_small.npy"
    elif dataset_size == 'tiny':
        X_test_list_path += "_tiny.txt"
        y_test_ages_path += "_tiny.npy"
    elif 'small' in dataset_size and dataset_size!='small':
        X_test_list_path += "_small.txt"
        y_test_ages_path += "_small.npy"
    else:
        X_test_list_path += ".txt"
        y_test_ages_path += ".npy"

    if dataset_size == 'han':
        X_test_list_path = "test_han.txt"
        y_test_ages_path = "test_age_han.npy"

    if dataset_size == 'everything':
        X_test_list_path = "test_everything.txt"
        y_test_ages_path = "test_age_everything.npy"

    X_test_list_path = 'datasets/' + X_test_list_path
    y_test_ages_path = 'datasets/' + y_test_ages_path


    # X_test_list_path = data_parameters['male_train']
    # y_test_ages_path = data_parameters['male_train_age']


    return X_test_list_path, y_test_ages_path, prediction_output_statistics_name



def get_datasets_dynamically(data_parameters):

    X_train_list_path, y_train_ages_path, X_validation_list_path, y_validation_ages_path = select_datasets_path(data_parameters)

    data_directory = data_parameters['data_directory']
    modality_flag = data_parameters['modality_flag']
    scaling_values_simple = pd.read_csv(data_parameters['scaling_values'], index_col=0)

    scale_factor = scaling_values_simple.loc[list(modality_flag)].scale_factor.to_list()
    resolution = scaling_values_simple.loc[list(modality_flag)].resolution.to_list()
    data_file = scaling_values_simple.loc[list(modality_flag)].data_file.to_list()

    rsfmri_volumes = []
    for modality in list(modality_flag):
        modality_flag_split = modality.rsplit('_', 1)
        if modality_flag_split[0] == 'rsfmri':
            rsfmri_volume = int(modality_flag_split[1])
        else:
            rsfmri_volume = None
        rsfmri_volumes.append(rsfmri_volume)
    if rsfmri_volumes == []:
        del rsfmri_volumes

    shift_transformation = data_parameters['shift_transformation']
    mirror_transformation = data_parameters['mirror_transformation']

    X_train_paths, _ = load_file_paths(X_train_list_path, data_directory, data_file)
    X_validation_paths, _ = load_file_paths(X_validation_list_path, data_directory, data_file)
    y_train_ages = np.load(y_train_ages_path)
    y_validation_ages = np.load(y_validation_ages_path)

    print('****************************************************************')
    print("DATASET INFORMATION")
    print('====================')
    print("Modality Name: ", modality_flag)
    if rsfmri_volumes != None:
        print("rsfMRI Volume: ", rsfmri_volumes)
    print("Resolution: ", resolution)
    print("Scale Factor: ", scale_factor)
    print("Data File Path: ", data_file)
    print('****************************************************************')

    t1t2ratio_flag = data_parameters['t1t2ratio_flag']

    return (
        DynamicDataMapper( X_paths=X_train_paths, y_ages=y_train_ages, modality_flag=modality_flag, 
                            scale_factor=scale_factor, resolution=resolution, t1t2ratio_flag=t1t2ratio_flag ,rsfmri_volume=rsfmri_volumes,
                            shift_transformation=shift_transformation, mirror_transformation=mirror_transformation),
        DynamicDataMapper( X_paths=X_validation_paths, y_ages=y_validation_ages, modality_flag=modality_flag, 
                            scale_factor=scale_factor, resolution=resolution, t1t2ratio_flag=t1t2ratio_flag, rsfmri_volume=rsfmri_volumes,
                            shift_transformation=False, mirror_transformation=False)
    )


def get_test_datasets_dynamically(data_parameters, mapping_evaluation_parameters):

    X_test_list_path, y_test_ages_path, prediction_output_statistics_name = select_test_datasets_path(data_parameters, mapping_evaluation_parameters)

    data_directory = data_parameters['data_directory']
    modality_flag = data_parameters['modality_flag']
    scaling_values_simple = pd.read_csv(data_parameters['scaling_values'], index_col=0)

    scale_factor = scaling_values_simple.loc[list(modality_flag)].scale_factor.to_list()
    resolution = scaling_values_simple.loc[list(modality_flag)].resolution.to_list()
    data_file = scaling_values_simple.loc[list(modality_flag)].data_file.to_list()

    rsfmri_volumes = []
    for modality in list(modality_flag):
        modality_flag_split = modality.rsplit('_', 1)
        if modality_flag_split[0] == 'rsfmri':
            rsfmri_volume = int(modality_flag_split[1])
        else:
            rsfmri_volume = None
        rsfmri_volumes.append(rsfmri_volume)
    if rsfmri_volumes == []:
        del rsfmri_volumes

    X_test_paths, X_test_volumes_to_be_used = load_file_paths(X_test_list_path, data_directory, data_file)
    y_test_ages = np.load(y_test_ages_path)

    print('****************************************************************')
    print("DATASET INFORMATION")
    print('====================')
    print("Modality Name: ", modality_flag)
    if rsfmri_volumes != None:
        print("rsfMRI Volume: ", rsfmri_volumes)
    print("Resolution: ", resolution)
    print("Scale Factor: ", scale_factor)
    print("Data File Path: ", data_file)
    print('****************************************************************')

    t1t2ratio_flag = data_parameters['t1t2ratio_flag']

    return (
        DynamicDataMapper( X_paths=X_test_paths, y_ages=y_test_ages, modality_flag=modality_flag, 
                            scale_factor=scale_factor, resolution=resolution, t1t2ratio_flag=t1t2ratio_flag, rsfmri_volume=rsfmri_volumes,
                            shift_transformation=False, mirror_transformation=False),
        X_test_volumes_to_be_used,
        prediction_output_statistics_name
    )


def load_file_paths(data_list, data_directory, mapping_data_file):

    volumes_to_be_used = load_subjects_from_path(data_list)
    
    file_paths = []
    for mapping_file in mapping_data_file:
        file_path = [os.path.join(data_directory, volume, mapping_file) for volume in volumes_to_be_used]
        file_paths.append(file_path)

    return file_paths, volumes_to_be_used 


def load_subjects_from_path(data_list):

    with open(data_list) as data_list_file:
        volumes_to_be_used = data_list_file.read().splitlines()

    return volumes_to_be_used


def load_and_preprocess_evaluation(file_path, modality_flag, resolution, scale_factor, rsfmri_volume=None):

    non_deterministic_modalities = ['T1_nonlinear', 'T1_linear', 'T2_nonlinear']

    if rsfmri_volume != None:
        volume = np.array(nib.load(file_path).dataobj)[:,:,:,rsfmri_volume]
    else:
        volume = np.array(nib.load(file_path).dataobj)

    if resolution == '2mm':
        crop_values = [5, 85, 6, 102, 0, 80]
    else:
        crop_values = [10, 170, 12, 204, 0, 160]

    volume = volume[crop_values[0]:crop_values[1],
                    crop_values[2]:crop_values[3], 
                    crop_values[4]:crop_values[5]]

    if modality_flag in non_deterministic_modalities:
        volume = volume / volume.mean()

    volume = volume / scale_factor

    volume = np.float32(volume)

    return volume


def num2vec(label, bin_range=[44,84], bin_step=1, std=1):

    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if bin_length % bin_step != 0:
        print("Error: Bin range should be divisible by the bin step!")
        return None
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + bin_step/2.0 + bin_step * np.arange(bin_number)
    
    if std == 0:
        # Uniform Distribution Case
        label = np.array(label)
        bin_values = np.floor((label - bin_start)/bin_step).astype(int)
    elif std < 0:
        print("Error! The standard deviation (& variance) must be positive")
        return None
    else:
        bin_values = np.zeros((bin_number))
        for i in range(bin_number):
            x1 = bin_centers[i] - bin_step/2.0
            x2 = bin_centers[i] + bin_step/2.0
            cdfs = norm.cdf([x1, x2], loc=label, scale=std)
            bin_values[i] = cdfs[1] - cdfs[0]       

    return bin_values, bin_centers