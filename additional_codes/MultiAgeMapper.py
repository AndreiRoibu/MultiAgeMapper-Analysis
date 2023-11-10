from typing import OrderedDict
import numpy as np
from torch import prod, tensor, cat
import torch.nn as nn
import torch.nn.functional as F
import torch


class torch_permute(nn.Module):
    def __init__(self, shp):
        super(torch_permute, self).__init__()
        self.shp = shp
    def forward(self, X):
        X = X.permute(self.shp)
        return X

class AgeMapper_input1(nn.Module):
    def __init__(self, 
                fused_data_flag = False,
                original_input_channels = 1,
                channel_number=[32,64,64,64,64],
                network_parameters = None,
                ):

        super(AgeMapper_input1, self).__init__()
        number_of_layers = len(channel_number)

        if network_parameters is not None:
            norm_flag = network_parameters['norm_flag']
            nonlin_flag = network_parameters['nonlin_flag']
            dropout_flag = network_parameters['dropout_flag']
        else:
            norm_flag = None
            nonlin_flag = None
            dropout_flag = 0

        self.fused_data_flag = fused_data_flag

        self.Feature_Extractor = nn.Sequential()
        for layer_number in range(number_of_layers):      
            if layer_number == 0:
                input_channels = original_input_channels
            else:
                input_channels = channel_number[layer_number - 1]
            output_channels = channel_number[layer_number]

            self.Feature_Extractor.add_module(
                name = 'Convolution_%d' % layer_number,
                module = self._convolutional_block(
                    input_channels,
                    output_channels,
                    maxpool_flag = True,
                    kernel_size = 3,
                    padding_flag= True, 
                    norm_flag = norm_flag,
                    nonlin_flag = nonlin_flag,
                )
            )

        self.FullyConnected = nn.Sequential()
        input_dimensions = 5 * 6 * 5 * output_channels

        if dropout_flag > 0:
            self.FullyConnected.add_module(
                name='Dropout_FullyConnected_3',
                module=nn.Dropout(dropout_flag)
            )

        self.FullyConnected.add_module(
            name = 'FullyConnected_3',
            module=nn.Linear(
                in_features=input_dimensions,
                out_features=96
            )
        )

        if nonlin_flag == 'leakyrelu':
            self.FullyConnected.add_module(
                name = 'LeakyReluActivation_3',
                module= nn.LeakyReLU()
            )
        else:
            self.FullyConnected.add_module(
                name = 'ReluActivation_3',
                module= nn.ReLU()
            )

        self.FullyConnected.add_module(
            name = 'FullyConnected_2',
            module=nn.Linear(
                in_features=96,
                out_features=32
            )
        )
        if nonlin_flag == 'leakyrelu':
            self.FullyConnected.add_module(
                name = 'LeakyReluActivation_2',
                module= nn.LeakyReLU()
            )
        else:
            self.FullyConnected.add_module(
                name = 'ReluActivation_2',
                module= nn.ReLU()
            )

        self.FullyConnected.add_module(
            name = 'FullyConnected_1',
            module= nn.Linear(
                in_features=32,
                out_features=1,
            )
        )
        self.FullyConnected.add_module(
            name = 'LinearActivation',
            module= nn.Identity()
        )

    @staticmethod
    def _convolutional_block(input_channels, output_channels, maxpool_flag=True, kernel_size=3, padding_flag=True, maxpool_stride=2,
                            norm_flag = None,
                            nonlin_flag = None,

                            ):
        
        if padding_flag == True:
            padding = int((kernel_size - 1) / 2)
        else:
            padding = 0

        if norm_flag is not None:
            if norm_flag == 'batch':
                normalization = nn.BatchNorm3d(num_features=output_channels, affine=True)
            elif norm_flag == 'instance':
                normalization = nn.InstanceNorm3d(num_features=output_channels,  affine=True, track_running_stats=True)
            elif norm_flag == 'instance_default':
                normalization = nn.InstanceNorm3d(num_features=output_channels)
            elif norm_flag == 'layer':
                normalization = nn.Sequential(
                    torch_permute(shp=(0,2,3,4,1)),
                    nn.LayerNorm(normalized_shape=output_channels),
                    torch_permute(shp=(0,-1,1,2,3))
                )
            elif norm_flag == 'group':
                normalization = nn.GroupNorm(num_groups=8, num_channels=output_channels)
            else:
                print("WARNING - NORM IS NOT VALID. Defaulting to BatchNorm")
                normalization = nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                )
        else:
            print("WARNING - NORM NOT PROVIDED. Defaulting to BatchNorm")
            normalization = nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                )

        if nonlin_flag is not None:
            if nonlin_flag == 'relu':
                nonlinearity = nn.ReLU()
            elif nonlin_flag == 'leakyrelu':
                nonlinearity = nn.LeakyReLU() 
            else:
                print("WARNING - NONLIN IS NOT VALID. Defaulting to ReLU")
                nonlinearity = nn.ReLU()
        else:
            print("WARNING - NONLI NOT PROVIDED VALID. Defaulting to ReLU")
            nonlinearity = nn.ReLU()
        
        if maxpool_flag is True:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                # nn.BatchNorm3d(
                #     num_features=output_channels,
                #     affine=True
                # ),
                normalization,
                nn.MaxPool3d(
                    kernel_size=2,
                    stride=maxpool_stride
                ),
                # nn.ReLU()
                nonlinearity
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                # nn.BatchNorm3d(
                #     num_features=output_channels,
                #     affine=True
                # ),
                normalization,
                # nn.ReLU()
                nonlinearity
            )

        return layer

    def forward(self, X):
        
        if self.fused_data_flag == False:
            print('ERROR! This network works with fused data')

        X = self.Feature_Extractor(X)
        X = X.reshape(-1, prod(tensor(X.shape)[1:]))
        X = self.FullyConnected(X)
        return X


class AgeMapper_input2(nn.Module):
    def __init__(self, 
                fused_data_flag = False,
                original_input_channels = 1,
                network_2_modality_filter_outputs = 1,
                channel_number=[32,64,64,64,64],
                ):

        self.fused_data_flag = fused_data_flag

        super(AgeMapper_input2, self).__init__()
        number_of_layers = len(channel_number)

        self.Feature_Extractor = nn.Sequential()

        self.Feature_Extractor.add_module(
            name = 'Convolution_Modality_Filter',
            module = nn.Conv3d(
                in_channels=original_input_channels,
                out_channels=network_2_modality_filter_outputs,
                kernel_size=1,
                padding=0,
                bias=False
            )
        )

        for layer_number in range(number_of_layers):      
            if layer_number == 0:
                input_channels = network_2_modality_filter_outputs
            else:
                input_channels = channel_number[layer_number - 1]
            output_channels = channel_number[layer_number]

            self.Feature_Extractor.add_module(
                name = 'Convolution_%d' % layer_number,
                module = self._convolutional_block(
                    input_channels,
                    output_channels,
                    maxpool_flag = True,
                    kernel_size = 3,
                    padding_flag= True
                )
            )

        self.FullyConnected = nn.Sequential()
        input_dimensions = 5 * 6 * 5 * output_channels
        self.FullyConnected.add_module(
            name = 'FullyConnected_3',
            module=nn.Linear(
                in_features=input_dimensions,
                out_features=96
            )
        )
        self.FullyConnected.add_module(
            name = 'ReluActivation_3',
            module= nn.ReLU()
        )

        self.FullyConnected.add_module(
            name = 'FullyConnected_2',
            module=nn.Linear(
                in_features=96,
                out_features=32
            )
        )
        self.FullyConnected.add_module(
            name = 'ReluActivation_2',
            module= nn.ReLU()
        )

        self.FullyConnected.add_module(
            name = 'FullyConnected_1',
            module= nn.Linear(
                in_features=32,
                out_features=1,
            )
        )
        self.FullyConnected.add_module(
            name = 'LinearActivation',
            module= nn.Identity()
        )

    @staticmethod
    def _convolutional_block(input_channels, output_channels, maxpool_flag=True, kernel_size=3, padding_flag=True, maxpool_stride=2):
        
        if padding_flag == True:
            padding = int((kernel_size - 1) / 2)
        else:
            padding = 0
        
        if maxpool_flag is True:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                ),
                nn.MaxPool3d(
                    kernel_size=2,
                    stride=maxpool_stride
                ),
                nn.ReLU()
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                ),
                nn.ReLU()
            )

        return layer

    def forward(self, X):
        
        if self.fused_data_flag == False:
            print('ERROR! This network works with fused data')

        X = self.Feature_Extractor(X)
        X = X.reshape(-1, prod(tensor(X.shape)[1:]))
        X = self.FullyConnected(X)
        return X


class AgeMapper_input3_hardcode_backup(nn.Module):
    def __init__(self, 
                fused_data_flag = False,
                original_input_channels = 1,
                channel_number=[32,64,64,64,64],
                ):

        # For this network I either need to create a static method for the convolutional paths, or another class that this one inherits
        # Alternativelly, I would need to hard code the structure (might be easier)
        # I will also need to read the separate input dictionary entries here and pass them through the relevant paths
        # I will also need to make sure that the various paths are united / fused at the relevant location

        # Doing things automatically might be easier later when I am testing how many modalities can come in before running out of memory


        self.fused_data_flag = fused_data_flag

        super(AgeMapper_input3, self).__init__()
        number_of_layers = len(channel_number)

        self.Feature_Extractor = nn.Sequential()
        for layer_number in range(number_of_layers):      
            if layer_number == 0:
                input_channels = original_input_channels
            else:
                input_channels = channel_number[layer_number - 1]
            output_channels = channel_number[layer_number]

            self.Feature_Extractor.add_module(
                name = 'Convolution_%d' % layer_number,
                module = self._convolutional_block(
                    input_channels,
                    output_channels,
                    maxpool_flag = True,
                    kernel_size = 3,
                    padding_flag= True
                )
            )

        self.FullyConnected = nn.Sequential()
        input_dimensions = 5 * 6 * 5 * output_channels
        self.FullyConnected.add_module(
            name = 'FullyConnected_3',
            module=nn.Linear(
                in_features=input_dimensions,
                out_features=96
            )
        )
        self.FullyConnected.add_module(
            name = 'ReluActivation_3',
            module= nn.ReLU()
        )

        self.FullyConnected.add_module(
            name = 'FullyConnected_2',
            module=nn.Linear(
                in_features=96,
                out_features=32
            )
        )
        self.FullyConnected.add_module(
            name = 'ReluActivation_2',
            module= nn.ReLU()
        )

        self.FullyConnected.add_module(
            name = 'FullyConnected_1',
            module= nn.Linear(
                in_features=32,
                out_features=1,
            )
        )
        self.FullyConnected.add_module(
            name = 'LinearActivation',
            module= nn.Identity()
        )

    @staticmethod
    def _convolutional_block(input_channels, output_channels, maxpool_flag=True, kernel_size=3, padding_flag=True, maxpool_stride=2):
        
        if padding_flag == True:
            padding = int((kernel_size - 1) / 2)
        else:
            padding = 0
        
        if maxpool_flag is True:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                ),
                nn.MaxPool3d(
                    kernel_size=2,
                    stride=maxpool_stride
                ),
                nn.ReLU()
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                ),
                nn.ReLU()
            )

        return layer

    def forward(self, X):
        
        if self.fused_data_flag == False:
            print('ERROR! This network works with fused data')

        X = self.Feature_Extractor(X)
        X = X.reshape(-1, prod(tensor(X.shape)[1:]))
        X = self.FullyConnected(X)
        return X


class AgeMapperConv(nn.Module):
    def __init__(
        self,
        channel_number=[32,64,64,64,64],
        path_number = 1, 
        norm_flag = None,
        nonlin_flag = None,
                ):

        super(AgeMapperConv, self).__init__()
        number_of_layers = len(channel_number)

        self.Feature_Extractor = nn.Sequential()
        for layer_number in range(number_of_layers):      
            if layer_number == 0:
                input_channels = 1
            else:
                input_channels = channel_number[layer_number - 1]
            output_channels = channel_number[layer_number]

            self.Feature_Extractor.add_module(
                name = 'Path_%d_Convolution_%d' % (path_number, layer_number),
                module = self._convolutional_block(
                    input_channels,
                    output_channels,
                    maxpool_flag = True,
                    kernel_size = 3,
                    padding_flag= True, 
                    norm_flag = norm_flag,
                    nonlin_flag = nonlin_flag,
                )
            )

    @staticmethod
    def _convolutional_block(input_channels, output_channels, maxpool_flag=True, kernel_size=3, padding_flag=True, maxpool_stride=2,
                            norm_flag = None,
                            nonlin_flag = None,

                            ):

        if padding_flag == True:
            padding = int((kernel_size - 1) / 2)
        else:
            padding = 0

        if norm_flag is not None:
            if norm_flag == 'batch':
                normalization = nn.BatchNorm3d(num_features=output_channels, affine=True)
            elif norm_flag == 'instance':
                normalization = nn.InstanceNorm3d(num_features=output_channels,  affine=True, track_running_stats=True)
            elif norm_flag == 'instance_default':
                normalization = nn.InstanceNorm3d(num_features=output_channels)
            elif norm_flag == 'layer':
                normalization = nn.Sequential(
                    torch_permute(shp=(0,2,3,4,1)),
                    nn.LayerNorm(normalized_shape=output_channels),
                    torch_permute(shp=(0,-1,1,2,3))
                )
            elif norm_flag == 'group':
                normalization = nn.GroupNorm(num_groups=8, num_channels=output_channels)
            else:
                print("WARNING - NORM IS NOT VALID. Defaulting to BatchNorm")
                normalization = nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                )
        else:
            print("WARNING - NORM NOT PROVIDED. Defaulting to BatchNorm")
            normalization = nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                )

        if nonlin_flag is not None:
            if nonlin_flag == 'relu':
                nonlinearity = nn.ReLU()
            elif nonlin_flag == 'leakyrelu':
                nonlinearity = nn.LeakyReLU() 
            else:
                print("WARNING - NONLIN IS NOT VALID. Defaulting to ReLU")
                nonlinearity = nn.ReLU()
        else:
            print("WARNING - NONLI NOT PROVIDED VALID. Defaulting to ReLU")
            nonlinearity = nn.ReLU()
        
        if maxpool_flag is True:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                # nn.BatchNorm3d(
                #     num_features=output_channels,
                #     affine=True
                # ),
                normalization,
                nn.MaxPool3d(
                    kernel_size=2,
                    stride=maxpool_stride
                ),
                # nn.ReLU()
                nonlinearity
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                # nn.BatchNorm3d(
                #     num_features=output_channels,
                #     affine=True
                # ),
                normalization,
                # nn.ReLU()
                nonlinearity
            )

        return layer

    def forward(self, X):
        X = self.Feature_Extractor(X)
        return X



class AgeMapper_input3(nn.Module):
    def __init__(self, 
                fused_data_flag = False,
                original_input_channels = 1,
                channel_number=[32,64,64,64,64],
                network_parameters = None,
                ):

        # For this network I either need to create a static method for the convolutional paths, or another class that this one inherits
        # Alternativelly, I would need to hard code the structure (might be easier)
        # I will also need to read the separate input dictionary entries here and pass them through the relevant paths
        # I will also need to make sure that the various paths are united / fused at the relevant location

        # Doing things automatically might be easier later when I am testing how many modalities can come in before running out of memory
        
        super(AgeMapper_input3, self).__init__()

        if network_parameters is not None:
            norm_flag = network_parameters['norm_flag']
            nonlin_flag = network_parameters['nonlin_flag']
            dropout_flag = network_parameters['dropout_flag']
        else:
            norm_flag = None
            nonlin_flag = None
            dropout_flag = 0

        self.fused_data_flag = fused_data_flag

        self.input_paths = original_input_channels

        self.Convolution_Paths = nn.ModuleDict({})
        for idx in range(self.input_paths):
            self.Convolution_Paths[str(idx)] = AgeMapperConv(channel_number=channel_number,
                                                            path_number=idx,
                                                            norm_flag=norm_flag,
                                                            nonlin_flag=nonlin_flag
                                                            )

        output_channels = channel_number[-1]

        self.FullyConnected = nn.Sequential()
        input_dimensions = 5 * 6 * 5 * output_channels * self.input_paths

        if dropout_flag > 0:
            self.FullyConnected.add_module(
                name='Dropout_FullyConnected_3',
                module=nn.Dropout(dropout_flag)
            )

        self.FullyConnected.add_module(
            name = 'FullyConnected_3',
            module=nn.Linear(
                in_features=input_dimensions,
                out_features=96
            )
        )
        if nonlin_flag == 'leakyrelu':
            self.FullyConnected.add_module(
                name = 'LeakyReluActivation_3',
                module= nn.LeakyReLU()
            )
        else:
            self.FullyConnected.add_module(
                name = 'ReluActivation_3',
                module= nn.ReLU()
            )

        self.FullyConnected.add_module(
            name = 'FullyConnected_2',
            module=nn.Linear(
                in_features=96,
                out_features=32
            )
        )
        if nonlin_flag == 'leakyrelu':
            self.FullyConnected.add_module(
                name = 'LeakyReluActivation_2',
                module= nn.LeakyReLU()
            )
        else:
            self.FullyConnected.add_module(
                name = 'ReluActivation_2',
                module= nn.ReLU()
            )

        self.FullyConnected.add_module(
            name = 'FullyConnected_1',
            module= nn.Linear(
                in_features=32,
                out_features=1,
            )
        )
        self.FullyConnected.add_module(
            name = 'LinearActivation',
            module= nn.Identity()
        )

    def forward(self, X):
        
        if self.fused_data_flag == True:
            print('ERROR! This network works with fused data')

        for idx in range(self.input_paths):
            X[idx] = self.Convolution_Paths[str(idx)](X[idx])
            if idx==0:
                X_output = X[idx].reshape(-1, prod(tensor(X[idx].shape)[1:]))
            else:
                X_output = cat((X_output, X[idx].reshape(-1, prod(tensor(X[idx].shape)[1:])) ), dim=1)

        del X

        X_output = self.FullyConnected(X_output)

        return X_output



class AgeMapperConvFC1(nn.Module):
    def __init__(
        self,
        channel_number=[32,64,64,64,64],
        path_number = 1
                ):

        super(AgeMapperConvFC1, self).__init__()
        number_of_layers = len(channel_number)

        self.Feature_Extractor = nn.Sequential()
        for layer_number in range(number_of_layers):      
            if layer_number == 0:
                input_channels = 1
            else:
                input_channels = channel_number[layer_number - 1]
            output_channels = channel_number[layer_number]

            self.Feature_Extractor.add_module(
                name = 'Path_%d_Convolution_%d' % (path_number, layer_number),
                module = self._convolutional_block(
                    input_channels,
                    output_channels,
                    maxpool_flag = True,
                    kernel_size = 3,
                    padding_flag= True
                )
            )

        self.FullyConnectedModality = nn.Sequential()
        input_dimensions = 5 * 6 * 5 * output_channels
        self.FullyConnectedModality.add_module(
            name = 'Path_%d_FullyConnected_3' % (path_number),
            module=nn.Linear(
                in_features=input_dimensions,
                out_features=96
            )
        )
        self.FullyConnectedModality.add_module(
            name = 'Path_%d_ReluActivation_3' % (path_number),
            module= nn.ReLU()
        )

    @staticmethod
    def _convolutional_block(input_channels, output_channels, maxpool_flag=True, kernel_size=3, padding_flag=True, maxpool_stride=2):
        
        if padding_flag == True:
            padding = int((kernel_size - 1) / 2)
        else:
            padding = 0
        
        if maxpool_flag is True:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                ),
                nn.MaxPool3d(
                    kernel_size=2,
                    stride=maxpool_stride
                ),
                nn.ReLU()
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                ),
                nn.ReLU()
            )

        return layer

    def forward(self, X):
        X = self.Feature_Extractor(X)
        X = X.reshape(-1, prod(tensor(X.shape)[1:]))
        X = self.FullyConnectedModality(X)
        return X



class AgeMapper_input4(nn.Module):
    def __init__(self, 
                fused_data_flag = False,
                original_input_channels = 1,
                channel_number=[32,64,64,64,64],
                ):

        self.fused_data_flag = fused_data_flag

        super(AgeMapper_input4, self).__init__()

        self.input_paths = original_input_channels

        self.Modality_Paths = nn.ModuleDict({})
        for idx in range(self.input_paths):
            self.Modality_Paths[str(idx)] = AgeMapperConvFC1(channel_number=channel_number,
                                                            path_number=idx)

        self.FullyConnected = nn.Sequential()
        input_dimensions = 96 * self.input_paths

        self.FullyConnected.add_module(
            name = 'FullyConnected_2',
            module=nn.Linear(
                in_features=input_dimensions,
                out_features=32
            )
        )
        self.FullyConnected.add_module(
            name = 'ReluActivation_2',
            module= nn.ReLU()
        )

        self.FullyConnected.add_module(
            name = 'FullyConnected_1',
            module= nn.Linear(
                in_features=32,
                out_features=1,
            )
        )
        self.FullyConnected.add_module(
            name = 'LinearActivation',
            module= nn.Identity()
        )

    def forward(self, X):
        
        if self.fused_data_flag == True:
            print('ERROR! This network works with fused data')

        for idx in range(self.input_paths):
            X[idx] = self.Modality_Paths[str(idx)](X[idx])
            if idx==0:
                X_output = X[idx]
            else:
                X_output = cat((X_output, X[idx]), dim=1)

        del X

        X_output = self.FullyConnected(X_output)

        return X_output


class AgeMapperConvFC2(nn.Module):
    def __init__(
        self,
        channel_number=[32,64,64,64,64],
        path_number = 1,
        norm_flag=None,
        nonlin_flag=None,
        dropout_flag=0
                ):

        super(AgeMapperConvFC2, self).__init__()
        number_of_layers = len(channel_number)

        self.Feature_Extractor = nn.Sequential()
        for layer_number in range(number_of_layers):      
            if layer_number == 0:
                input_channels = 1
            else:
                input_channels = channel_number[layer_number - 1]
            output_channels = channel_number[layer_number]

            self.Feature_Extractor.add_module(
                name = 'Path_%d_Convolution_%d' % (path_number, layer_number),
                module = self._convolutional_block(
                    input_channels,
                    output_channels,
                    maxpool_flag = True,
                    kernel_size = 3,
                    padding_flag= True, 
                    norm_flag = norm_flag,
                    nonlin_flag = nonlin_flag,
                )
            )

        self.FullyConnectedModality = nn.Sequential()
        input_dimensions = 5 * 6 * 5 * output_channels

        if dropout_flag > 0:
            self.FullyConnectedModality.add_module(
                name='Path_%d_Dropout_FullyConnected_3' % (path_number),
                module=nn.Dropout(dropout_flag)
            )

        self.FullyConnectedModality.add_module(
            name = 'Path_%d_FullyConnected_3' % (path_number),
            module=nn.Linear(
                in_features=input_dimensions,
                out_features=96
            )
        )

        if nonlin_flag == 'leakyrelu':
            self.FullyConnectedModality.add_module(
                name = 'Path_%d_LeakyReluActivation_3' % (path_number),
                module= nn.LeakyReLU()
            )
        else:
            self.FullyConnectedModality.add_module(
                name = 'Path_%d_ReluActivation_3' % (path_number),
                module= nn.ReLU()
            )

        self.FullyConnectedModality.add_module(
            name = 'Path_%d_FullyConnected_2' % (path_number),
            module=nn.Linear(
                in_features=96,
                out_features=32
            )
        )

        if nonlin_flag == 'leakyrelu':
            self.FullyConnectedModality.add_module(
                name = 'Path_%d_LeakyReluActivation_2' % (path_number),
                module= nn.LeakyReLU()
            )
        else:
            self.FullyConnectedModality.add_module(
                name = 'Path_%d_ReluActivation_2' % (path_number),
                module= nn.ReLU()
            )

    @staticmethod
    def _convolutional_block(input_channels, output_channels, maxpool_flag=True, kernel_size=3, padding_flag=True, maxpool_stride=2,
                            norm_flag = None,
                            nonlin_flag = None,
    ):
        
        if padding_flag == True:
            padding = int((kernel_size - 1) / 2)
        else:
            padding = 0

        if norm_flag is not None:
            if norm_flag == 'batch':
                normalization = nn.BatchNorm3d(num_features=output_channels, affine=True)
            elif norm_flag == 'instance':
                normalization = nn.InstanceNorm3d(num_features=output_channels,  affine=True, track_running_stats=True)
            elif norm_flag == 'instance_default':
                normalization = nn.InstanceNorm3d(num_features=output_channels)
            elif norm_flag == 'layer':
                normalization = nn.Sequential(
                    torch_permute(shp=(0,2,3,4,1)),
                    nn.LayerNorm(normalized_shape=output_channels),
                    torch_permute(shp=(0,-1,1,2,3))
                )
            elif norm_flag == 'group':
                normalization = nn.GroupNorm(num_groups=8, num_channels=output_channels)
            else:
                print("WARNING - NORM IS NOT VALID. Defaulting to BatchNorm")
                normalization = nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                )
        else:
            print("WARNING - NORM NOT PROVIDED. Defaulting to BatchNorm")
            normalization = nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                )

        if nonlin_flag is not None:
            if nonlin_flag == 'relu':
                nonlinearity = nn.ReLU()
            elif nonlin_flag == 'leakyrelu':
                nonlinearity = nn.LeakyReLU() 
            else:
                print("WARNING - NONLIN IS NOT VALID. Defaulting to ReLU")
                nonlinearity = nn.ReLU()
        else:
            print("WARNING - NONLI NOT PROVIDED VALID. Defaulting to ReLU")
            nonlinearity = nn.ReLU()
        
        if maxpool_flag is True:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                # nn.BatchNorm3d(
                #     num_features=output_channels,
                #     affine=True
                # ),
                normalization,
                nn.MaxPool3d(
                    kernel_size=2,
                    stride=maxpool_stride
                ),
                # nn.ReLU()
                nonlinearity
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                # nn.BatchNorm3d(
                #     num_features=output_channels,
                #     affine=True
                # ),
                normalization,
                # nn.ReLU()
                nonlinearity
            )

        return layer

    def forward(self, X):
        X = self.Feature_Extractor(X)
        X = X.reshape(-1, prod(tensor(X.shape)[1:]))
        X = self.FullyConnectedModality(X)
        return X



class AgeMapper_input5(nn.Module):
    def __init__(self, 
                fused_data_flag = False,
                original_input_channels = 1,
                channel_number=[32,64,64,64,64],
                network_parameters = None,
                ):

        super(AgeMapper_input5, self).__init__()

        if network_parameters is not None:
            norm_flag = network_parameters['norm_flag']
            nonlin_flag = network_parameters['nonlin_flag']
            dropout_flag = network_parameters['dropout_flag']
        else:
            norm_flag = None
            nonlin_flag = None
            dropout_flag = 0

        self.fused_data_flag = fused_data_flag

        self.input_paths = original_input_channels

        self.Modality_Paths = nn.ModuleDict({})
        for idx in range(self.input_paths):
            self.Modality_Paths[str(idx)] = AgeMapperConvFC2(channel_number=channel_number,
                                                            path_number=idx,
                                                            norm_flag=norm_flag,
                                                            nonlin_flag=nonlin_flag,
                                                            dropout_flag=dropout_flag
                                                            )

        self.FullyConnected = nn.Sequential()
        input_dimensions = 32 * self.input_paths

        self.FullyConnected.add_module(
            name = 'FullyConnected_1',
            module= nn.Linear(
                in_features=input_dimensions,
                out_features=1,
            )
        )
        self.FullyConnected.add_module(
            name = 'LinearActivation',
            module= nn.Identity()
        )

    def forward(self, X):
        
        if self.fused_data_flag == True:
            print('ERROR! This network works with fused data')

        for idx in range(self.input_paths):
            X[idx] = self.Modality_Paths[str(idx)](X[idx])
            if idx==0:
                X_output = X[idx]
            else:
                X_output = cat((X_output, X[idx]), dim=1)

        del X

        X_output = self.FullyConnected(X_output)

        return X_output




class AgeMapperConvFC3(nn.Module):
    def __init__(
        self,
        channel_number=[32,64,64,64,64],
        path_number = 1
                ):

        super(AgeMapperConvFC3, self).__init__()
        number_of_layers = len(channel_number)

        self.Feature_Extractor = nn.Sequential()
        for layer_number in range(number_of_layers):      
            if layer_number == 0:
                input_channels = 1
            else:
                input_channels = channel_number[layer_number - 1]
            output_channels = channel_number[layer_number]

            self.Feature_Extractor.add_module(
                name = 'Path_%d_Convolution_%d' % (path_number, layer_number),
                module = self._convolutional_block(
                    input_channels,
                    output_channels,
                    maxpool_flag = True,
                    kernel_size = 3,
                    padding_flag= True
                )
            )

        self.FullyConnectedModality = nn.Sequential()
        input_dimensions = 5 * 6 * 5 * output_channels
        self.FullyConnectedModality.add_module(
            name = 'Path_%d_FullyConnected_3' % (path_number),
            module=nn.Linear(
                in_features=input_dimensions,
                out_features=96
            )
        )
        self.FullyConnectedModality.add_module(
            name = 'Path_%d_ReluActivation_3' % (path_number),
            module= nn.ReLU()
        )
        self.FullyConnectedModality.add_module(
            name = 'Path_%d_FullyConnected_2' % (path_number),
            module=nn.Linear(
                in_features=96,
                out_features=32
            )
        )
        self.FullyConnectedModality.add_module(
            name = 'Path_%d_ReluActivation_2' % (path_number),
            module= nn.ReLU()
        )

        self.FullyConnectedModality.add_module(
            name = 'Path_%d_FullyConnected_1' % (path_number),
            module= nn.Linear(
                in_features=32,
                out_features=1,
            )
        )
        self.FullyConnectedModality.add_module(
            name = 'Path_%d_LinearActivation' % (path_number),
            module= nn.Identity()
        )

    @staticmethod
    def _convolutional_block(input_channels, output_channels, maxpool_flag=True, kernel_size=3, padding_flag=True, maxpool_stride=2):
        
        if padding_flag == True:
            padding = int((kernel_size - 1) / 2)
        else:
            padding = 0
        
        if maxpool_flag is True:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                ),
                nn.MaxPool3d(
                    kernel_size=2,
                    stride=maxpool_stride
                ),
                nn.ReLU()
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                ),
                nn.ReLU()
            )

        return layer

    def forward(self, X):
        X = self.Feature_Extractor(X)
        X = X.reshape(-1, prod(tensor(X.shape)[1:]))
        X = self.FullyConnectedModality(X)
        return X



class AgeMapper_input6(nn.Module):
    def __init__(self, 
                fused_data_flag = False,
                original_input_channels = 1,
                channel_number=[32,64,64,64,64],
                ):

        self.fused_data_flag = fused_data_flag

        super(AgeMapper_input6, self).__init__()

        self.input_paths = original_input_channels

        self.Modality_Paths = nn.ModuleDict({})
        for idx in range(self.input_paths):
            self.Modality_Paths[str(idx)] = AgeMapperConvFC3(channel_number=channel_number,
                                                            path_number=idx)

        self.FullyConnected = nn.Sequential()
        input_dimensions = 1 * self.input_paths

        self.FullyConnected.add_module(
            name = 'FullyConnected_0',
            module= nn.Linear(
                in_features=input_dimensions,
                out_features=1,
            )
        )
        self.FullyConnected.add_module(
            name = 'LinearActivation_0',
            module= nn.Identity()
        )

    def forward(self, X):
        
        if self.fused_data_flag == True:
            print('ERROR! This network works with fused data')

        for idx in range(self.input_paths):
            X[idx] = self.Modality_Paths[str(idx)](X[idx])
            if idx==0:
                X_output = X[idx]
            else:
                X_output = cat((X_output, X[idx]), dim=1)

        del X

        X_output = self.FullyConnected(X_output)

        return X_output



class AgeMapper_input7(nn.Module):
    def __init__(self, 
                fused_data_flag = False,
                original_input_channels = 1,
                channel_number=[32,64,64,64,64],
                ):

        self.fused_data_flag = fused_data_flag

        super(AgeMapper_input7, self).__init__()

        self.input_paths = original_input_channels

        self.Modality_Paths = nn.ModuleDict({})
        for idx in range(self.input_paths):
            self.Modality_Paths[str(idx)] = AgeMapperConv(channel_number=channel_number,
                                                            path_number=idx)

        output_channels = channel_number[-1]

        self.FullyConnected = nn.Sequential()
        input_dimensions = 5 * 6 * 5 * output_channels

        self.FullyConnected.add_module(
            name = 'FullyConnected_Link',
            module=nn.Linear(
                in_features=input_dimensions * self.input_paths,
                out_features=input_dimensions,
            )
        )
        self.FullyConnected.add_module(
            name = 'ReluActivation_Link',
            module= nn.ReLU()
        )

        self.FullyConnected.add_module(
            name = 'FullyConnected_3',
            module=nn.Linear(
                in_features=input_dimensions,
                out_features=96
            )
        )
        self.FullyConnected.add_module(
            name = 'ReluActivation_3',
            module= nn.ReLU()
        )

        self.FullyConnected.add_module(
            name = 'FullyConnected_2',
            module=nn.Linear(
                in_features=96,
                out_features=32
            )
        )
        self.FullyConnected.add_module(
            name = 'ReluActivation_2',
            module= nn.ReLU()
        )

        self.FullyConnected.add_module(
            name = 'FullyConnected_1',
            module= nn.Linear(
                in_features=32,
                out_features=1,
            )
        )
        self.FullyConnected.add_module(
            name = 'LinearActivation',
            module= nn.Identity()
        )

    def forward(self, X):
        
        if self.fused_data_flag == True:
            print('ERROR! This network works with fused data')

        for idx in range(self.input_paths):
            X[idx] = self.Modality_Paths[str(idx)](X[idx])
            if idx==0:
                X_output = X[idx].reshape(-1, prod(tensor(X[idx].shape)[1:]))
            else:
                X_output = cat((X_output, X[idx].reshape(-1, prod(tensor(X[idx].shape)[1:])) ), dim=1)

        del X

        X_output = self.FullyConnected(X_output)

        return X_output


class AgeMapper_input8(nn.Module):
    def __init__(self, 
                fused_data_flag = False,
                original_input_channels = 1,
                channel_number=[32,64,64,64,64],
                ):

        self.fused_data_flag = fused_data_flag

        super(AgeMapper_input8, self).__init__()

        self.input_paths = original_input_channels

        self.Modality_Paths = nn.ModuleDict({})
        for idx in range(self.input_paths):
            self.Modality_Paths[str(idx)] = AgeMapperConvFC1(channel_number=channel_number,
                                                            path_number=idx)

        self.FullyConnected = nn.Sequential()
        input_dimensions = 96

        self.FullyConnected.add_module(
            name = 'FullyConnected_Link',
            module=nn.Linear(
                in_features=input_dimensions * self.input_paths,
                out_features=input_dimensions,
            )
        )
        self.FullyConnected.add_module(
            name = 'ReluActivation_Link',
            module= nn.ReLU()
        )

        self.FullyConnected.add_module(
            name = 'FullyConnected_2',
            module=nn.Linear(
                in_features=input_dimensions,
                out_features=32
            )
        )
        self.FullyConnected.add_module(
            name = 'ReluActivation_2',
            module= nn.ReLU()
        )

        self.FullyConnected.add_module(
            name = 'FullyConnected_1',
            module= nn.Linear(
                in_features=32,
                out_features=1,
            )
        )
        self.FullyConnected.add_module(
            name = 'LinearActivation',
            module= nn.Identity()
        )

    def forward(self, X):
        
        if self.fused_data_flag == True:
            print('ERROR! This network works with fused data')

        for idx in range(self.input_paths):
            X[idx] = self.Modality_Paths[str(idx)](X[idx])
            if idx==0:
                X_output = X[idx]
            else:
                X_output = cat((X_output, X[idx]), dim=1)

        del X

        X_output = self.FullyConnected(X_output)

        return X_output


class AgeMapper_input9(nn.Module):
    def __init__(self, 
                fused_data_flag = False,
                original_input_channels = 1,
                channel_number=[32,64,64,64,64],
                ):

        self.fused_data_flag = fused_data_flag

        super(AgeMapper_input9, self).__init__()

        self.input_paths = original_input_channels

        self.Modality_Paths = nn.ModuleDict({})
        for idx in range(self.input_paths):
            self.Modality_Paths[str(idx)] = AgeMapperConvFC2(channel_number=channel_number,
                                                            path_number=idx)

        self.FullyConnected = nn.Sequential()
        input_dimensions = 32

        self.FullyConnected.add_module(
            name = 'FullyConnected_Link',
            module=nn.Linear(
                in_features=input_dimensions * self.input_paths,
                out_features=input_dimensions,
            )
        )
        self.FullyConnected.add_module(
            name = 'ReluActivation_Link',
            module= nn.ReLU()
        )

        self.FullyConnected.add_module(
            name = 'FullyConnected_1',
            module= nn.Linear(
                in_features=32,
                out_features=1,
            )
        )
        self.FullyConnected.add_module(
            name = 'LinearActivation',
            module= nn.Identity()
        )

    def forward(self, X):
        
        if self.fused_data_flag == True:
            print('ERROR! This network works with fused data')

        for idx in range(self.input_paths):
            X[idx] = self.Modality_Paths[str(idx)](X[idx])
            if idx==0:
                X_output = X[idx]
            else:
                X_output = cat((X_output, X[idx]), dim=1)

        del X

        X_output = self.FullyConnected(X_output)

        return X_output