import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from view_hdf import get_locomotion_vec, Guppy_Calculator, Guppy_Dataset
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
from guppy_model import LSTM_fixed, LSTM_multi_modal
import sys
import copy
from hyper_params import *

torch.manual_seed(1)

# get the files for 4, 6 and 8 guppys
trainpath = "guppy_data/live_female_female/train/" if live_data else "guppy_data/couzin_torus/train/"
files = [join(trainpath, f) for f in listdir(trainpath) if isfile(join(trainpath, f)) and f.endswith(".hdf5") ]
files.sort()
num_files = len(files) // 8
files = files[-3:]
print(files)

torch.set_default_dtype(torch.float64)

# now we use a regression model, just predict the absolute values of linear speed and angular turn
# so we need squared_error loss

if output_model == "multi_modal":
    model = LSTM_multi_modal()
    loss_function = nn.CrossEntropyLoss()
else:
    model = LSTM_fixed()
    loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)
# training

dataset = Guppy_Dataset(files, 0, num_guppy_bins, num_wall_rays, livedata=live_data, output_model=output_model)
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

epochs = 12
for i in range(epochs):
    try:
        #h = model.init_hidden(batch_size, num_layers, hidden_layer_size)
        states = [model.init_hidden(batch_size, 1, hidden_layer_size) for _ in range(num_layers * 2)]
        loss = 0
        confidence=conf1=conf2=0
        for inputs, targets in dataloader:
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            model.zero_grad()
            #h = tuple([each.data for each in h])
            states = [tuple([each.data for each in s]) for s in states]

            if output_model == "multi_modal":
                targets = targets.type(torch.LongTensor)
               # angle_pred, speed_pred, h = model.forward(inputs, h)
                angle_pred, speed_pred, states = model.forward(inputs, states)
                #print(angle_pred.size())
                #print(speed_pred.size())

                #print(targets.size())
                angle_pred = angle_pred.view(angle_pred.shape[0] * angle_pred.shape[1], -1)
                speed_pred = speed_pred.view(speed_pred.shape[0] * speed_pred.shape[1], -1)
                #print(angle_pred.size())
                #print(speed_pred.size())
                targets = targets.view(targets.shape[0] * targets.shape[1], 2)
                #print(targets.size())
                angle_targets = targets[:, 0]
                speed_targets = targets[:, 1]
                # print("------ANGLE SCORES-------")
                # print(angle_pred)
                # print("------ANGLE TARGETS -------")
                # print(angle_targets)
                # with torch.no_grad():
                # print("------SPEED PROBS-------")
                # with torch.no_grad():
                #     print(nn.Softmax(0)(speed_pred[0]))
                #     print("------SPEED TARGETS -------")
                #     print(speed_targets)
                #accuracy=model.accuracy(angle_pred,angle_targets)
                #print(nn.Softmax(0)(angle_pred[0]))
                #print("die Länge davon ist " + str(len(angle_pred)) + str(len(speed_pred)))
                for x in range(len(angle_pred)):
                    conf1+=model.confidence(nn.Softmax(0)(angle_pred[x]))
                    conf2+=model.confidence(nn.Softmax(0)(speed_pred[x]))
                confidence+=(conf1+conf2)/2
                #print(len(angle_targets))
                #print("Angle target looks like this: " + str(angle_targets))
                #print("Speed target looks like this: " + str(speed_targets))
                #print(conf1)
                conf1=conf2=0
                loss1 = loss_function(angle_pred, angle_targets)
                loss2 = loss_function(speed_pred, speed_targets)
                loss += loss1 + loss2


            else:
                #prediction, states = model.forward(inputs, states)
                prediction, h = model.forward(inputs, h)
                loss += loss_function(prediction, targets)

        loss = loss / dataset.length
        #accuracy=accuracy/dataset.length
        #print(dataset.length)
        #print("The confidence after the loop is" + str(confidence))
        confidence=float(confidence/(2992*dataset.length/batch_size))
        loss.backward()
        optimizer.step()

    except KeyboardInterrupt:
        if input("Do you want to save the model trained so far? y/n") == "y":
            torch.save(model.state_dict(), network_path + f".epochs{i}")
            print("network saved at " + network_path + f".epochs{i}")
        sys.exit(0)

    print(f'epoch: {i:3} loss: {loss.item():10.10f}')
    print("confidence: " + str(confidence))

torch.save(model.state_dict(), network_path + f".epochs{epochs}")
print("network saved at " + network_path + f".epochs{epochs}")


