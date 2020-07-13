import torch.nn as nn
import torch
from hyper_params import *
from auxiliary_funcs import *
from torch.distributions.categorical import Categorical

loss_function = nn.MSELoss()

#testing
# inspired by https://github.com/LeanManager/NLP-PyTorch/blob/master/Character-Level%20LSTM%20with%20PyTorch.ipynb
# TODO: Klassifizierung vs Regression
# TODO: Prediction, Training Error!
class LSTM_fixed(nn.Module):
    def __init__(self, input_size=input_dim, hidden_layer_size=hidden_layer_size):
        # output size has to be the number of bins for first loc vec component + for the second
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=0.01)
        # # predict the two components
        self.linear = nn.Linear(hidden_layer_size, 2)

        # self.dis_layers = nn.ModuleList([nn.LSTM(hidden_layer_size, hidden_layer_size, 1, batch_first=True )
        #                                  for _ in range(num_layers - 1)])
        # self.dis_layers.insert(0, nn.LSTM(input_size, hidden_layer_size, 1, batch_first=True))
        #
        # #self.top_gen_layer = nn.LSTM(hidden_layer_size, hidden_layer_size, 1, batch_first=True)
        # self.gen_layers = nn.ModuleList([nn.LSTM(hidden_layer_size * 2, hidden_layer_size, 1, batch_first=True)
        #                                  for _ in range(num_layers - 1)])
        # self.gen_layers.append(nn.LSTM(hidden_layer_size, hidden_layer_size, 1, batch_first=True))
        #
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x, hc):
        x, (h, c) = self.lstm(x, hc)
        #m = nn.LayerNorm(x.size()[1:])
        #x = m(x)
        out = self.linear(x)
        return out, (h, c)

    def forward_ey(self, x, states):
        dis_states, gen_states = states[: num_layers], states[num_layers:][::-1]
        seq_len = x.size()[1]

        # discriminative network applies the lstm layers and saves the results
        # and hidden state
        layer_results = [(self.dis_layers[0](x, dis_states[0]))]
        for l in range(1, num_layers):
            layer_results.append(
                self.dis_layers[l](
                    self.dropout((layer_results[-1][0])),
                    dis_states[l]
                )
            )
        # don't need this here but maybe later
        # dis_out = (layer_results[-1][0] + 1.0) / 2.0

        # generative network gets as input its the previous input plus an input from the discriminative layer
        top_layer = num_layers - 1
        layer_results.append(self.gen_layers[top_layer]((layer_results[-1][0]), gen_states[top_layer]))
        for l in range(top_layer - 1, -1, -1):
            # next = torch.cat((layer_results[-1][0], layer_results[l][0]), 2)
            # print(next.size())
            layer_results.append(
                self.gen_layers[l](
                    self.dropout((torch.cat((layer_results[-1][0], layer_results[l][0]), 2))),
                    gen_states[l]
                )
            )

        # transform the output into bins
        gen_out = layer_results[-1][0]
        out = self.linear(gen_out)

        # get the whole hidden state history
        next_states = [results[1] for results in layer_results]
        return out, next_states

    def predict(self, x, hc):

        # x, (h, c) = self.lstm(x, hc)
        # m = nn.LayerNorm(x.size()[1:])
        # x = m(x)
        # out = self.linear(x)
        return self.forward(x, hc)

    def init_hidden(self, batch_size, num_layers, hidden_layer_size):
        ''' Initializes hidden state '''
        weight = next(self.parameters()).data
        return (weight.new(num_layers, batch_size, hidden_layer_size).zero_(),
                weight.new(num_layers, batch_size, hidden_layer_size).zero_())

class LSTM_multi_modal(nn.Module):
    def __init__(self, input_size=input_dim, hidden_layer_size=hidden_layer_size):
        # output size has to be the number of bins for first loc vec component + for the second
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        #self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)

        self.linear1 = nn.Linear(hidden_layer_size, num_angle_bins)
        self.linear2 = nn.Linear(hidden_layer_size, num_speed_bins)

        # self.hidden_state = self.init_hidden(batch_size, num_layers)

        # self.bottom_dis_layer = nn.LSTM(input_size, hidden_layer_size, 1, batch_first=True)
        self.dis_layers = nn.ModuleList([nn.LSTM(hidden_layer_size, hidden_layer_size, 1, batch_first=True )
                           for _ in range(num_layers - 1)])
        self.dis_layers.insert(0, nn.LSTM(input_size, hidden_layer_size, 1, batch_first=True))

        #self.top_gen_layer = nn.LSTM(hidden_layer_size, hidden_layer_size, 1, batch_first=True)
        self.gen_layers = nn.ModuleList([nn.LSTM(hidden_layer_size * 2, hidden_layer_size, 1, batch_first=True)
                           for _ in range(num_layers - 1)])
        self.gen_layers.append(nn.LSTM(hidden_layer_size, hidden_layer_size, 1, batch_first=True))

        self.dropout = nn.Dropout(0.2)
        #self.layernorm_dis = nn.LayerNorm(hidden_layer_size)
        #self.layernorm_gen = nn.LayerNorm(hidden_layer_size * 2)


    def forward_old(self, x, hc):
        x, (h, c) = self.lstm(x, hc)
        m = nn.LayerNorm(x.size()[1:])
        x = m(x)
        angle_out = self.linear1(x)
        speed_out = self.linear2(x)
        return angle_out, speed_out, (h, c)

    # from Moritz Maxeiner, eyolfsdottirs method
    def forward(self, x, states):
        dis_states, gen_states = states[: num_layers], states[num_layers:][::-1]
        seq_len = x.size()[1]

        # discriminative network applies the lstm layers and saves the results
        # and hidden state
        layer_results = [(self.dis_layers[0](x, dis_states[0]))]
        for l in range(1, num_layers):
            layer_results.append(
                self.dis_layers[l](
                    self.dropout((layer_results[-1][0])),
                    dis_states[l]
                )
            )


        # don't need this here but maybe later
        # dis_out = (layer_results[-1][0] + 1.0) / 2.0

        # generative network gets as input its the previous input plus an input from the discriminative layer
        top_layer = num_layers - 1
        layer_results.append(self.gen_layers[top_layer]((layer_results[-1][0]), gen_states[top_layer]))
        for l in range(top_layer - 1, -1, -1):
            #next = torch.cat((layer_results[-1][0], layer_results[l][0]), 2)
            #print(next.size())
            layer_results.append(
                self.gen_layers[l](
                    self.dropout((torch.cat((layer_results[-1][0], layer_results[l][0]), 2))),
                    gen_states[l]
                )
            )

        # transform the output into bins
        gen_out = layer_results[-1][0]
        angle_out = self.linear1(gen_out)
        speed_out = self.linear2(gen_out)

        # get the whole hidden state history
        next_states = [results[1] for results in layer_results]
        return angle_out, speed_out, next_states

    def predict(self, x, states):
        # works only with batchsize = sequencesize = 1
        angle_out, speed_out, states = self.forward(x, states)
        angle_out = angle_out.view(num_angle_bins)
        speed_out = speed_out.view(num_speed_bins)
        # print("-----------RAW SCORES------------")
        # print(angle_out)
        # print(speed_out)
        m = nn.Softmax(0)
        angle_prob = m(angle_out)
        speed_prob = m(speed_out)
        # print("---------------SOFTMAX PROBABILITIES ---------------")
        # print(angle_prob)
        # print(speed_prob)
        angle_bin = Categorical(angle_prob).sample()
        speed_bin = Categorical(speed_prob).sample()
        # print("angle bin predicted: ", angle_bin)
        # print("speed bin predicted: ", speed_bin)
        angle_value = angle_bin_to_value(angle_bin, angle_min, angle_max, num_angle_bins, 0.001)
        speed_value = speed_bin_to_value(speed_bin, speed_min, speed_max, num_speed_bins, 0.001)
        # print("angle value:", angle_value)
        # print("speed value:", speed_value)
        return (angle_value, speed_value), states

    def predict_old(self, x, hc):
        # works only with batchsize = sequencesize = 1
        x, (h, c) = self.lstm(x, hc)
        angle_out = self.linear1(x)
        speed_out = self.linear2(x)
        angle_out = angle_out.view(num_angle_bins)
        speed_out = speed_out.view(num_speed_bins)
        m = nn.Softmax(0)
        angle_prob = m(angle_out)
        speed_prob = m(speed_out)
        print(angle_prob)
        print(speed_prob)
        angle_bin = Categorical(angle_prob).sample()
        speed_bin = Categorical(speed_prob).sample()
        angle_value = angle_bin_to_value(angle_bin, angle_min, angle_max, num_angle_bins, 0.001)
        speed_value = speed_bin_to_value(speed_bin, speed_min, speed_max, num_speed_bins, 0.001)
        return (angle_value, speed_value), (h, c)

    def confidence(self,pred):
        highest=0
        for x in pred:
           if x>highest:
               highest=x
        return highest

    def init_hidden(self, batch_size, num_layers, hidden_layer_size):
        ''' Initializes hidden state '''
        weight = next(self.parameters()).data
        return (weight.new(num_layers, batch_size, self.hidden_layer_size).zero_(),
                weight.new(num_layers, batch_size, self.hidden_layer_size).zero_())

    def simulate(self, initial_pose, initial_loc_sensory, frames):
        pass

        # for i in range(frames):
