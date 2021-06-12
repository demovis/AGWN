# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.

import json

import pygsp
import numpy as np
import pandas as pd
import networkx as nx
from torch_sparse import spspmm, spmm
from texttable import Texttable
from sklearn.preprocessing import normalize
import torch
from scipy import sparse
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch_geometric.nn as gnn
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import math

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


gt = np.load("./gt_county2.npy")
data = np.load("./data_county2.npy")

adjacent = np.load("./adjacent_county_2.npy")
edge_index = np.load("edge_index_traffic_county_2.npy")
weight = np.load("weight_county_2.npy")

device = torch.device("cuda")
gt=torch.tensor(gt, dtype=torch.float)
data=torch.tensor(data, dtype=torch.float)
#edge_index=torch.tensor(edge_index, dtype=torch.int64).to(device)
weight=torch.tensor(weight, dtype=torch.float)

data[:,:,:,0]=(data[:,:,:,0]-data[:,:,:,0].mean())/data[:,:,:,0].var()
data[:,:,:,1]=(data[:,:,:,1]-data[:,:,:,1].mean())/data[:,:,:,1].var()

data=data.view(3940, 455, -1)



class WaveletSparsifier(object):
    """
    Object to sparsify the wavelet coefficients for a graph.
    """
    def __init__(self, graph, scale=1, approximation_order=3, tolerance=10**-4):
        """
        :param graph: NetworkX graph object.
        :param scale: Kernel scale length parameter.
        :param approximation_order: Chebyshev polynomial order.
        :param tolerance: Tolerance for sparsification.
        """
        self.graph = graph
        self.pygsp_graph = pygsp.graphs.Graph(nx.adjacency_matrix(self.graph))
        self.pygsp_graph.estimate_lmax()
        self.scales = [-scale, scale]
        self.approximation_order = approximation_order
        self.tolerance = tolerance
        self.phi_matrices = []

    def calculate_wavelet(self):
        """
        Creating sparse wavelets.
        :return remaining_waves: Sparse matrix of attenuated wavelets.
        """
        impulse = np.eye(self.graph.number_of_nodes(), dtype=int)
        wavelet_coefficients = pygsp.filters.approximations.cheby_op(self.pygsp_graph,
                                                                     self.chebyshev,
                                                                     impulse)
        wavelet_coefficients[wavelet_coefficients < self.tolerance] = 0
        ind_1, ind_2 = wavelet_coefficients.nonzero()
        n_count = self.graph.number_of_nodes()
        remaining_waves = sparse.csr_matrix((wavelet_coefficients[ind_1, ind_2], (ind_1, ind_2)),
                                            shape=(n_count, n_count),
                                            dtype=np.float32)
        return remaining_waves

    def normalize_matrices(self):
        """
        Normalizing the wavelet and inverse wavelet matrices.
        """
        print("\nNormalizing the sparsified wavelets.\n")
        for i, phi_matrix in enumerate(self.phi_matrices):
            self.phi_matrices[i] = normalize(self.phi_matrices[i], norm='l1', axis=1)

    def calculate_density(self):
        """
        Calculating the density of the sparsified wavelet matrices.
        """
        wavelet_density = len(self.phi_matrices[0].nonzero()[0])/(self.graph.number_of_nodes()**2)
        wavelet_density = str(round(100*wavelet_density, 2))
        inverse_wavelet_density = len(self.phi_matrices[1].nonzero()[0])/(self.graph.number_of_nodes()**2)
        inverse_wavelet_density = str(round(100*inverse_wavelet_density, 2))
        print("Density of wavelets: "+wavelet_density+"%.")
        print("Density of inverse wavelets: "+inverse_wavelet_density+"%.\n")

    def calculate_all_wavelets(self):
        """
        Graph wavelet coefficient calculation.
        """
        print("\nWavelet calculation and sparsification started.\n")
        for i, scale in enumerate(self.scales):
            self.heat_filter = pygsp.filters.Heat(self.pygsp_graph,
                                                  tau=[scale])
            self.chebyshev = pygsp.filters.approximations.compute_cheby_coeff(self.heat_filter,
                                                                              m=self.approximation_order)
            sparsified_wavelets = self.calculate_wavelet()          
            self.phi_matrices.append(sparsified_wavelets)
        self.normalize_matrices()
        self.calculate_density()

edge_rec=[]
for i in range(len(edge_index[0])):
    edge_rec.append((edge_index[0][i],edge_index[1][i]))

graph = nx.from_edgelist(edge_rec)

sparsifier = WaveletSparsifier(graph)

sparsifier.calculate_all_wavelets()



class GraphWaveletLayer(torch.nn.Module):
    """
    Abstract Graph Wavelet Layer class.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param ncount: Number of nodes.
    :param device: Device to train on.
    """
    def __init__(self, in_channels, out_channels, ncount, device):
        super(GraphWaveletLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncount = ncount
        self.device = device
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining diagonal filter matrix (Theta in the paper) and weight matrix.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.diagonal_weight_indices = torch.LongTensor([[node for node in range(self.ncount)],
                                                         [node for node in range(self.ncount)]])

        self.diagonal_weight_indices = self.diagonal_weight_indices.to(self.device)
        self.diagonal_weight_filter = torch.nn.Parameter(torch.Tensor(self.ncount, 1))

    def init_parameters(self):
        """
        Initializing the diagonal filter and the weight matrix.
        """
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.9, 1.1)
        torch.nn.init.xavier_uniform_(self.weight_matrix)

class SparseGraphWaveletLayer(GraphWaveletLayer):
    """
    Sparse Graph Wavelet Layer Class.
    """
    def forward(self, phi_indices, phi_values, phi_inverse_indices,
                phi_inverse_values, feature_indices, feature_values, dropout):
        """
        Forward propagation pass.
        :param phi_indices: Sparse wavelet matrix index pairs.
        :param phi_values: Sparse wavelet matrix values.
        :param phi_inverse_indices: Inverse wavelet matrix index pairs.
        :param phi_inverse_values: Inverse wavelet matrix values.
        :param feature_indices: Feature matrix index pairs.
        :param feature_values: Feature matrix values.
        :param dropout: Dropout rate.
        :return dropout_features: Filtered feature matrix extracted.
        """
        rescaled_phi_indices, rescaled_phi_values = spspmm(phi_indices,
                                                           phi_values,
                                                           self.diagonal_weight_indices,
                                                           self.diagonal_weight_filter.view(-1),
                                                           self.ncount,
                                                           self.ncount,
                                                           self.ncount)

        phi_product_indices, phi_product_values = spspmm(rescaled_phi_indices,
                                                         rescaled_phi_values,
                                                         phi_inverse_indices,
                                                         phi_inverse_values,
                                                         self.ncount,
                                                         self.ncount,
                                                         self.ncount)

        filtered_features = spmm(feature_indices,
                                 feature_values,
                                 self.ncount,
                                 self.in_channels,
                                 self.weight_matrix)

        localized_features = spmm(phi_product_indices,
                                  phi_product_values,
                                  self.ncount,
                                  self.ncount,
                                  filtered_features)

        dropout_features = torch.nn.functional.dropout(torch.nn.functional.relu(localized_features),
                                                       training=self.training,
                                                       p=dropout)
        return dropout_features

class DenseGraphWaveletLayer(GraphWaveletLayer):
    """
    Dense Graph Wavelet Layer Class.
    """
    def forward(self, phi_indices, phi_values, phi_inverse_indices, phi_inverse_values, features):
        """
        Forward propagation pass.
        :param phi_indices: Sparse wavelet matrix index pairs.
        :param phi_values: Sparse wavelet matrix values.
        :param phi_inverse_indices: Inverse wavelet matrix index pairs.
        :param phi_inverse_values: Inverse wavelet matrix values.
        :param features: Feature matrix.
        :return localized_features: Filtered feature matrix extracted.
        """
        rescaled_phi_indices, rescaled_phi_values = spspmm(phi_indices,
                                                           phi_values,
                                                           self.diagonal_weight_indices,
                                                           self.diagonal_weight_filter.view(-1),
                                                           self.ncount,
                                                           self.ncount,
                                                           self.ncount)

        phi_product_indices, phi_product_values = spspmm(rescaled_phi_indices,
                                                         rescaled_phi_values,
                                                         phi_inverse_indices,
                                                         phi_inverse_values,
                                                         self.ncount,
                                                         self.ncount,
                                                         self.ncount)

        filtered_features = torch.mm(features, self.weight_matrix)

        localized_features = spmm(phi_product_indices,
                                  phi_product_values,
                                  self.ncount,
                                  self.ncount,
                                  filtered_features)

        return localized_features



dropout=0.3
# gt=torch.tensor(gt, dtype=torch.float).to(device)
# data=torch.tensor(data, dtype=torch.float).to(device)
# #edge_index=torch.tensor(edge_index, dtype=torch.int64).to(device)
# weight=torch.tensor(weight, dtype=torch.float).to(device)
class GraphWaveletNeuralNetwork(torch.nn.Module):
    """
    Graph Wavelet Neural Network class.
    For details see: Graph Wavelet Neural Network.
    Bingbing Xu, Huawei Shen, Qi Cao, Yunqi Qiu, Xueqi Cheng. ICLR, 2019
    :param args: Arguments object.
    :param ncount: Number of nodes.
    :param feature_number: Number of features.
    :param class_number: Number of classes.
    :param device: Device used for training.
    """
    def __init__(self, ncount, feature_number, class_number, device):
        super(GraphWaveletNeuralNetwork, self).__init__()
        self.filters=32
        self.ncount = ncount
        self.feature_number = feature_number
        self.class_number = class_number
        self.device = device
        self.setup_layers()

    def setup_layers(self):
        """
        Setting up a sparse and a dense layer.
        """
        self.convolution_1 = SparseGraphWaveletLayer(self.feature_number,
                                                     self.filters,
                                                     self.ncount,
                                                     self.device)

        self.convolution_2 = DenseGraphWaveletLayer(self.filters,
                                                    self.class_number,
                                                    self.ncount,
                                                    self.device)
        self.linear_1 = nn.Linear(32, 1)
        self.linear_2 = nn.Linear(455, 5)
        self.act = nn.ReLU()
    def forward(self, phi_indices, phi_values, phi_inverse_indices,
                phi_inverse_values, feature_indices, feature_values):
        """
        Forward propagation pass.
        :param phi_indices: Sparse wavelet matrix index pairs.
        :param phi_values: Sparse wavelet matrix values.
        :param phi_inverse_indices: Inverse wavelet matrix index pairs.
        :param phi_inverse_values: Inverse wavelet matrix values.
        :param feature_indices: Feature matrix index pairs.
        :param feature_values: Feature matrix values.
        :param predictions: Predicted node label vector.
        """
        deep_features_1 = self.convolution_1(phi_indices,
                                             phi_values,
                                             phi_inverse_indices,
                                             phi_inverse_values,
                                             feature_indices,
                                             feature_values,
                                             dropout)

        # deep_features_2 = self.convolution_2(phi_indices,
        #                                      phi_values,
        #                                      phi_inverse_indices,
        #                                      phi_inverse_values,
        #                                      deep_features_1)
        # print(deep_features_1.shape)
        # print("===")
        predictions=self.act(self.linear_1(deep_features_1))
        predictions=predictions.view(1,-1)
        predictions=self.act(self.linear_2(predictions))
        #predictions = torch.nn.functional.log_softmax(deep_features_1, dim=1)
        return predictions.squeeze()

class GWNNTrainer(object):
    """
    Graph Wavelet Neural Network Trainer object.
    :param args: Arguments object.
    :param sparsifier: Sparsifier object with sparse wavelet filters.
    :param features: Sparse feature matrix.
    :param target: Target vector.
    """
    def __init__(self, sparsifier, features, target):
        
        self.sparsifier = sparsifier
        self.features = features
        self.target = target
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.setup_logs()
        self.setup_features()
        self.setup_model()
        # self.train_test_split()

    # def setup_logs(self):
    #     """
    #     Creating a log for performance measurements.
    #     """
    #     self.logs = dict()
    #     self.logs["parameters"] =  vars(self.args)
    #     self.logs["performance"] = [["Epoch", "Loss"]]
    #     self.logs["training_time"] = [["Epoch", "Seconds"]]

    # def update_log(self, loss, epoch):
    #     """
    #     Updating the logs.
    #     :param loss:
    #     :param epoch:
    #     """
    #     self.epochs.set_description("GWNN (Loss=%g)" % round(loss.item(), 4))
    #     self.logs["performance"].append([epoch, round(loss.item(), 4)])
    #     self.logs["training_time"].append([epoch, time.time()-self.time])

    def setup_features(self):
        """
        Defining PyTorch tensors for sparse matrix multiplications.
        """
        self.ncount = self.sparsifier.phi_matrices[0].shape[0]
        self.feature_number = self.features.shape[1]
        self.class_number = max(self.target)+1
        self.target = torch.LongTensor(self.target).to(self.device)
        self.feature_indices = torch.LongTensor([self.features.row, self.features.col])
        self.feature_indices = self.feature_indices.to(self.device)
        self.feature_values = torch.FloatTensor(self.features.data).view(-1).to(self.device)
        self.phi_indices = torch.LongTensor(self.sparsifier.phi_matrices[0].nonzero()).to(self.device)
        self.phi_values = torch.FloatTensor(self.sparsifier.phi_matrices[0][self.sparsifier.phi_matrices[0].nonzero()])
        self.phi_values = self.phi_values.view(-1).to(self.device)
        self.phi_inverse_indices = torch.LongTensor(self.sparsifier.phi_matrices[1].nonzero()).to(self.device)
        self.phi_inverse_values = torch.FloatTensor(self.sparsifier.phi_matrices[1][self.sparsifier.phi_matrices[1].nonzero()])
        self.phi_inverse_values = self.phi_inverse_values.view(-1).to(self.device)

    def setup_model(self):
        """
        Creating a log.
        """
        self.model = GraphWaveletNeuralNetwork(self.args,
                                               self.ncount,
                                               self.feature_number,
                                               self.class_number,
                                               self.device)
        self.model = self.model.to(self.device)

def feature_reader(features):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param feature_path: Path to the JSON file.
    :return features: Feature sparse COO matrix.
    """
    #features = json.load(open(path))
    index_1 = [int(k) for k, v in features.items() for fet in v]
    index_2 = [int(fet) for k, v in features.items() for fet in v]
    values = [1.0]*len(index_1)
    nodes = [int(k) for k, v in features.items()]
    node_count = max(nodes)+1
    feature_count = max(index_2)+1
    features = sparse.coo_matrix((values, (index_1, index_2)),
                                 shape=(node_count, feature_count),
                                 dtype=np.float32)
    return features
tmp=data[0]
traffic_features={}
for i in range(455):
    traffic_features[i]=tmp[i].cpu().numpy().tolist()

traffic_features_update= feature_reader(traffic_features)

ncount = sparsifier.phi_matrices[0].shape[0]
feature_number = traffic_features_update.shape[1]
# class_number = max(self.target)+1
# target = torch.LongTensor(self.target).to(self.device)
feature_indices = torch.LongTensor([traffic_features_update.row, traffic_features_update.col])
feature_indices = feature_indices.to(device)
feature_values = torch.FloatTensor(traffic_features_update.data).view(-1).to(device)
phi_indices = torch.LongTensor(sparsifier.phi_matrices[0].nonzero()).to(device)
phi_values = torch.FloatTensor(sparsifier.phi_matrices[0][sparsifier.phi_matrices[0].nonzero()])
phi_values = phi_values.view(-1).to(device)
phi_inverse_indices = torch.LongTensor(sparsifier.phi_matrices[1].nonzero()).to(device)
phi_inverse_values = torch.FloatTensor(sparsifier.phi_matrices[1][sparsifier.phi_matrices[1].nonzero()])
phi_inverse_values = phi_inverse_values.view(-1).to(device)


edge_index=torch.tensor(edge_index, dtype=torch.int64).to(device)
data_list=[]
for i in range(len(data)):
    data_total = Data(x=data[i], edge_index=edge_index,edge_attr=weight, y=gt[i])
    data_list.append(data_total)

# train_loader = DataLoader(data_list[:7000], batch_size=2,shuffle=True)
# valid_loader = DataLoader(data_list[7000:8000], batch_size=2,shuffle=True)


l = len(data_list)
train_loader = DataLoader(data_list[:3150], batch_size=5,shuffle=True)
valid_loader = DataLoader(data_list[3150:3545], batch_size=5,shuffle=True)
test_data   =  DataLoader(data_list[3545:], batch_size=5,shuffle=True)


#
# train_data = LoadData(data_torch[:int(l * 0.8)], gt_torch[:int(l * 0.8)])
# val_data = LoadData(data_torch[int(l * 0.8):int(l * 0.9)], gt_torch[int(l * 0.8):int(l * 0.9)])
# test_data = LoadData(data_torch[int(l * 0.9):], gt_torch[int(l * 0.9):])
# feature_number = 5
class_number=455*2


my_net=GraphWaveletNeuralNetwork(
                                               ncount,
                                               feature_number,
                                               class_number,
                                               device)
my_net = my_net.to(device)
criterion = nn.MSELoss()
# my_net.load_state_dict(torch.load("GWNN_train_new.pkl"))

optimizer = optim.Adam(params=my_net.parameters())
Epoch=200

test=True
plot=True

train_min=25000.0
val_min=25000.0

criterion = nn.MSELoss()
criterion_mae_loss = nn.L1Loss()
def MAPELoss(output, target):
    return torch.mean(torch.abs((target - output) / target))  
if test == False:
    for epoch in range(Epoch):
        epoch_loss = 0.0
        valid_loss =0.0
        start_time = time.time()
        for data_total in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            my_net.zero_grad()

            prediction = my_net (phi_indices,
                         phi_values,
                         phi_inverse_indices,
                         phi_inverse_values,
                         feature_indices,
                         feature_values)
            #print(predict_value[:10])

            target = data_total.y
            #print(target[:10])
            #print("===")
            # print(target.shape)
            # print(prediction.shape)
            target=target.to(device)
            loss = criterion(prediction,target)

            epoch_loss += loss.item()

            loss.backward()

            optimizer.step()
        end_time = time.time()

        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, epoch_loss / len(train_loader),(end_time-start_time)/60))
        if epoch_loss / len(train_loader) < train_min:
            train_min=epoch_loss / len(train_loader)

            # torch.save(my_net.state_dict(), "GWNN_train_new.pkl")
            print("RMSE: {:02.4f}".format(math.sqrt(epoch_loss) / len(train_loader)))

        my_net.eval()
        with torch.no_grad():

            total_mse_loss = 0.0
            total_mae_loss = 0.0
            total_mape_loss = 0.0


            for data_total in valid_loader:

                predict_value = my_net (phi_indices,
                    phi_values,
                    phi_inverse_indices,
                    phi_inverse_values,
                    feature_indices,
                    feature_values)
                target = data_total.y

                # loss = criterion(predict_value,target)
                # valid_loss += loss
                # predict_value = my_net(x, W,device)
                target=target.to(device)

                mse_loss = criterion(predict_value, target)
                mae_loss=criterion_mae_loss(predict_value, target)
                mape_loss=MAPELoss(predict_value, target)
                total_mse_loss += mse_loss.item()
                total_mae_loss += mae_loss.item()
                total_mape_loss += mape_loss.item()

            print("Min loss: {:02.4f}, Val Loss: {:02.4f}"\
            .format(val_min, total_mse_loss / len(valid_loader)))
            if total_mse_loss/ len(valid_loader) < val_min:
                val_min=total_mse_loss/ len(valid_loader)
            #             MAE = np.mean(np.abs(y-y_hat))
            # MAPE = np.mean(np.abs((y - y_hat) / y)) * 100
                torch.save(my_net.state_dict(), "GWNN_county2.pkl")
                print("RMSE: {:02.4f}".format(math.sqrt(total_mse_loss)/ len(valid_loader)))
                print("MAE: {:02.4f}".format( total_mae_loss/ len(valid_loader)))
                print("MAPE: {:02.4f}".format( total_mape_loss/ len(valid_loader)))

if test ==True and plot==False:
    my_net=GraphWaveletNeuralNetwork(
                                                   ncount,
                                                   feature_number,
                                                   class_number,
                                                   device)
    my_net = my_net.to(device)

    my_net.load_state_dict(torch.load("GWNN_county2.pkl"))


    batch_size=5
    # test_data=DataLoader(data_list[8000:8595], batch_size=5,shuffle=True)

    criterion = nn.MSELoss()
    criterion_mae_loss = nn.L1Loss()

    def MAPELoss(output, target):
      return torch.mean(torch.abs((target - output) / target))





    with torch.no_grad():

        total_mse_loss = []
        total_mae_loss = []
        total_mape_loss = []


        for data_total in test_data:

            predict_value = my_net (phi_indices,
                    phi_values,
                    phi_inverse_indices,
                    phi_inverse_values,
                    feature_indices,
                    feature_values)
            target = data_total.y
            target=target.to(device)
            mse_loss = criterion(predict_value, target)
            mae_loss= criterion_mae_loss(predict_value, target)
            mape_loss=MAPELoss(predict_value, target)

            # total_mse_loss += mse_loss.item()
            # total_mae_loss += mae_loss.item()
            # total_mape_loss += mape_loss.item()


            print("RMSE: {:02.4f}".format(math.sqrt(mse_loss.item())/batch_size))
            print("MAE: {:02.4f}".format( mae_loss.item()/batch_size))
            print("MAPE: {:02.4f}".format( mape_loss.item() /batch_size))
            total_mse_loss.append(math.sqrt(mse_loss.item())/batch_size)
            total_mae_loss.append(mae_loss.item()/batch_size)
            total_mape_loss.append(mape_loss.item() /batch_size)
            print("======")

    mse_np= np.array(total_mse_loss)
    mae_np= np.array(total_mae_loss)
    mape_np= np.array(total_mape_loss)

    rmse_np_mean=np.mean(mse_np)
    mae_np_mean=np.mean(mae_np)
    mape_np_mean=np.mean(mape_np)
    # print(mse_np)
    print(rmse_np_mean,mae_np_mean,mape_np_mean)

    rmse_np_std=math.sqrt(np.std(mse_np, ddof=1))
    mae_np_std=math.sqrt(np.std(mae_np, ddof=1))
    mape_np_std=np.std(mape_np, ddof=1)

    print(rmse_np_std,mae_np_std,mape_np_std)

if test ==True and plot==True:
    my_net=GraphWaveletNeuralNetwork(
                                                   ncount,
                                                   feature_number,
                                                   class_number,
                                                   device)
    my_net = my_net.to(device)

    my_net.load_state_dict(torch.load("GWNN_county2.pkl"))


    batch_size=5
    test_data=DataLoader(data_list[:50], batch_size=5,shuffle=True)

    criterion = nn.MSELoss()
    criterion_mae_loss = nn.L1Loss()

    def MAPELoss(output, target):
      return torch.mean(torch.abs((target - output) / target))


    with torch.no_grad():

        for data_total in test_data:
            predict_value = my_net (phi_indices,
                    phi_values,
                    phi_inverse_indices,
                    phi_inverse_values,
                    feature_indices,
                    feature_values)
            target = data_total.y
            print(predict_value)
            print(target)
            print("====")