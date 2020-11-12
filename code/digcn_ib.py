import argparse
import torch
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from DIGCNConv import DIGCNConv
from torch_scatter import scatter_add
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected, to_networkx
from torch.nn import Linear
from datasets import get_citation_dataset
from train_eval import run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--gpu-no', type=int, default=0)
parser.add_argument('--runs', type=int, default=20)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=200)
parser.add_argument('--hidden', type=int, default=32)

parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--recache', action="store_true", help="clean up the old adj data", default=True)
parser.add_argument('--normalize-features', action="store_true", default=True)
parser.add_argument('--adj-type', type=str, default='ib')
args = parser.parse_args()


class InceptionBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(InceptionBlock, self).__init__()
        self.ln = Linear(in_dim, out_dim)
        self.conv1 = DIGCNConv(in_dim, out_dim)
        self.conv2 = DIGCNConv(in_dim, out_dim)
    def reset_parameters(self):
        self.ln.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    def forward(self, x, edge_index, edge_weight, edge_index2, edge_weight2):
        x0 = self.ln(x)
        x1 = self.conv1(x, edge_index, edge_weight)
        x2 = self.conv2(x, edge_index2, edge_weight2)
        return x0, x1, x2

class Sparse_Three_Sum(torch.nn.Module):
    def __init__(self, dataset):
        super(Sparse_Three_Sum, self).__init__()
        self.ib1 = InceptionBlock(dataset.num_features, args.hidden)
        self.ib2 = InceptionBlock(args.hidden, args.hidden)
        self.ib3 = InceptionBlock(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.ib1.reset_parameters()
        self.ib2.reset_parameters()
        self.ib3.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        edge_index2, edge_weight2 = data.edge_index2, data.edge_weight2
        
        x0,x1,x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=args.dropout, training=self.training)
        x1 = F.dropout(x1, p=args.dropout, training=self.training)
        x2 = F.dropout(x2, p=args.dropout, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=args.dropout, training=self.training)

        x0,x1,x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=args.dropout, training=self.training)
        x1 = F.dropout(x1, p=args.dropout, training=self.training)
        x2 = F.dropout(x2, p=args.dropout, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=args.dropout, training=self.training)

        x0,x1,x2 = self.ib3(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=args.dropout, training=self.training)
        x1 = F.dropout(x1, p=args.dropout, training=self.training)
        x2 = F.dropout(x2, p=args.dropout, training=self.training)
        x = x0+x1+x2

        return F.log_softmax(x, dim=1)

class Sparse_Three_Concat(torch.nn.Module):
    def __init__(self, dataset):
        super(Sparse_Three_Concat, self).__init__()
        self.ib1 = InceptionBlock(dataset.num_features, args.hidden)
        self.ib2 = InceptionBlock(args.hidden, args.hidden)
        self.ib3 = InceptionBlock(args.hidden, dataset.num_classes)
        
        self.ln1 = Linear(args.hidden * 3, args.hidden)
        self.ln2 = Linear(args.hidden * 3, args.hidden)
        self.ln3 = Linear(dataset.num_classes * 3, dataset.num_classes)


    def reset_parameters(self):
        self.ib1.reset_parameters()
        self.ib2.reset_parameters()
        self.ib3.reset_parameters()

        self.ln1.reset_parameters()
        self.ln2.reset_parameters()
        self.ln3.reset_parameters()


    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        edge_index2, edge_weight2 = data.edge_index2, data.edge_weight2
        
        x0,x1,x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=args.dropout, training=self.training)
        x1 = F.dropout(x1, p=args.dropout, training=self.training)
        x2 = F.dropout(x2, p=args.dropout, training=self.training)

        x = torch.cat((x0,x1,x2),1)
        x = self.ln1(x)
        
        x = F.dropout(x, p=args.dropout, training=self.training)

        x0,x1,x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=args.dropout, training=self.training)
        x1 = F.dropout(x1, p=args.dropout, training=self.training)
        x2 = F.dropout(x2, p=args.dropout, training=self.training)

        x = torch.cat((x0,x1,x2),1)
        x = self.ln2(x)

        x = F.dropout(x, p=args.dropout, training=self.training)

        x0,x1,x2 = self.ib3(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=args.dropout, training=self.training)
        x1 = F.dropout(x1, p=args.dropout, training=self.training)
        x2 = F.dropout(x2, p=args.dropout, training=self.training)
       
        x = torch.cat((x0,x1,x2),1)
        x = self.ln3(x)

        return F.log_softmax(x, dim=1)

def run_digcn(dataset,gpu_no):
    dataset = get_citation_dataset(dataset, args.alpha, args.recache, args.normalize_features, args.adj_type)
    # Replace Sparse_Three_Sum with Sparse_Three_Concat to test concat
    val_loss, test_acc, test_std, time = run(dataset, gpu_no, Sparse_Three_Sum(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
        args.early_stopping)
    return val_loss, test_acc, test_std, time

if __name__ == '__main__':
    if args.dataset is not None:
        dataset_name = [args.dataset]
    else:
        dataset_name = ['cora_ml','citeseer']
    outputs = ['val_loss', 'test_acc', 'test_std', 'time']                
    result = pd.DataFrame(np.arange(len(outputs)*len(dataset_name), dtype=np.float32).reshape(
        (len(dataset_name), len(outputs))), index=dataset_name, columns=outputs)
    for dataset in dataset_name:
        val_loss, test_acc, test_std, time = run_digcn(dataset,args.gpu_no)
        result.loc[dataset]['val_loss'] = val_loss
        result.loc[dataset]['test_acc'] = test_acc
        result.loc[dataset]['test_std'] = test_std
        result.loc[dataset]['time'] = time

