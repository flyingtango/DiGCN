import argparse
import torch
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from datasets import get_citation_dataset
from train_eval import run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--gpu-no', type=int, default=0)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=20)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=200)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--recache', action="store_true", help="clean up the old adj data", default=True)
parser.add_argument('--normalize-features', action="store_true", default=True)
parser.add_argument('--adj-type', type=str, default='or')

args = parser.parse_args()

class Net(torch.nn.Module):
    def __init__(self, dataset, cached=True):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

def run_gcn(dataset,gpu_no):
    dataset = get_citation_dataset(dataset, args.alpha, args.recache, args.normalize_features, args.adj_type)
    print("Num of edges ",dataset[0].num_edges)
    val_loss, test_acc, test_std, time = run(dataset, gpu_no, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
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
        val_loss, test_acc, test_std, time = run_gcn(dataset,args.gpu_no)
        result.loc[dataset]['val_loss'] = val_loss
        result.loc[dataset]['test_acc'] = test_acc
        result.loc[dataset]['test_std'] = test_std
        result.loc[dataset]['time'] = time