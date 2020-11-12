import os.path as osp
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd
import os
import torch
import shutil
import torch_geometric.transforms as T
from torch_geometric.data import Data
from Citation import Citation
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.data import InMemoryDataset


def get_citation_dataset(name, alpha=0.1, recache=False, normalize_features=False, adj_type=None, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)),'data')
    file_path = osp.join(path, name, 'processed')
    if recache == True:
        print("Delete old processed data cache...")
        if osp.exists(file_path):
            shutil.rmtree(file_path)
        os.mkdir(file_path)
        print('Finish cleaning.')
    dataset = Citation(path, name, alpha, adj_type=adj_type)
    print('Finish dataset preprocessing.')
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset

if __name__ == "__main__":
    pass