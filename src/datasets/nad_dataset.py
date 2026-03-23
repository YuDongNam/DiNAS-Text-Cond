import os
import json
import torch
from torch.utils.data import random_split, Dataset
import torch_geometric.utils
import torch.nn.functional as F
from src.datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule
from src import utils

def parse_graph(graph_str):
    """Parses custom graph string format."""
    lines = graph_str.strip().split('\n')
    nodes = {}
    edges = []
    
    # 1st line is usually header: ##ResNetBasicBlock_CIFAR##
    for line in lines[1:]:
        if ':' in line and '->' not in line:
            idx, op = line.split(':', 1)
            nodes[int(idx)] = op.strip()
        elif '->' in line:
            src, dst = line.split('->')
            edges.append((int(src), int(dst)))
            
    # Sorted by node index
    node_list = [nodes[i] for i in sorted(nodes.keys())]
    return node_list, edges

class NADDataset(Dataset):
    def __init__(self, data_file, embeddings_file=None):
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)
        self.filename = os.path.join(base_path, data_file)
        
        self.embeddings_dict = None
        if embeddings_file:
            emb_path = os.path.join(base_path, embeddings_file)
            if os.path.exists(emb_path):
                print(f"Loading text embeddings from {emb_path}")
                self.embeddings_dict = torch.load(emb_path)
            else:
                print(f"Warning: Embeddings file {emb_path} not found. Fallback to dummy 768-dim embeddings.")
        else:
            print("No text embedding file specified. Using dummy 768-dim embeddings.")
            
        self.data_rows = []
        with open(self.filename, 'r') as f:
            for line in f:
                if not line.strip(): continue
                self.data_rows.append(json.loads(line))
                
        # Build vocabulary across both parent and child graphs
        self.vocab = {}
        for row in self.data_rows:
            p_nodes, _ = parse_graph(row['parent_graph'])
            c_nodes, _ = parse_graph(row['child_graph'])
            for op in p_nodes + c_nodes:
                if op not in self.vocab:
                    self.vocab[op] = len(self.vocab)
                    
        # We calculate max_n_nodes and store it to pad the parent structures
        max_n = 0
        for row in self.data_rows:
            p_n = len(parse_graph(row['parent_graph'])[0])
            c_n = len(parse_graph(row['child_graph'])[0])
            max_n = max(max_n, p_n, c_n)
        # Hardcoded to 110 as requested (max in dataset is 106)
        self.max_n_nodes = 110
        
        self.num_classes = len(self.vocab)
        print(f"Loaded NAD Triplet Dataset. Total examples: {len(self.data_rows)}. Vocab size: {self.num_classes}. Max nodes: {self.max_n_nodes}")

    def __len__(self):
        return len(self.data_rows)

    def _get_graph_data(self, nodes, edges, encode_dense=False):
        n = len(nodes)
        X_idx = [self.vocab[op] for op in nodes]
        X = F.one_hot(torch.tensor(X_idx, dtype=torch.long), num_classes=self.num_classes).float()
        
        if not encode_dense:
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1.0  # index 1 means 'edge exists', 0 means 'no edge'
            return X, edge_index, edge_attr, n
        else:
            # Pad dense X
            pad_size = self.max_n_nodes - n
            X_dense = F.pad(X, (0, 0, 0, pad_size))
            
            # Form dense E
            E_dense = torch.zeros((self.max_n_nodes, self.max_n_nodes, 2), dtype=torch.float)
            E_dense[:, :, 0] = 1.0  # Default to 'no edge'
            for src, dst in edges:
                E_dense[src, dst, 0] = 0.0
                E_dense[src, dst, 1] = 1.0
            return X_dense, E_dense, n

    def __getitem__(self, idx):
        row = self.data_rows[idx]
        
        p_nodes, p_edges = parse_graph(row['parent_graph'])
        c_nodes, c_edges = parse_graph(row['child_graph'])
        
        X_p_dense, E_p_dense, n_p = self._get_graph_data(p_nodes, p_edges, encode_dense=True)
        X_c, edge_idx_c, edge_attr_c, n_c = self._get_graph_data(c_nodes, c_edges, encode_dense=False)
        
        # Text embedding
        text = row['text']
        if self.embeddings_dict is not None and text in self.embeddings_dict:
            y = self.embeddings_dict[text].float()
        else:
            y = torch.randn(768).float() # Dummy 768-D representation
            
        y = y.unsqueeze(0) # (1, 768)
        
        num_nodes_c = torch.tensor([n_c], dtype=torch.long)
        
        data = torch_geometric.data.Data(
            x=X_c, 
            edge_index=edge_idx_c, 
            edge_attr=edge_attr_c,
            y=y, 
            idx=idx, 
            n_nodes=num_nodes_c,
            X_parent=X_p_dense,
            E_parent=E_p_dense
        )
        return data

class NADDataModule(MolecularDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dataset_name = cfg.dataset.datadir
        self.embeddings_file = getattr(cfg.dataset, 'embeddings_file', None)
        
    def prepare_data(self) -> None:
        pass
        
    def setup(self, stage=None):
        from collections import defaultdict
        import numpy as np
        from torch.utils.data import Subset
        
        graphs = NADDataset(self.dataset_name, self.embeddings_file)
        
        # 1. Group indices by dataset attribute for stratified splitting
        dataset_indices = defaultdict(list)
        for i, row in enumerate(graphs.data_rows):
            ds_name = row.get('dataset', 'unknown')
            dataset_indices[ds_name].append(i)
            
        train_indices = []
        val_indices = []
        test_indices = []
        
        # 2. Random but deterministic generation to ensure repeatability
        rng = np.random.default_rng(1234)
        
        for ds_name, indices in dataset_indices.items():
            rng.shuffle(indices)
            total_ds = len(indices)
            
            # Exact 10% ratios per sub-dataset as previously documented
            test_len = int(total_ds * 0.1)
            val_len = int(total_ds * 0.1)
            train_len = total_ds - test_len - val_len
            
            train_indices.extend(indices[:train_len])
            val_indices.extend(indices[train_len:train_len+val_len])
            test_indices.extend(indices[train_len+val_len:])
            
        train_dataset = Subset(graphs, train_indices)
        val_dataset = Subset(graphs, val_indices)
        test_dataset = Subset(graphs, test_indices)
        
        print(f'Dataset sizes: train {len(train_dataset)}, val {len(val_dataset)}, test {len(test_dataset)} (Stratified)')
        
        datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        super().prepare_datas(datasets)

class NADDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        graphs = NADDataset(cfg.dataset.datadir, getattr(cfg.dataset, 'embeddings_file', None))
        self.atom_encoder = graphs.vocab
        self.atom_decoder = {v: k for k, v in self.atom_encoder.items()}
        self.num_atom_types = len(self.atom_encoder)
        
        self.max_n_nodes = graphs.max_n_nodes
        datamodule.setup()
        
        self.n_nodes = datamodule.node_counts(self.max_n_nodes + 1)
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
        
        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        super().compute_input_output_dims(datamodule, extra_features, domain_features)
        
        # Add parent graph feature augmentation to input channel dimensions
        self.input_dims['X'] += self.output_dims['X']
        self.input_dims['E'] += self.output_dims['E']
