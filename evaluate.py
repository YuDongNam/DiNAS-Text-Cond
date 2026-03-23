import networkx as nx
import json
import torch
import torch.nn.functional as F

class DiNASEvaluator:
    def __init__(self, training_data_path, vocab_dict):
        """
        vocab_dict: Dictionary mapping node text names to integer IDs.
                    e.g. {'input': 0, 'output': 1, 'Conv2d': 2, ...}
        """
        self.vocab_dict = vocab_dict
        self.inv_vocab = {v: k for k, v in vocab_dict.items()}
        self.training_graphs = self._load_training_graphs(training_data_path)
        
    def _load_training_graphs(self, filepath):
        """Build string representation of all target child graphs in training set to check Novelty."""
        train_graphs = set()
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                c_nodes, c_edges = self._parse_graph(data['child_graph'])
                str_rep = self._graph_to_string(c_nodes, c_edges)
                train_graphs.add(str_rep)
        return train_graphs
        
    def _parse_graph(self, graph_str):
        lines = graph_str.strip().split('\n')
        nodes = {}
        edges = []
        for line in lines[1:]: # Skip header
            if ':' in line and '->' not in line:
                idx, op = line.split(':', 1)
                nodes[int(idx)] = op.strip()
            elif '->' in line:
                src, dst = line.split('->')
                edges.append((int(src), int(dst)))
        
        # Sorted to ensure node mapping is aligned
        node_list = [nodes[i] for i in sorted(nodes.keys())]
        return node_list, edges

    def _graph_to_string(self, nodes, edges):
        """Canonicalize graph to a string for uniqueness/novelty checks."""
        # Simple representation: list of nodes and list of edges
        node_str = ",".join(nodes)
        edge_str = "|".join([f"{u}->{v}" for u, v in sorted(edges)])
        return f"{node_str}||{edge_str}"

    def decode_tensors_to_graph(self, X_tensor, E_tensor):
        """
        Decodes output tensors `X_tensor` (node types) and `E_tensor` (edges) into NetworkX Graph.
        X_tensor: (N, C) probabilities or one-hot
        E_tensor: (N, N, 2) probabilities or one-hot (index 1 means edge)
        Returns: (nodes_list, edges_list, nx.DiGraph)
        """
        if len(X_tensor.shape) == 2:
            node_ids = torch.argmax(X_tensor, dim=-1).tolist()
        else:
            node_ids = X_tensor.tolist()
            
        nodes = [self.inv_vocab.get(i, 'Unknown') for i in node_ids]
        
        if len(E_tensor.shape) == 3:
            adj = torch.argmax(E_tensor, dim=-1) # (N, N) where 1 means edge exists
        else:
            adj = E_tensor
            
        edges = []
        n = len(nodes)
        for i in range(n):
            for j in range(n):
                if adj[i, j].item() == 1:
                    edges.append((i, j))
                    
        G = nx.DiGraph()
        for i, node in enumerate(nodes):
            G.add_node(i, op=node)
        G.add_edges_from(edges)
        
        return nodes, edges, G

    def check_validity(self, G):
        """
        Validity criteria:
        1. Must be a Directed Acyclic Graph (DAG) with no cycles.
        2. Must have exactly one node labeled 'input' (with in-degree 0).
        3. Must have exactly one node labeled 'output' (with out-degree 0).
        """
        # 1. DAG Check
        if not nx.is_directed_acyclic_graph(G):
            return False
            
        # 2 & 3. Input / Output Checks
        input_nodes = [n for n, d in G.in_degree() if d == 0]
        output_nodes = [n for n, d in G.out_degree() if d == 0]
        
        if len(input_nodes) != 1 or len(output_nodes) != 1:
            return False
            
        in_op = G.nodes[input_nodes[0]]['op']
        out_op = G.nodes[output_nodes[0]]['op']
        
        if in_op != 'input' or out_op != 'output':
            return False
            
        return True

    def estimate_latency(self, nodes, edges):
        """
        Dummy Latency Estimator:
        Currently, returns an arbitrary computation based on # of heavy operations (like Conv2d/Linear).
        Integrate with proper NASBench-101/201 predictor here later.
        """
        latency = 0.0
        for node in nodes:
            if 'Conv2d' in node:
                latency += 1.5
            elif 'Linear' in node:
                latency += 0.8
            elif 'Pool' in node:
                latency += 0.5
            else:
                latency += 0.1 # Relu, BN, etc.
        
        # Penalty for more edges (communications)
        latency += len(edges) * 0.05
        return latency

    def evaluate_batch(self, X_tensors, E_tensors):
        """
        Evaluate a batch of generated graphs.
        X_tensors: List of node tensors or batched tensor
        E_tensors: List of edge tensors or batched tensor
        Returns a dict of metrics.
        """
        generated_graphs = []
        valid_graphs = []
        
        # 1. Decode and check validity
        batch_size = len(X_tensors)
        for X, E in zip(X_tensors, E_tensors):
            nodes, edges, G = self.decode_tensors_to_graph(X, E)
            str_rep = self._graph_to_string(nodes, edges)
            generated_graphs.append(str_rep)
            
            if self.check_validity(G):
                valid_graphs.append({'nodes': nodes, 'edges': edges, 'str': str_rep})
                
        validity = len(valid_graphs) / batch_size if batch_size > 0 else 0.0
        
        # 2. Uniqueness (across valid graphs)
        unique_strs = set([g['str'] for g in valid_graphs])
        uniqueness = len(unique_strs) / len(valid_graphs) if valid_graphs else 0.0
        
        # 3. Novelty (Valid unique graphs not in training set)
        novel_graphs = [s for s in unique_strs if s not in self.training_graphs]
        novelty = len(novel_graphs) / len(unique_strs) if unique_strs else 0.0
        
        # 4. Latency
        latencies = [self.estimate_latency(g['nodes'], g['edges']) for g in valid_graphs]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        metrics = {
            'Validity': validity,
            'Uniqueness': uniqueness,
            'Novelty': novelty,
            'Avg_Latency_ms': avg_latency,
            'Num_Generated': batch_size,
            'Num_Valid': len(valid_graphs)
        }
        return metrics

# Mock Usage Test
if __name__ == '__main__':
    print("Testing DiNASEvaluator setup... (using dummy values)")
    # Since we can't fully instantiate dataset locally easily due to torch errors, we mock the dictionary
    dummy_vocab = {'input': 0, 'output': 1, 'Conv2d(out_channels=C,kernel_size=3,stride=1)': 2, 'ReLU': 3}
    evaluator = DiNASEvaluator("NAD_triplet_dataset.jsonl", dummy_vocab)
    print(f"Loaded {len(evaluator.training_graphs)} training graphs for Novelty checks.")
    
    # Create a dummy DAG graph tensor representation that is valid
    # Nodes: [input, Conv2d..., ReLU, output]
    X_dummy = torch.tensor([[1,0,0,0], [0,0,1,0], [0,0,0,1], [0,1,0,0]]) 
    
    # Edges: 0->1, 1->2, 2->3
    E_dummy = torch.zeros((4,4,2))
    E_dummy[:,:,0] = 1 # default no edge
    edges = [(0,1), (1,2), (2,3)]
    for u, v in edges:
        E_dummy[u,v,0] = 0
        E_dummy[u,v,1] = 1
        
    metrics = evaluator.evaluate_batch([X_dummy], [E_dummy])
    print("Dummy Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")
