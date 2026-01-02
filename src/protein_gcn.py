"""
ProteinAI Research Platform - Graph Convolutional Network Module
================================================================
GCN-based protein structure analysis using molecular graph representations.
Demonstrates: Graph Neural Networks, Molecular modeling, Structure prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict


class GraphConvolutionLayer(nn.Module):
    """
    Graph Convolution Layer for molecular graphs.
    Implements message passing between connected atoms.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Glorot initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GCN layer.
        
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Linear transformation
        support = torch.mm(x, self.weight)
        
        # Aggregate neighbor information
        output = torch.sparse.mm(adj, support) if adj.is_sparse else torch.mm(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer with multi-head attention.
    Learns to weight neighbor contributions dynamically.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        # Per-head transformations
        self.W = nn.Parameter(torch.FloatTensor(num_heads, in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * out_features, 1))
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention mechanism.
        
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Attention-weighted node features
        """
        N = x.size(0)
        
        # Transform features for each head
        h = torch.einsum('nf,hfo->hno', x, self.W)  # [heads, nodes, out_features]
        
        # Compute attention coefficients
        a_input = torch.cat([
            h.unsqueeze(2).expand(-1, -1, N, -1),
            h.unsqueeze(1).expand(-1, N, -1, -1)
        ], dim=-1)  # [heads, N, N, 2*out_features]
        
        e = self.leakyrelu(torch.einsum('hnmf,hfk->hnm', a_input, self.a).squeeze(-1))
        
        # Mask non-adjacent nodes
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.unsqueeze(0) > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to features
        h_prime = torch.einsum('hnm,hmo->hno', attention, h)
        
        if self.concat:
            return h_prime.permute(1, 0, 2).reshape(N, -1)  # [N, heads*out_features]
        else:
            return h_prime.mean(dim=0)  # [N, out_features]


class ProteinGCN(nn.Module):
    """
    Graph Convolutional Network for protein structure analysis.
    Encodes protein molecular graph into meaningful representations.
    """
    
    def __init__(
        self,
        node_features: int = 64,
        hidden_dims: List[int] = [128, 256, 128],
        num_classes: int = 10,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        self.use_attention = use_attention
        
        # Input embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GCN/GAT layers
        self.conv_layers = nn.ModuleList()
        in_dim = hidden_dims[0]
        
        for out_dim in hidden_dims[1:]:
            if use_attention:
                self.conv_layers.append(GraphAttentionLayer(in_dim, out_dim // 4, num_heads=4))
            else:
                self.conv_layers.append(GraphConvolutionLayer(in_dim, out_dim))
            in_dim = out_dim
        
        # Readout and classification
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
    
    def forward(
        self, 
        node_features: torch.Tensor, 
        adjacency: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GCN.
        
        Args:
            node_features: Node features [num_nodes, node_features]
            adjacency: Adjacency matrix [num_nodes, num_nodes]
            batch_indices: Optional batch assignment for nodes
        
        Returns:
            tuple: (class_logits, node_embeddings)
        """
        # Encode nodes
        x = self.node_encoder(node_features)
        
        # Graph convolutions
        for conv in self.conv_layers:
            x = F.relu(conv(x, adjacency))
        
        # Graph-level readout (mean pooling)
        if batch_indices is not None:
            # Batch-aware pooling
            graph_emb = self._batch_pool(x, batch_indices)
        else:
            graph_emb = x.mean(dim=0, keepdim=True)
        
        # Classification
        logits = self.classifier(graph_emb)
        
        return logits, x
    
    def _batch_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Pool node embeddings per graph in batch."""
        num_graphs = batch.max().item() + 1
        pooled = []
        for i in range(num_graphs):
            mask = batch == i
            pooled.append(x[mask].mean(dim=0))
        return torch.stack(pooled)


class ProteinGraphBuilder:
    """
    Builds molecular graph representations from protein sequences.
    Creates nodes for residues and edges for contacts.
    """
    
    # Amino acid properties
    AA_PROPERTIES = {
        'A': {'hydrophobic': 1, 'polar': 0, 'charge': 0, 'size': 0.3},
        'C': {'hydrophobic': 0.5, 'polar': 0.5, 'charge': 0, 'size': 0.4},
        'D': {'hydrophobic': 0, 'polar': 1, 'charge': -1, 'size': 0.5},
        'E': {'hydrophobic': 0, 'polar': 1, 'charge': -1, 'size': 0.6},
        'F': {'hydrophobic': 1, 'polar': 0, 'charge': 0, 'size': 0.8},
        'G': {'hydrophobic': 0.5, 'polar': 0.5, 'charge': 0, 'size': 0.1},
        'H': {'hydrophobic': 0, 'polar': 1, 'charge': 0.5, 'size': 0.7},
        'I': {'hydrophobic': 1, 'polar': 0, 'charge': 0, 'size': 0.6},
        'K': {'hydrophobic': 0, 'polar': 1, 'charge': 1, 'size': 0.7},
        'L': {'hydrophobic': 1, 'polar': 0, 'charge': 0, 'size': 0.6},
        'M': {'hydrophobic': 1, 'polar': 0, 'charge': 0, 'size': 0.7},
        'N': {'hydrophobic': 0, 'polar': 1, 'charge': 0, 'size': 0.5},
        'P': {'hydrophobic': 0.5, 'polar': 0.5, 'charge': 0, 'size': 0.4},
        'Q': {'hydrophobic': 0, 'polar': 1, 'charge': 0, 'size': 0.6},
        'R': {'hydrophobic': 0, 'polar': 1, 'charge': 1, 'size': 0.8},
        'S': {'hydrophobic': 0, 'polar': 1, 'charge': 0, 'size': 0.3},
        'T': {'hydrophobic': 0, 'polar': 1, 'charge': 0, 'size': 0.4},
        'V': {'hydrophobic': 1, 'polar': 0, 'charge': 0, 'size': 0.5},
        'W': {'hydrophobic': 1, 'polar': 0, 'charge': 0, 'size': 1.0},
        'Y': {'hydrophobic': 0.5, 'polar': 0.5, 'charge': 0, 'size': 0.9},
    }
    
    def __init__(self, contact_threshold: float = 8.0, feature_dim: int = 64):
        self.contact_threshold = contact_threshold
        self.feature_dim = feature_dim
        self.property_keys = ['hydrophobic', 'polar', 'charge', 'size']
    
    def sequence_to_graph(
        self, 
        sequence: str,
        positions: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert protein sequence to graph representation.
        
        Args:
            sequence: Amino acid sequence
            positions: Optional 3D coordinates [seq_len, 3]
        
        Returns:
            tuple: (node_features, adjacency_matrix)
        """
        n = len(sequence)
        
        # Build node features
        node_features = self._compute_node_features(sequence)
        
        # Build adjacency matrix
        if positions is not None:
            adjacency = self._contact_adjacency(positions)
        else:
            # Use sequence adjacency (k-nearest neighbors in sequence)
            adjacency = self._sequence_adjacency(n, k=3)
        
        return node_features, adjacency
    
    def _compute_node_features(self, sequence: str) -> torch.Tensor:
        """Compute feature vector for each residue."""
        features = []
        
        for aa in sequence.upper():
            props = self.AA_PROPERTIES.get(aa, self.AA_PROPERTIES['G'])
            
            # Base properties
            base = [props[k] for k in self.property_keys]
            
            # One-hot encoding (20 AAs)
            one_hot = [0.0] * 20
            aa_idx = 'ACDEFGHIKLMNPQRSTVWY'.find(aa)
            if aa_idx >= 0:
                one_hot[aa_idx] = 1.0
            
            # Positional features (to be embedded)
            feat = base + one_hot
            
            # Pad to feature_dim
            while len(feat) < self.feature_dim:
                feat.append(0.0)
            
            features.append(feat[:self.feature_dim])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _sequence_adjacency(self, n: int, k: int = 3) -> torch.Tensor:
        """Create adjacency matrix based on sequence proximity."""
        adj = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(max(0, i-k), min(n, i+k+1)):
                if i != j:
                    adj[i, j] = 1.0
        
        # Add self-loops
        adj += torch.eye(n)
        
        # Normalize (D^-1/2 A D^-1/2)
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj = D_inv_sqrt @ adj @ D_inv_sqrt
        
        return adj
    
    def _contact_adjacency(self, positions: np.ndarray) -> torch.Tensor:
        """Create adjacency based on 3D contact distances."""
        n = len(positions)
        adj = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < self.contact_threshold:
                    adj[i, j] = 1.0
        
        # Normalize
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj = D_inv_sqrt @ adj @ D_inv_sqrt
        
        return adj


class ProteinStructureAnalyzer:
    """
    High-level interface for protein structure analysis using GCN.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.graph_builder = ProteinGraphBuilder()
        self.gcn = ProteinGCN().to(self.device)
        
        # Structure classes
        self.structure_classes = [
            'Alpha helix', 'Beta sheet', 'Loop', 'Turn',
            'Coil', 'Helix-turn-helix', 'Beta barrel',
            'Alpha/Beta', 'All alpha', 'All beta'
        ]
    
    def analyze_structure(self, sequence: str) -> Dict[str, float]:
        """
        Analyze protein structure from sequence.
        
        Args:
            sequence: Amino acid sequence
        
        Returns:
            Dictionary of structure class probabilities
        """
        # Build graph
        node_features, adjacency = self.graph_builder.sequence_to_graph(sequence)
        node_features = node_features.to(self.device)
        adjacency = adjacency.to(self.device)
        
        # Predict
        self.gcn.eval()
        with torch.no_grad():
            logits, _ = self.gcn(node_features, adjacency)
            probs = F.softmax(logits, dim=-1).squeeze()
        
        return {
            cls: prob.item() 
            for cls, prob in zip(self.structure_classes, probs)
        }
    
    def get_residue_embeddings(self, sequence: str) -> np.ndarray:
        """Get learned embeddings for each residue."""
        node_features, adjacency = self.graph_builder.sequence_to_graph(sequence)
        node_features = node_features.to(self.device)
        adjacency = adjacency.to(self.device)
        
        self.gcn.eval()
        with torch.no_grad():
            _, embeddings = self.gcn(node_features, adjacency)
        
        return embeddings.cpu().numpy()
    
    def compare_structures(self, seq1: str, seq2: str) -> float:
        """
        Compare structural similarity between two sequences.
        
        Returns:
            Similarity score between 0 and 1
        """
        emb1 = torch.tensor(self.get_residue_embeddings(seq1).mean(axis=0))
        emb2 = torch.tensor(self.get_residue_embeddings(seq2).mean(axis=0))
        
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return similarity.item()


def demo():
    """Demonstrate GCN protein structure analysis."""
    print("=" * 60)
    print("ProteinAI - GCN-based Structure Analysis")
    print("=" * 60)
    
    analyzer = ProteinStructureAnalyzer()
    
    # Example sequences
    alpha_helix_seq = "AEEAAKAFIEAIKALK"  # Alpha helix-forming
    beta_sheet_seq = "TVTVTVEVKVRVKVK"   # Beta sheet-forming
    
    print("\n--- Alpha Helix Sequence ---")
    print(f"Sequence: {alpha_helix_seq}")
    results = analyzer.analyze_structure(alpha_helix_seq)
    for cls, prob in sorted(results.items(), key=lambda x: -x[1])[:5]:
        print(f"  {cls}: {prob:.4f}")
    
    print("\n--- Beta Sheet Sequence ---")
    print(f"Sequence: {beta_sheet_seq}")
    results = analyzer.analyze_structure(beta_sheet_seq)
    for cls, prob in sorted(results.items(), key=lambda x: -x[1])[:5]:
        print(f"  {cls}: {prob:.4f}")
    
    print("\n--- Structure Comparison ---")
    similarity = analyzer.compare_structures(alpha_helix_seq, beta_sheet_seq)
    print(f"Similarity score: {similarity:.4f}")
    
    print("\n--- Residue Embeddings ---")
    embeddings = analyzer.get_residue_embeddings(alpha_helix_seq)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Mean embedding norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    
    print("\n" + "=" * 60)
    return analyzer


if __name__ == "__main__":
    demo()
