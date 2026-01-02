"""
ProteinAI Research Platform - Protein LLM Module
================================================
LLM-based protein function prediction using Transformer architecture.
Demonstrates: NLP, Few-shot learning, Sequence modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict

# Amino acid vocabulary
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {idx: aa for idx, aa in enumerate(AMINO_ACIDS)}
VOCAB_SIZE = len(AMINO_ACIDS) + 2  # +2 for <PAD> and <UNK>
PAD_IDX = len(AMINO_ACIDS)
UNK_IDX = len(AMINO_ACIDS) + 1


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ProteinTransformerEncoder(nn.Module):
    """
    Transformer encoder for protein sequence representation.
    Uses multi-head self-attention to capture long-range dependencies.
    """
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer encoder.
        
        Args:
            src: Input token indices [seq_len, batch_size]
            src_mask: Attention mask
            src_key_padding_mask: Padding mask
        
        Returns:
            Encoded representations [seq_len, batch_size, d_model]
        """
        # Embed and add positional encoding
        x = self.embedding(src) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        output = self.transformer_encoder(
            x, 
            mask=src_mask, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        return self.layer_norm(output)


class ProteinFunctionPredictor(nn.Module):
    """
    Predicts protein function from sequence using LLM-style architecture.
    Supports multi-label classification for function ontology terms.
    """
    
    def __init__(
        self,
        num_functions: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = ProteinTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Pooling and classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_functions)
        )
    
    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict function probabilities for input sequences.
        
        Args:
            src: Token indices [seq_len, batch_size]
            src_key_padding_mask: Padding mask [batch_size, seq_len]
        
        Returns:
            Function logits [batch_size, num_functions]
        """
        # Encode sequence
        encoded = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        # Pool over sequence dimension: [seq_len, batch, d_model] -> [batch, d_model]
        pooled = encoded.permute(1, 2, 0)  # [batch, d_model, seq_len]
        pooled = self.pool(pooled).squeeze(-1)  # [batch, d_model]
        
        # Classify
        return self.classifier(pooled)


class ProteinLLM:
    """
    High-level interface for protein LLM operations.
    Provides tokenization, encoding, and function prediction.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.encoder = ProteinTransformerEncoder().to(self.device)
        self.predictor = ProteinFunctionPredictor().to(self.device)
        
        # Function labels (simplified for demo)
        self.function_labels = [
            'DNA binding', 'RNA binding', 'ATP binding', 'Metal binding',
            'Enzyme', 'Transporter', 'Receptor', 'Kinase', 'Protease',
            'Oxidoreductase', 'Transferase', 'Hydrolase', 'Lyase',
            'Isomerase', 'Ligase', 'Structural', 'Signaling', 'Immunity',
            'Cell adhesion', 'Transcription factor'
        ] + [f'Function_{i}' for i in range(80)]
    
    def tokenize(self, sequence: str) -> torch.Tensor:
        """Convert amino acid sequence to token indices."""
        indices = [AA_TO_IDX.get(aa, UNK_IDX) for aa in sequence.upper()]
        return torch.tensor(indices, dtype=torch.long, device=self.device)
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Get embedding representation of protein sequence."""
        tokens = self.tokenize(sequence).unsqueeze(1)  # [seq_len, 1]
        with torch.no_grad():
            encoded = self.encoder(tokens)
        return encoded.squeeze(1)  # [seq_len, d_model]
    
    def predict_function(self, sequence: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict most likely functions for a protein sequence.
        
        Args:
            sequence: Amino acid sequence string
            top_k: Number of top predictions to return
        
        Returns:
            List of (function_name, probability) tuples
        """
        tokens = self.tokenize(sequence).unsqueeze(1)  # [seq_len, 1]
        
        with torch.no_grad():
            logits = self.predictor(tokens)
            probs = F.softmax(logits, dim=-1).squeeze(0)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, min(top_k, len(self.function_labels)))
        
        predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            predictions.append((self.function_labels[idx], prob))
        
        return predictions
    
    def get_attention_weights(self, sequence: str) -> np.ndarray:
        """Extract attention weights for visualization."""
        tokens = self.tokenize(sequence).unsqueeze(1)
        
        # Get attention from first layer (simplified)
        self.encoder.eval()
        with torch.no_grad():
            embedded = self.encoder.embedding(tokens) * np.sqrt(self.encoder.d_model)
            embedded = self.encoder.pos_encoder(embedded)
            
            # Extract from first encoder layer
            layer = self.encoder.transformer_encoder.layers[0]
            attn_output, attn_weights = layer.self_attn(
                embedded, embedded, embedded,
                need_weights=True
            )
        
        return attn_weights.squeeze().cpu().numpy()
    
    def few_shot_predict(
        self, 
        query_sequence: str,
        support_sequences: List[str],
        support_labels: List[str]
    ) -> Dict[str, float]:
        """
        Few-shot learning: predict function based on similar labeled examples.
        
        Args:
            query_sequence: Sequence to classify
            support_sequences: List of example sequences
            support_labels: Labels for support sequences
        
        Returns:
            Dictionary mapping labels to similarity scores
        """
        # Encode query
        query_emb = self.encode_sequence(query_sequence).mean(dim=0)
        
        # Encode support set and compute similarities
        label_scores: Dict[str, List[float]] = {}
        
        for seq, label in zip(support_sequences, support_labels):
            support_emb = self.encode_sequence(seq).mean(dim=0)
            similarity = F.cosine_similarity(
                query_emb.unsqueeze(0), 
                support_emb.unsqueeze(0)
            ).item()
            
            if label not in label_scores:
                label_scores[label] = []
            label_scores[label].append(similarity)
        
        # Average scores per label
        predictions = {
            label: np.mean(scores) 
            for label, scores in label_scores.items()
        }
        
        return predictions


# Demo function
def demo():
    """Demonstrate protein LLM capabilities."""
    print("=" * 60)
    print("ProteinAI - LLM-based Protein Function Prediction")
    print("=" * 60)
    
    # Initialize model
    llm = ProteinLLM()
    
    # Example protein sequence (insulin)
    insulin_a = "GIVEQCCTSICSLYQLENYCN"
    
    print(f"\nInput Sequence: {insulin_a}")
    print(f"Length: {len(insulin_a)} amino acids")
    
    # Function prediction
    print("\n--- Function Prediction ---")
    predictions = llm.predict_function(insulin_a)
    for func, prob in predictions:
        print(f"  {func}: {prob:.4f}")
    
    # Sequence embedding
    print("\n--- Sequence Embedding ---")
    embedding = llm.encode_sequence(insulin_a)
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding mean: {embedding.mean().item():.4f}")
    
    # Few-shot learning demo
    print("\n--- Few-Shot Learning ---")
    support_seqs = [
        "MALWMRLLPLLALLALWGPDPAAA",  # Insulin-like
        "MKTAYIAKQRQISFVK",  # DNA binding
    ]
    support_labels = ["Hormone", "DNA binding"]
    
    test_seq = "FVNQHLCGSHLVEALYLVCGERGFF"
    few_shot_result = llm.few_shot_predict(test_seq, support_seqs, support_labels)
    print(f"  Query: {test_seq[:20]}...")
    for label, score in sorted(few_shot_result.items(), key=lambda x: -x[1]):
        print(f"    {label}: {score:.4f}")
    
    print("\n" + "=" * 60)
    return llm


if __name__ == "__main__":
    demo()
