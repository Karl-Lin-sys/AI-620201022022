"""
ProteinAI - Reinforcement Learning for Sequence Optimization
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Callable

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
NUM_AA = len(AMINO_ACIDS)


class PolicyNetwork(nn.Module):
    """Policy network for sequence action selection."""
    
    def __init__(self, seq_len: int = 50, hidden: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(NUM_AA, 32)
        self.lstm = nn.LSTM(32, hidden, batch_first=True, bidirectional=True)
        self.policy = nn.Linear(hidden * 2, NUM_AA)
        self.value = nn.Linear(hidden * 2, 1)
    
    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(seq)
        h, _ = self.lstm(x)
        logits = self.policy(h)
        value = self.value(h.mean(dim=1))
        return logits, value


class SequenceEnvironment:
    """Environment for protein sequence optimization."""
    
    def __init__(self, target_props: Dict[str, float]):
        self.target = target_props
        self.aa_props = {
            'A': {'hydro': 1.8, 'charge': 0}, 'C': {'hydro': 2.5, 'charge': 0},
            'D': {'hydro': -3.5, 'charge': -1}, 'E': {'hydro': -3.5, 'charge': -1},
            'F': {'hydro': 2.8, 'charge': 0}, 'G': {'hydro': -0.4, 'charge': 0},
            'H': {'hydro': -3.2, 'charge': 0.5}, 'I': {'hydro': 4.5, 'charge': 0},
            'K': {'hydro': -3.9, 'charge': 1}, 'L': {'hydro': 3.8, 'charge': 0},
            'M': {'hydro': 1.9, 'charge': 0}, 'N': {'hydro': -3.5, 'charge': 0},
            'P': {'hydro': -1.6, 'charge': 0}, 'Q': {'hydro': -3.5, 'charge': 0},
            'R': {'hydro': -4.5, 'charge': 1}, 'S': {'hydro': -0.8, 'charge': 0},
            'T': {'hydro': -0.7, 'charge': 0}, 'V': {'hydro': 4.2, 'charge': 0},
            'W': {'hydro': -0.9, 'charge': 0}, 'Y': {'hydro': -1.3, 'charge': 0},
        }
    
    def compute_properties(self, seq: str) -> Dict[str, float]:
        if not seq:
            return {'hydro': 0, 'charge': 0}
        hydro = sum(self.aa_props.get(aa, {'hydro': 0})['hydro'] for aa in seq) / len(seq)
        charge = sum(self.aa_props.get(aa, {'charge': 0})['charge'] for aa in seq)
        return {'hydro': hydro, 'charge': charge}
    
    def reward(self, seq: str) -> float:
        props = self.compute_properties(seq)
        score = 0
        for k, v in self.target.items():
            if k in props:
                score -= abs(props[k] - v)
        return score


class ProteinRL:
    """Reinforcement Learning for protein optimization."""
    
    def __init__(self, seq_len: int = 20, device: str = 'cpu'):
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.policy = PolicyNetwork(seq_len).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.env = SequenceEnvironment({'hydro': 0, 'charge': 0})
    
    def idx_to_seq(self, idx: torch.Tensor) -> str:
        return ''.join([AMINO_ACIDS[i] for i in idx.tolist()])
    
    def sample_action(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)
    
    def generate_sequence(self) -> Tuple[str, float]:
        """Generate optimized sequence using policy."""
        self.policy.eval()
        with torch.no_grad():
            init = torch.randint(0, NUM_AA, (1, self.seq_len), device=self.device)
            logits, _ = self.policy(init)
            action, _ = self.sample_action(logits)
            seq = self.idx_to_seq(action[0])
            reward = self.env.reward(seq)
        return seq, reward
    
    def optimize(self, target: Dict[str, float], steps: int = 50) -> List[Tuple[str, float]]:
        """Optimize sequence for target properties."""
        self.env.target = target
        results = []
        
        for step in range(steps):
            self.policy.train()
            init = torch.randint(0, NUM_AA, (1, self.seq_len), device=self.device)
            logits, value = self.policy(init)
            action, log_prob = self.sample_action(logits)
            seq = self.idx_to_seq(action[0])
            reward = self.env.reward(seq)
            
            # Policy gradient
            advantage = reward - value.item()
            loss = -(log_prob.sum() * advantage) + F.mse_loss(value, torch.tensor([[reward]], device=self.device))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            results.append((seq, reward))
        
        return results


def demo():
    print("=" * 50)
    print("ProteinAI - RL Sequence Optimization")
    print("=" * 50)
    
    rl = ProteinRL()
    
    # Optimize for hydrophilic, neutral sequence
    target = {'hydro': -2.0, 'charge': 0}
    print(f"\nTarget: {target}")
    print("Optimizing...")
    
    results = rl.optimize(target, steps=20)
    best = max(results, key=lambda x: x[1])
    print(f"\nBest sequence: {best[0]}")
    print(f"Reward: {best[1]:.4f}")
    
    props = rl.env.compute_properties(best[0])
    print(f"Properties: hydro={props['hydro']:.2f}, charge={props['charge']:.2f}")
    
    print("=" * 50)
    return rl


if __name__ == "__main__":
    demo()
