"""
ProteinAI Research Platform - Main Entry Point
===============================================
Unified interface for all AI modules.
"""

import argparse
from src.protein_llm import ProteinLLM, demo as llm_demo
from src.protein_gcn import ProteinStructureAnalyzer, demo as gcn_demo
from src.protein_diffusion import ProteinDiffusion, demo as diffusion_demo
from src.protein_rl import ProteinRL, demo as rl_demo
from src.protein_utils import analyze_sequence, validate_sequence


def run_demo():
    """Run all module demonstrations."""
    print("\n" + "=" * 70)
    print(" ProteinAI Research Platform - Complete Demo ")
    print("=" * 70 + "\n")
    
    # LLM Demo
    print("\n[1/4] LLM-based Function Prediction")
    print("-" * 40)
    llm_demo()
    
    # GCN Demo
    print("\n[2/4] GCN Structure Analysis")
    print("-" * 40)
    gcn_demo()
    
    # Diffusion Demo
    print("\n[3/4] Diffusion Sequence Generation")
    print("-" * 40)
    diffusion_demo()
    
    # RL Demo
    print("\n[4/4] RL Sequence Optimization")
    print("-" * 40)
    rl_demo()
    
    print("\n" + "=" * 70)
    print(" Demo Complete! ")
    print("=" * 70)


def analyze(sequence: str):
    """Analyze a protein sequence."""
    valid, msg = validate_sequence(sequence)
    if not valid:
        print(f"Error: {msg}")
        return
    
    print(f"\nAnalyzing: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
    print("-" * 40)
    
    # Basic analysis
    props = analyze_sequence(sequence)
    print(f"Length: {props['length']} aa")
    print(f"Molecular Weight: {props['molecular_weight']:.1f} Da")
    print(f"Hydropathy: {props['hydropathy']:.2f}")
    print(f"Polarity: {props['polarity']*100:.1f}%")
    
    # LLM prediction
    print("\nFunction Prediction (LLM):")
    llm = ProteinLLM()
    predictions = llm.predict_function(sequence)
    for func, prob in predictions[:3]:
        print(f"  - {func}: {prob:.3f}")
    
    # Structure analysis
    print("\nStructure Analysis (GCN):")
    analyzer = ProteinStructureAnalyzer()
    structure = analyzer.analyze_structure(sequence)
    for cls, prob in sorted(structure.items(), key=lambda x: -x[1])[:3]:
        print(f"  - {cls}: {prob:.3f}")


def main():
    parser = argparse.ArgumentParser(description="ProteinAI Research Platform")
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--analyze', type=str, help='Analyze sequence')
    parser.add_argument('--generate', type=int, default=0, help='Generate N sequences')
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.analyze:
        analyze(args.analyze)
    elif args.generate > 0:
        print(f"Generating {args.generate} sequences...")
        diffusion = ProteinDiffusion()
        seqs = diffusion.sample(args.generate, 25)
        for i, s in enumerate(seqs, 1):
            print(f"  {i}: {s}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
