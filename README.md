# 🧬 ProteinAI Research Platform

AI医学研究のためのタンパク質解析・生成プラットフォーム

## 概要

このプロジェクトは、最先端のAI技術を用いてタンパク質の解析・生成を行う統合プラットフォームです。

### 主要技術

| 技術 | 用途 | モジュール |
|------|------|-----------|
| **LLM/Transformer** | タンパク質機能予測 | `protein_llm.py` |
| **GCN (Graph Neural Network)** | 構造解析 | `protein_gcn.py` |
| **Diffusion Model** | 配列生成 (AIGC) | `protein_diffusion.py` |
| **Reinforcement Learning** | 配列最適化 | `protein_rl.py` |
| **Flask** | Webダッシュボード | `app.py` |

## プロジェクト構造

```
ai_medical_research/
├── src/
│   ├── protein_llm.py      # LLMベース機能予測
│   ├── protein_gcn.py      # GCN構造解析
│   ├── protein_diffusion.py # Diffusion配列生成
│   ├── protein_rl.py       # 強化学習最適化
│   └── protein_utils.py    # ユーティリティ
├── templates/
│   └── index.html          # Webダッシュボード
├── app.py                  # Flask Webアプリ
├── main.py                 # CLI エントリポイント
└── requirements.txt        # 依存パッケージ
```

## インストール

```bash
# 依存パッケージのインストール
pip install -r requirements.txt

# または個別に
pip install torch numpy flask
```

## 使用方法

### コマンドライン

```bash
# デモ実行
python main.py --demo

# 配列解析
python main.py --analyze MVLSPADKTNVKAAWGKVGAH

# 配列生成
python main.py --generate 5
```

### Webダッシュボード

```bash
python app.py
# http://localhost:5000 でアクセス
```

## 技術詳細

### 1. LLM-based Function Prediction

Transformerアーキテクチャを用いたタンパク質機能予測:

- **Multi-head Self-Attention**: 長距離依存性の捕捉
- **Positional Encoding**: 配列位置情報のエンコード
- **Few-shot Learning**: 少数サンプルからの予測

```python
from src.protein_llm import ProteinLLM

llm = ProteinLLM()
predictions = llm.predict_function("MVLSPADKTNVKAAWGKVGAH")
```

### 2. Graph Convolutional Network

分子グラフ表現による構造解析:

- **Graph Attention Layer**: 動的な隣接ノード重み付け
- **Residue Embedding**: アミノ酸特徴量の学習
- **Structure Classification**: 二次構造クラス分類

```python
from src.protein_gcn import ProteinStructureAnalyzer

analyzer = ProteinStructureAnalyzer()
structure = analyzer.analyze_structure("AEEAAKAFIEAIKALK")
```

### 3. Diffusion Model

Denoising Diffusion Probabilistic Model (DDPM):

- **De Novo Generation**: ゼロからの配列生成
- **Conditional Sampling**: 部分配列の補完
- **Mutation Generation**: 既存配列の変異体生成

```python
from src.protein_diffusion import ProteinDiffusion

diffusion = ProteinDiffusion()
new_sequences = diffusion.sample(n=3, length=30)
```

### 4. Reinforcement Learning

Policy Gradientによる配列最適化:

- **PPO-style Training**: 安定した学習
- **Property Optimization**: 疎水性・電荷の最適化
- **Reward Engineering**: 目標特性への報酬設計

```python
from src.protein_rl import ProteinRL

rl = ProteinRL()
results = rl.optimize({'hydro': -2.0, 'charge': 0}, steps=50)
```

## 関連技術・ツール

このプロジェクトは以下のツール・技術に関連しています:

- **AlphaFold**: タンパク質構造予測
- **RFdiffusion**: Diffusionベース構造生成
- **ProteinMPNN**: 配列設計
- **ESM**: タンパク質言語モデル

## ライセンス

MIT License

## 著者

AI Medical Research Portfolio Project
