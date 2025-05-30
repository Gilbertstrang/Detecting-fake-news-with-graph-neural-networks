# Fake News Detection using Graph Neural Networks

This is a fake news detection system that leverages **Graph Neural Networks (GNNs)** to analyze information propagation patterns in social networks. This project tries to demonstrates that **how information spreads** is as important as **what information says** for detecting misinformation.


## Architecture Overview

### Information Cascade Modeling
- **Nodes**: Individual tweets/retweets with 8 structural features
- **Edges**: Information flow (retweet relationships)
- **Graph Structure**: Directed acyclic graphs (DAGs) representing news propagation

### Model Components
1. **Node Embedding**: Linear projection of features to hidden dimension
2. **GCN Layers**: 2-layer Graph Convolutional Network
3. **Graph Pooling**: Combined mean and max pooling
4. **Classifier**: Binary classification (fake vs. real)

## Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install -r requirements.txt
```

### Installation

```bash
git clone <repository-url>
cd fake_news_gnn
pip install -r requirements.txt
```

### Basic Usage

#### Train Structure-only Model
```bash
python cascade_gnn_no_text.py
```

#### Train Structure + BERT Model
```bash
python cascade_gnn_with_text.py
```

#### Analyze Dataset
```bash
python analyze_dataset.py
```

#### Generate Visualizations
```bash
python figures.py
```

## 📁 Project Structure

```
fake_news_gnn/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── TECHNICAL_REPORT.md           # Detailed technical documentation
├── cascade_gnn_no_text.py        # Structure-only GNN model
├── cascade_gnn_with_text.py      # Structure + BERT GNN model
├── analyze_dataset.py            # Dataset analysis tools
├── extract_bert.py               # BERT feature extraction
├── figures.py                     # Visualization generation
├── model_insight_summary.py      # Model analysis tools
├── data/                          # Dataset directory
│   ├── raw/                      # Raw data files
│   ├── processed/                # Processed graph data
│   └── *.pkl                     # Cached features and embeddings
├── models/                        # Trained model checkpoints
├── results/                       # Experiment results and plots
├── src/                          # Source modules
│   ├── upfd_cascade_dataset.py   # Dataset loader
│   ├── upfd_dataset.py           # Alternative dataset formats
│   └── simple_gcn.py             # Baseline models
└── __pycache__/                  # Python cache files
```

## 🔧 Dependencies

### Core Requirements
```
torch >= 2.0.0
torch-geometric >= 2.3.0
transformers >= 4.30.0
sentence-transformers
pandas >= 1.5.0
scikit-learn >= 1.2.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
numpy >= 1.21.0
tqdm
```

### Optional (for development)
```
jupyter
ipython
```

## 📈 Dataset Information

- **Source**: UPFD (User Preference-aware Fake news Detection) framework
- **Total Articles**: ~1,000 news articles
- **Total Tweets**: ~50,000 tweets/retweets
- **Balance**: 1:1 ratio (fake vs. real)
- **Features**: 8 structural features per node + optional 384D BERT embeddings

### Node Features (8D)
1. Time since news publication
2. Node degree in cascade
3. Cascade depth
4. User follower count
5. Retweet count
6. Reply count
7. User verification status
8. Local influence score

## 🎛️ Configuration

### Model Hyperparameters
```python
# Structure-only Model
input_dim = 8
hidden_dim = 64
num_layers = 2
dropout = 0.2
learning_rate = 0.001

# Training
batch_size = 64
max_epochs = 30
patience = 5
```

### BERT Configuration
```python
# Text Model
bert_model = 'all-MiniLM-L6-v2'
embedding_dim = 384
combined_features = 8 + 384  # Structure + BERT
```

## 📊 Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area Under the ROC Curve
- **Training Time**: Wall-clock training duration
- **Parameter Count**: Total trainable parameters


## 🔄 Reproducing Results

### Full Experiment Pipeline
```bash
# 1. Analyze dataset
python analyze_dataset.py

# 2. Extract BERT features (optional)
python extract_bert.py

# 3. Train both models
python cascade_gnn_no_text.py
python cascade_gnn_with_text.py

# 4. Generate analysis plots
python figures.py

# 5. Model insights
python model_insight_summary.py
```

## 📋 Limitations & Future Work

### Current Limitations
- Small dataset size (303 cascades)
- Platform-specific (Twitter focused)
- Static graph analysis
- Balanced dataset assumption




## 🙏 Acknowledgments

- UPFD
- PyTorch Geometric team
- Hugging Face Transformers
