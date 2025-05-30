import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import DataLoader, Batch, Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os
import pickle
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CascadeGNNWithNLP(nn.Module):
    
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 dropout=0.2, num_classes=2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
       
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
     
        self.pool_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.LogSoftmax(dim=1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
    
        h = self.node_embedding(x) 
        h = F.relu(h)
        h = self.dropout(h)
        
       
        for gnn in self.gnn_layers:
            h = F.relu(gnn(h, edge_index))
            h = self.dropout(h)
        
     
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=1)
        h_graph = self.pool_projection(h_graph)
        
        return self.classifier(h_graph)

def add_nlp_features_to_dataset(dataset, cache_path='data/dataset_with_nlp_features.pkl'):
  
    
    if os.path.exists(cache_path):
        print(f"Loading cached dataset with NLP features from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("Adding NLP features to dataset...")
    
    # Load BERT model
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load tweet texts
    tweets_df = None
    text_map = {}
    if os.path.exists('data/raw/upfd_matched_tweets.csv'):
        try:
            tweets_df = pd.read_csv('data/raw/upfd_matched_tweets.csv')
            text_map = dict(zip(tweets_df['tweet_id'].astype(str), tweets_df['text'].fillna('')))
            print(f"Loaded {len(tweets_df)} tweet texts")
        except Exception as e:
            print(f"Could not load tweet texts: {e}")
    
    # Create new dataset with NLP features
    new_dataset = []
    
    for i, data in enumerate(tqdm(dataset, desc="Processing graphs")):
        # Collect texts for this graph
        texts = []
        
        # News node
        texts.append(f"News article {i}")
        
        # Tweet nodes
        if hasattr(data, 'tweet_ids'):
            for tweet_id in data.tweet_ids:
                tweet_text = text_map.get(str(tweet_id), f"Tweet {tweet_id}")
                texts.append(tweet_text)
        else:
            for j in range(data.cascade_size):
                texts.append(f"Tweet {j} in cascade {i}")
        
        # Get BERT embeddings
        embeddings = bert_model.encode(texts, convert_to_tensor=True, device='cpu')
        
        # Concatenate original features with BERT embeddings
        original_features = data.x  # [num_nodes, 8]
        nlp_features = embeddings   # [num_nodes, 384]
        
        # Combine features
        combined_features = torch.cat([original_features, nlp_features], dim=1)  # [num_nodes, 392]
        
        # Create new data object with combined features
        new_data = Data(
            x=combined_features,
            edge_index=data.edge_index,
            y=data.y,
            cascade_size=data.cascade_size,
            node_type=data.node_type
        )
        
        new_dataset.append(new_data)
    
    # Cache the enhanced dataset
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(new_dataset, f)
    
    print(f"Cached enhanced dataset to {cache_path}")
    return new_dataset

def train_model(model, train_loader, val_loader, device, num_epochs=30):
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.005, epochs=num_epochs, steps_per_epoch=len(train_loader)
    )
    
    best_val_f1 = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                             'acc': f'{train_correct/train_total:.4f}'})
        
        # Validation every 2 epochs
        if epoch % 2 == 0:
            val_acc, val_f1, _ = evaluate_model(model, val_loader, device)
            print(f"\nVal Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), 'models/best_cascade_with_nlp.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return best_val_f1

def evaluate_model(model, loader, device):
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            out = model(batch)
            pred = out.argmax(dim=1)
            probs = torch.exp(out[:, 1])
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    return acc, f1, auc

def main():

    
    print("🔬 CASCADE GNN WITH NLP FEATURES (Minimal Change)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load original dataset
    from src.upfd_cascade_dataset import UPFDCascadeDataset
    
    print("\nLoading dataset...")
    dataset = UPFDCascadeDataset(
        root='data',
        tweets_file='upfd_matched_tweets.csv',
        edges_file='upfd_matched_edges.csv'
    )
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"Original node features: {dataset[0].x.shape[1]}")
    
    # Add NLP features to dataset
    start_time = time.time()
    enhanced_dataset = add_nlp_features_to_dataset(dataset)
    enhance_time = time.time() - start_time
    
    print(f"\nEnhanced dataset:")
    print(f"Node features: {enhanced_dataset[0].x.shape[1]} (8 original + 384 NLP)")
    print(f"Enhancement time: {enhance_time/60:.1f} minutes")
    
    # Split dataset (SAME as no-text model)
    labels = [data.y.item() for data in enhanced_dataset]
    train_idx, test_idx = train_test_split(range(len(enhanced_dataset)), test_size=0.2, 
                                          stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, 
                                        stratify=[labels[i] for i in test_idx], random_state=42)
    
    # Create datasets
    train_dataset = [enhanced_dataset[i] for i in train_idx]
    val_dataset = [enhanced_dataset[i] for i in val_idx]
    test_dataset = [enhanced_dataset[i] for i in test_idx]
    
    # Create data loaders (SAME as no-text model)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # Initialize model - SAME architecture, just different input size
    model = CascadeGNNWithNLP(
        input_dim=enhanced_dataset[0].x.shape[1],  # 392 instead of 8
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        num_classes=2
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n" + "="*60)
    print("🚀 TRAINING CASCADE GNN WITH NLP FEATURES")
    print("="*60)
    
    train_start = time.time()
    best_val_f1 = train_model(model, train_loader, val_loader, device, num_epochs=30)
    train_time = time.time() - train_start
    
    # Evaluate
    model.load_state_dict(torch.load('models/best_cascade_with_nlp.pth'))
    test_acc, test_f1, test_auc = evaluate_model(model, test_loader, device)
    
    # Results
    total_time = enhance_time + train_time
    
    print("\n" + "="*60)
    print("📊 RESULTS - CASCADE GNN WITH NLP FEATURES")
    print("="*60)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"\n⏱️  TIMING:")
    print(f"NLP Feature Time: {enhance_time/60:.1f} minutes")
    print(f"Training Time: {train_time/60:.1f} minutes")
    print(f"Total Time: {total_time/60:.1f} minutes")
    
    # Compare with no-text results if available
    if os.path.exists('results/model_comparison.txt'):
        print("\n" + "="*60)
        print("📈 COMPARISON WITH STRUCTURE-ONLY MODEL")
        print("="*60)
        
        with open('results/model_comparison.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Test F1 Score:" in line and "WITHOUT TEXT" not in line:
                    no_text_f1 = float(line.split(": ")[1].strip())
                    improvement = ((test_f1 - no_text_f1) / no_text_f1) * 100
                    print(f"Structure-only F1: {no_text_f1:.4f}")
                    print(f"With NLP F1: {test_f1:.4f}")
                    print(f"Improvement: {improvement:+.1f}%")
                    break
    
    # Save results
    with open('results/nlp_feature_results.txt', 'w') as f:
        f.write(f"CASCADE GNN WITH NLP FEATURES\n")
        f.write(f"="*40 + "\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write(f"Test AUC: {test_auc:.4f}\n")
        f.write(f"Total Time: {total_time/60:.1f} minutes\n")
        f.write(f"Parameters: {sum(p.numel() for p in model.parameters())}\n")
        f.write(f"\nNode features: 8 structural + 384 NLP = 392 total\n")
    
    print(f"\n📁 Results saved to results/nlp_feature_results.txt")

if __name__ == "__main__":
    main()