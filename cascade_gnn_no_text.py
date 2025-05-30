import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import DataLoader, Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CascadeGNN(nn.Module):
    
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, 
                 dropout=0.2, num_classes=2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Project cascade features to hidden dimension
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Pooling projection
        self.pool_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.LogSoftmax(dim=1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Embed cascade features only (no text)
        h = self.node_embedding(x)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Apply GNN layers
        for gnn in self.gnn_layers:
            h = F.relu(gnn(h, edge_index))
            h = self.dropout(h)
        
        # Pool to graph level
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=1)
        h_graph = self.pool_projection(h_graph)
        
        return self.classifier(h_graph)

def train_no_text(model, train_loader, val_loader, device, num_epochs=30):
    """Training loop for non-text model"""
    
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
            val_acc, val_f1, _ = evaluate_no_text(model, val_loader, device)
            print(f"\nVal Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), 'models/best_cascade_no_text.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return best_val_f1

def evaluate_no_text(model, loader, device):
    """Evaluation for non-text model"""
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

def compare_models():
    """Main function to train both models and compare"""
    
    print("🔬 COMPARING CASCADE GNN WITH AND WITHOUT TEXT")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load dataset
    from src.upfd_cascade_dataset import UPFDCascadeDataset
    
    print("\nLoading dataset...")
    dataset = UPFDCascadeDataset(
        root='data',
        tweets_file='upfd_matched_tweets.csv',
        edges_file='upfd_matched_edges.csv'
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Node features: {dataset[0].x.shape[1]}")
    
    # Split dataset
    labels = [data.y.item() for data in dataset]
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, 
                                          stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, 
                                        stratify=[labels[i] for i in test_idx], random_state=42)
    
    # Create datasets
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    # Create data loaders (no custom collate needed)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # Initialize model WITHOUT text
    model_no_text = CascadeGNN(
        input_dim=dataset[0].x.shape[1],
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        num_classes=2
    ).to(device)
    
    print(f"\nModel WITHOUT text parameters: {sum(p.numel() for p in model_no_text.parameters()):,}")
    
    # Train model WITHOUT text
    print("\n" + "="*60)
    print("🚀 TRAINING CASCADE GNN WITHOUT TEXT FEATURES")
    print("="*60)
    
    start_time = time.time()
    best_val_f1_no_text = train_no_text(model_no_text, train_loader, val_loader, device, num_epochs=30)
    train_time_no_text = time.time() - start_time
    
    # Evaluate
    model_no_text.load_state_dict(torch.load('models/best_cascade_no_text.pth'))
    test_acc_no_text, test_f1_no_text, test_auc_no_text = evaluate_no_text(model_no_text, test_loader, device)
    
    # Load results from text model if available
    text_results = None
    if os.path.exists('results/fast_training_times.txt'):
        with open('results/fast_training_times.txt', 'r') as f:
            lines = f.readlines()
            text_f1 = float(lines[3].split(': ')[1].strip())
            text_time = float(lines[0].split(': ')[1].split(' ')[0])
            text_results = {'f1': text_f1, 'time': text_time}
    
    # Print comparison
    print("\n" + "="*60)
    print("📊 COMPARISON RESULTS")
    print("="*60)
    
    print("\n🔹 CASCADE GNN WITHOUT TEXT:")
    print(f"   Test Accuracy: {test_acc_no_text:.4f}")
    print(f"   Test F1 Score: {test_f1_no_text:.4f}")
    print(f"   Test AUC: {test_auc_no_text:.4f}")
    print(f"   Best Val F1: {best_val_f1_no_text:.4f}")
    print(f"   Training Time: {train_time_no_text/60:.1f} minutes")
    print(f"   Parameters: {sum(p.numel() for p in model_no_text.parameters()):,}")
    
    if text_results:
        print("\n🔹 CASCADE GNN WITH BERT TEXT:")
        print(f"   Test F1 Score: {text_results['f1']:.4f}")
        print(f"   Total Time: {text_results['time']:.1f} minutes")
        print(f"   Parameters: ~50,000")
        
        # Calculate improvements
        f1_improvement = ((text_results['f1'] - test_f1_no_text) / test_f1_no_text) * 100
        print(f"\n📈 IMPROVEMENT WITH TEXT:")
        print(f"   F1 Score: {f1_improvement:+.1f}%")
        print(f"   Absolute F1 Gain: {text_results['f1'] - test_f1_no_text:+.4f}")
    
    # Save comparison results
    with open('results/model_comparison.txt', 'w') as f:
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("="*40 + "\n\n")
        f.write("CASCADE GNN WITHOUT TEXT:\n")
        f.write(f"Test Accuracy: {test_acc_no_text:.4f}\n")
        f.write(f"Test F1 Score: {test_f1_no_text:.4f}\n")
        f.write(f"Test AUC: {test_auc_no_text:.4f}\n")
        f.write(f"Training Time: {train_time_no_text/60:.1f} minutes\n")
        f.write(f"Parameters: {sum(p.numel() for p in model_no_text.parameters())}\n")
        
        if text_results:
            f.write("\nCASCADE GNN WITH BERT TEXT:\n")
            f.write(f"Test F1 Score: {text_results['f1']:.4f}\n")
            f.write(f"Total Time: {text_results['time']:.1f} minutes\n")
            f.write(f"F1 Improvement: {f1_improvement:+.1f}%\n")
    
    print(f"\n📁 Results saved to results/model_comparison.txt")
    

    return {
        'no_text': {
            'test_acc': test_acc_no_text,
            'test_f1': test_f1_no_text,
            'test_auc': test_auc_no_text,
            'time': train_time_no_text
        },
        'with_text': text_results
    }

if __name__ == "__main__":
    compare_models()