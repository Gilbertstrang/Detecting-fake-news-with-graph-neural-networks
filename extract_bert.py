import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import pickle

import sys
import os


"""To see BERT data from model"""

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from cascade_gnn_no_text import CascadeGNN
except:
    try:
        from cascade_gnn_without_text import CascadeGNN
    except:
        # Define the model inline if import fails
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
        
        class CascadeGNN(nn.Module):
            def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, 
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

# Try to import the NLP model
try:
    from cascade_gnn_with_text import CascadeGNNWithNLP
except:
    try:
        from cascade_gnn_with_nlp import CascadeGNNWithNLP
    except:
        # Define inline
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

from src.upfd_cascade_dataset import UPFDCascadeDataset

def load_trained_models(device):
    """Load your actual trained models"""
    
    print("Loading your trained models...")
    
    # Load dataset to get feature dimensions
    dataset = UPFDCascadeDataset(
        root='data',
        tweets_file='upfd_matched_tweets.csv',
        edges_file='upfd_matched_edges.csv'
    )
    
    # Model 1: Structure-only
    structure_model = CascadeGNN(
        input_dim=dataset[0].x.shape[1],  # 8 features
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        num_classes=2
    ).to(device)
    
    # Load trained weights
    structure_model.load_state_dict(torch.load('models/best_cascade_no_text.pth', map_location=device))
    structure_model.eval()
    print("✓ Loaded structure-only model")
    
    # Model 2: Structure + NLP
    # First check if we have the enhanced dataset with NLP features
    try:
        with open('data/dataset_with_nlp_features.pkl', 'rb') as f:
            enhanced_dataset = pickle.load(f)
        
        nlp_model = CascadeGNNWithNLP(
            input_dim=enhanced_dataset[0].x.shape[1],  # 392 features
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            num_classes=2
        ).to(device)
        
        nlp_model.load_state_dict(torch.load('models/best_cascade_with_nlp.pth', map_location=device))
        nlp_model.eval()
        print("✓ Loaded structure+NLP model")
    except:
        print("! Could not load NLP model - enhanced dataset not found")
        nlp_model = None
        enhanced_dataset = None
    
    return structure_model, nlp_model, dataset, enhanced_dataset

def analyze_model_predictions(model, dataset, model_name, device):
    """Analyze what a trained model has learned"""
    
    print(f"\nAnalyzing {model_name} predictions...")
    
    # Create test loader
    from sklearn.model_selection import train_test_split
    labels = [data.y.item() for data in dataset]
    _, test_idx = train_test_split(range(len(dataset)), test_size=0.2, 
                                  stratify=labels, random_state=42)
    test_dataset = [dataset[i] for i in test_idx]
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get predictions
    all_preds = []
    all_probs = []
    all_labels = []
    all_graph_features = []  # Changed to graph-level features
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Get model outputs
            out = model(batch)
            probs = torch.exp(out)  # Convert log probs to probs
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
            # Aggregate node features to graph-level features
            from torch_geometric.utils import scatter
            graph_features = []
            
            # For each graph in the batch, aggregate its node features
            for i in range(len(batch.y)):
                # Get nodes belonging to this graph
                node_mask = (batch.batch == i)
                if node_mask.sum() > 0:
                    # Get features for nodes in this graph
                    graph_nodes = batch.x[node_mask]  # [num_nodes_in_graph, num_features]
                    
                    # Aggregate: mean of all node features in this graph
                    graph_feat = graph_nodes.mean(dim=0).cpu().numpy()  # [num_features]
                    graph_features.append(graph_feat)
                else:
                    # Fallback: zero features if no nodes (shouldn't happen)
                    graph_features.append(np.zeros(batch.x.shape[1]))
            
            all_graph_features.extend(graph_features)
    
    # Convert to arrays
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_graph_features = np.array(all_graph_features)
    
    print(f"  Collected {len(all_preds)} predictions")
    print(f"  Graph features shape: {all_graph_features.shape}")
    
    # Analysis results
    results = {
        'predictions': all_preds,
        'probabilities': all_probs,
        'true_labels': all_labels,
        'features': all_graph_features,  # Now these match in size!
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'classification_report': classification_report(all_labels, all_preds, output_dict=True)
    }
    
    return results

def extract_learned_patterns(structure_results, nlp_results=None):
    """Extract what patterns the models learned"""
    
    print("\nExtracting learned patterns...")
    
    # Find cases where models disagree
    if nlp_results:
        structure_preds = structure_results['predictions']
        nlp_preds = nlp_results['predictions']
        disagree_mask = structure_preds != nlp_preds
        
        print(f"Models disagree on {disagree_mask.sum()} out of {len(structure_preds)} cases")
        
        # Analyze disagreement cases
        disagree_indices = np.where(disagree_mask)[0]
        if len(disagree_indices) > 0:
            print("\nExample disagreements:")
            for idx in disagree_indices[:5]:
                print(f"  Case {idx}: Structure says {structure_preds[idx]}, "
                      f"NLP says {nlp_preds[idx]}, Truth is {structure_results['true_labels'][idx]}")
    
    # Find high confidence predictions
    high_conf_fake = np.where(structure_results['probabilities'][:, 1] > 0.9)[0]
    high_conf_real = np.where(structure_results['probabilities'][:, 0] > 0.9)[0]
    
    print(f"\nHigh confidence predictions:")
    print(f"  Very confident FAKE: {len(high_conf_fake)} cases")
    print(f"  Very confident REAL: {len(high_conf_real)} cases")
    
    # Analyze feature patterns in high confidence cases
    if len(high_conf_fake) > 0 and len(high_conf_real) > 0:
        fake_features = structure_results['features'][high_conf_fake].mean(axis=0)
        real_features = structure_results['features'][high_conf_real].mean(axis=0)
        
        feature_names = ['time_since_news', 'node_degree', 'cascade_depth', 
                        'user_followers', 'retweet_count', 'reply_count',
                        'user_verified', 'cascade_size']  # Adjust based on your features
        
        print("\nAverage features in high-confidence cases:")
        print(f"{'Feature':<20} | {'Fake':>10} | {'Real':>10} | {'Ratio':>10}")
        print("-" * 55)
        for i, name in enumerate(feature_names[:len(fake_features)]):
            ratio = fake_features[i] / (real_features[i] + 0.001)
            print(f"{name:<20} | {fake_features[i]:>10.2f} | {real_features[i]:>10.2f} | {ratio:>10.2f}")

def visualize_model_analysis(structure_results, nlp_results=None):
    """Create visualizations of model analysis"""
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Confusion matrices
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(structure_results['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    ax1.set_title('Structure Model Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    if nlp_results:
        ax2 = plt.subplot(2, 3, 2)
        sns.heatmap(nlp_results['confusion_matrix'], annot=True, fmt='d', 
                    cmap='Greens', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        ax2.set_title('Structure+NLP Model Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
    
    # 2. Confidence distribution
    ax3 = plt.subplot(2, 3, 3)
    
    # Structure model confidence
    fake_probs_structure = structure_results['probabilities'][structure_results['true_labels'] == 1, 1]
    real_probs_structure = structure_results['probabilities'][structure_results['true_labels'] == 0, 0]
    
    ax3.hist(fake_probs_structure, bins=20, alpha=0.5, label='Fake (Structure)', color='red')
    ax3.hist(real_probs_structure, bins=20, alpha=0.5, label='Real (Structure)', color='blue')
    
    if nlp_results:
        fake_probs_nlp = nlp_results['probabilities'][nlp_results['true_labels'] == 1, 1]
        real_probs_nlp = nlp_results['probabilities'][nlp_results['true_labels'] == 0, 0]
        ax3.hist(fake_probs_nlp, bins=20, alpha=0.3, label='Fake (NLP)', color='darkred')
        ax3.hist(real_probs_nlp, bins=20, alpha=0.3, label='Real (NLP)', color='darkblue')
    
    ax3.set_xlabel('Prediction Confidence')
    ax3.set_ylabel('Count')
    ax3.set_title('Model Confidence Distribution')
    ax3.legend()
    
    # 3. Feature importance (approximate using prediction correlation)
    ax4 = plt.subplot(2, 3, 4)
    
    # Calculate correlation between features and predictions
    feature_names = ['time', 'degree', 'depth', 'followers', 'retweets', 
                    'replies', 'verified', 'size']
    
    correlations = []
    for i in range(min(structure_results['features'].shape[1], len(feature_names))):
        try:
            # Only calculate if we have valid data
            feature_data = structure_results['features'][:, i]
            prediction_data = structure_results['predictions']
            
            # Check for valid data
            if len(feature_data) == len(prediction_data) and np.var(feature_data) > 0:
                corr = np.corrcoef(feature_data, prediction_data)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                correlations.append(abs(corr))
            else:
                correlations.append(0.0)
        except:
            correlations.append(0.0)
    
    if len(correlations) > 0:
        # Sort by importance
        sorted_idx = np.argsort(correlations)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx if i < len(feature_names)]
        sorted_corr = [correlations[i] for i in sorted_idx if i < len(correlations)]
        
        if len(sorted_corr) > 0 and max(sorted_corr) > 0:
            bars = ax4.barh(sorted_features, sorted_corr, color='skyblue')
            ax4.set_xlabel('|Correlation with Prediction|')
            ax4.set_title('Feature Importance (Structure Model)')
            ax4.set_xlim(0, max(sorted_corr) * 1.1)
        else:
            ax4.text(0.5, 0.5, 'No valid correlations', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'No features to analyze', ha='center', va='center', transform=ax4.transAxes)
    
    # 4. Error analysis
    ax5 = plt.subplot(2, 3, 5)
    
    # Find misclassified examples
    errors = structure_results['predictions'] != structure_results['true_labels']
    error_features = structure_results['features'][errors]
    correct_features = structure_results['features'][~errors]
    
    if len(error_features) > 0 and len(correct_features) > 0:
        # Compare average features
        error_avg = error_features.mean(axis=0)
        correct_avg = correct_features.mean(axis=0)
        
        x = np.arange(len(feature_names[:len(error_avg)]))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, error_avg, width, label='Errors', color='red', alpha=0.7)
        bars2 = ax5.bar(x + width/2, correct_avg, width, label='Correct', color='green', alpha=0.7)
        
        ax5.set_xlabel('Features')
        ax5.set_ylabel('Average Value')
        ax5.set_title('Feature Comparison: Errors vs Correct')
        ax5.set_xticks(x)
        ax5.set_xticklabels([f[:3] for f in feature_names[:len(error_avg)]], rotation=45)
        ax5.legend()
    
    # 5. Model comparison
    if nlp_results:
        ax6 = plt.subplot(2, 3, 6)
        
        metrics = ['Precision', 'Recall', 'F1-Score']
        structure_scores = [
            structure_results['classification_report']['1']['precision'],
            structure_results['classification_report']['1']['recall'],
            structure_results['classification_report']['1']['f1-score']
        ]
        nlp_scores = [
            nlp_results['classification_report']['1']['precision'],
            nlp_results['classification_report']['1']['recall'],
            nlp_results['classification_report']['1']['f1-score']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, structure_scores, width, label='Structure', color='blue', alpha=0.7)
        bars2 = ax6.bar(x + width/2, nlp_scores, width, label='Structure+NLP', color='green', alpha=0.7)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        ax6.set_xlabel('Metrics')
        ax6.set_ylabel('Score')
        ax6.set_title('Model Performance Comparison')
        ax6.set_xticks(x)
        ax6.set_xticklabels(metrics)
        ax6.legend()
        ax6.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('trained_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def extract_model_embeddings(model, dataset, device):
    """Extract learned embeddings from GNN layers"""
    
    print("\nExtracting learned embeddings...")
    
    # Create a hook to capture intermediate representations
    node_embeddings = []
    batch_info = []
    
    def hook_fn(module, input, output):
        node_embeddings.append(output.detach().cpu())
    
    # Register hook on last GNN layer
    hook = model.gnn_layers[-1].register_forward_hook(hook_fn)
    
    # Process a few examples
    loader = DataLoader(dataset[:50], batch_size=10, shuffle=False)
    
    labels = []
    graph_embeddings = []
    
    with torch.no_grad():
        batch_idx = 0
        for batch in loader:
            batch = batch.to(device)
            _ = model(batch)
            labels.extend(batch.y.cpu().numpy())
            
            # Store batch info for later aggregation
            batch_info.append({
                'batch_tensor': batch.batch.cpu(),
                'num_graphs': len(batch.y),
                'batch_idx': batch_idx
            })
            batch_idx += 1
    
    # Remove hook
    hook.remove()
    
    # Aggregate node embeddings to graph embeddings
    current_embedding_idx = 0
    for i, info in enumerate(batch_info):
        batch_tensor = info['batch_tensor']
        num_graphs = info['num_graphs']
        
        # Get node embeddings for this batch
        if current_embedding_idx < len(node_embeddings):
            batch_node_embeddings = node_embeddings[current_embedding_idx]
            
            # Aggregate nodes to graphs
            for graph_id in range(num_graphs):
                # Get nodes belonging to this graph
                node_mask = (batch_tensor == graph_id)
                if node_mask.sum() > 0:
                    # Get embeddings for nodes in this graph
                    graph_nodes_emb = batch_node_embeddings[node_mask]
                    # Aggregate: mean of node embeddings
                    graph_emb = graph_nodes_emb.mean(dim=0)
                    graph_embeddings.append(graph_emb.numpy())
                else:
                    # Fallback: zero embedding
                    graph_embeddings.append(np.zeros(batch_node_embeddings.shape[1]))
        
        current_embedding_idx += 1
    
    # Convert to array
    graph_embeddings = np.array(graph_embeddings)
    labels = np.array(labels)
    
    print(f"  Extracted {len(graph_embeddings)} graph embeddings")
    print(f"  Embedding dimension: {graph_embeddings.shape[1] if len(graph_embeddings) > 0 else 0}")
    print(f"  Labels: {len(labels)}")
    
    return graph_embeddings, labels

def main():
    """Main analysis function"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load your trained models
    structure_model, nlp_model, dataset, enhanced_dataset = load_trained_models(device)
    
    # Analyze structure model
    structure_results = analyze_model_predictions(structure_model, dataset, "Structure-Only Model", device)
    
    # Analyze NLP model if available
    nlp_results = None
    if nlp_model and enhanced_dataset:
        nlp_results = analyze_model_predictions(nlp_model, enhanced_dataset, "Structure+NLP Model", device)
    
    # Extract learned patterns
    extract_learned_patterns(structure_results, nlp_results)
    
    # Create visualizations
    visualize_model_analysis(structure_results, nlp_results)
    
    # Extract and visualize embeddings
    embeddings, labels = extract_model_embeddings(structure_model, dataset[:100], device)
    
    # Visualize embedding space
    from sklearn.manifold import TSNE
    
    plt.figure(figsize=(8, 6))
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    colors = ['red' if l == 1 else 'green' for l in labels]
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.6, s=50)
    plt.title('GNN Learned Embeddings (t-SNE projection)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    # Add legend
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Fake')
    green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Real')
    plt.legend(handles=[red_patch, green_patch])
    
    plt.tight_layout()
    plt.savefig('gnn_embeddings.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nAnalysis complete! Check the generated visualizations.")
    
    # Save analysis results
    with open('results/model_analysis_results.pkl', 'wb') as f:
        pickle.dump({
            'structure_results': structure_results,
            'nlp_results': nlp_results
        }, f)
    
    print("Results saved to results/model_analysis_results.pkl")

if __name__ == "__main__":
    main()