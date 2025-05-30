import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_max_pool

class OfficialUPFDModel(nn.Module):
    """The actual model from UPFD paper"""
    
    def __init__(self, num_features, hidden_dim=128, num_classes=2, 
                 model_type='sage', concat_news=True):
        super().__init__()
        
        self.concat = concat_news
        
        # Choose convolution type
        if model_type == 'gcn':
            self.conv1 = GCNConv(num_features, hidden_dim)
        elif model_type == 'sage':
            self.conv1 = SAGEConv(num_features, hidden_dim)
        elif model_type == 'gat':
            self.conv1 = GATConv(num_features, hidden_dim)
        
        # If concatenating news features
        if self.concat:
            self.lin0 = nn.Linear(num_features, hidden_dim)
            self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Final classifier
        self.lin2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch):
        # Single convolution layer
        x = F.relu(self.conv1(x, edge_index))
        
        # Global pooling
        x = global_max_pool(x, batch)
        
        if self.concat:
            # Get news node features (first node of each graph)
            news_features = []
            for i in range(batch.max().item() + 1):
                mask = (batch == i).nonzero(as_tuple=True)[0]
                news_features.append(x[mask[0]])  # First node is news
            
            news = torch.stack(news_features)
            news = F.relu(self.lin0(news))
            
            # Concatenate
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.lin1(x))
        
        # Classify
        x = F.log_softmax(self.lin2(x), dim=-1)
        return x

# Even simpler version without concat because performance issues...
class SimpleGCN(nn.Module):
    """Ultra simple GCN for testing"""
    
    def __init__(self, num_features, hidden_dim=64, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch):
        # Two conv layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Pool and classify
        x = global_max_pool(x, batch)
        return F.log_softmax(self.classifier(x), dim=1)

def test_simple_model():
    from src.upfd_dataset import UPFDStyleDataset
    from torch_geometric.loader import DataLoader
    
    # Load  problematic dataset
    dataset = UPFDStyleDataset(
        root='data',
        tweets_file='upfd_meaningful_tweets.csv',
        edges_file='upfd_matched_edges_fixed.csv'
    )
    
    print(f"Dataset: {len(dataset)} graphs")
    
    # Use the SIMPLE model
    model = SimpleGCN(
        num_features=dataset[0].x.shape[1],
        hidden_dim=32,  # Small!
        num_classes=2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Quick test
    loader = DataLoader(dataset[:10], batch_size=2)
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.batch)
        print(f"Output shape: {out.shape}")
        print(f"Predictions: {out.argmax(dim=1)}")
        break
    
    return model

if __name__ == "__main__":
    model = test_simple_model()
    