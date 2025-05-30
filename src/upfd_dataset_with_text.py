import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import re

class UPFDWithTextDataset(InMemoryDataset):
    """UPFD dataset with tweet text embeddings"""
    
    def __init__(self, root, tweets_file, edges_file, 
                 text_model='bert-base-uncased', transform=None):
        self.tweets_file = tweets_file
        self.edges_file = edges_file
        self.text_model = text_model
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return [f'upfd_with_text_{self.text_model.replace("/", "_")}.pt']
    
    def process(self):
        # Load data
        tweets_df = pd.read_csv(f'{self.root}/raw/{self.tweets_file}')
        edges_df = pd.read_csv(f'{self.root}/raw/{self.edges_file}')
        
        # Initialize text encoder
        print(f"Loading {self.text_model} for text encoding...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model)
        self.model = AutoModel.from_pretrained(self.text_model)
        self.model.eval()
        
        # Group by news
        news_groups = tweets_df.groupby('news_id')
        data_list = []
        
        print("Creating UPFD graphs with text features...")
        for news_id, news_tweets in tqdm(news_groups):
            data = self._create_news_user_graph_with_text(
                news_id, news_tweets, edges_df
            )
            if data is not None:
                data_list.append(data)
        
        print(f"Created {len(data_list)} graphs with text features")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def _create_news_user_graph_with_text(self, news_id, news_tweets, edges_df):
        """Create graph with text-enhanced features"""
        
        users = news_tweets['user_id'].unique()
        if len(users) < 3:
            return None
        
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        
        # === NEWS NODE FEATURES ===
        # 1. Aggregate tweet texts for news representation
        all_tweets_text = ' '.join(news_tweets['text'].fillna('').tolist())
        news_text_embed = self._get_text_embedding(all_tweets_text, max_length=512)
        
        # 2. Statistical features
        news_stats = torch.tensor([
            len(news_tweets),
            news_tweets['user_id'].nunique(),
            news_tweets['likes'].sum(),
            news_tweets['retweets'].sum(),
            news_tweets['likes'].mean(),
            news_tweets['retweets'].mean(),
            self._compute_text_diversity(news_tweets['text']),  # Text diversity
            self._compute_sentiment_variance(news_tweets['text']),  # Sentiment variance
        ], dtype=torch.float)
        
        # Combine text and stats for news node
        news_features = torch.cat([news_text_embed, news_stats])
        
        # === USER NODE FEATURES ===
        user_features = []
        
        for user in users:
            user_tweets = news_tweets[news_tweets['user_id'] == user]
            
            # 1. User's tweet text embedding
            user_text = ' '.join(user_tweets['text'].fillna('').tolist())
            user_text_embed = self._get_text_embedding(user_text, max_length=256)
            
            # 2. User behavioral features
            user_stats = torch.tensor([
                len(user_tweets),
                user_tweets['likes'].mean(),
                user_tweets['retweets'].mean(),
                float(user_tweets['is_root'].any()),
                user_tweets['user_followers'].mean(),
                user_tweets.get('user_following', user_tweets['user_followers'] * 0.1).mean(),  # Estimated
                float(user_tweets['user_verified'].any()),
                user_tweets['user_credibility'].mean(),
                self._compute_writing_style_features(user_text),  # Writing style
            ], dtype=torch.float)
            
            # Combine text and stats
            user_feat = torch.cat([user_text_embed, user_stats])
            user_features.append(user_feat)
        
        user_features = torch.stack(user_features)
        
        # Combine all nodes
        x = torch.cat([news_features.unsqueeze(0), user_features], dim=0)
        
        # Create edges (same as before)
        edge_index = self._create_edges(news_tweets, edges_df, user_to_idx, len(users))
        
        # Node types
        node_type = torch.zeros(x.shape[0], dtype=torch.long)
        node_type[0] = 0  # News
        node_type[1:] = 1  # Users
        
        # Label
        y = torch.tensor([news_tweets['label'].iloc[0]], dtype=torch.long)
        
        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            node_type=node_type,
            num_nodes=len(users) + 1,
            news_idx=0
        )
    
    def _get_text_embedding(self, text, max_length=128):
        """Get BERT/RoBERTa embedding"""
        inputs = self.tokenizer(
            text, 
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token
            embedding = outputs.last_hidden_state[0, 0, :]
        
        return embedding
    
    def _compute_text_diversity(self, texts):
        """Measure vocabulary diversity"""
        all_words = ' '.join(texts.fillna('')).lower().split()
        unique_words = len(set(all_words))
        total_words = len(all_words)
        return unique_words / (total_words + 1)  # Avoid division by zero
    
    def _compute_sentiment_variance(self, texts):
        """Simple sentiment variance (you could use a real sentiment model)"""
        # Count exclamation marks, capitals, etc.
        excitement_scores = []
        for text in texts.fillna(''):
            score = (
                text.count('!') + 
                text.count('?') + 
                sum(1 for c in text if c.isupper()) / (len(text) + 1)
            )
            excitement_scores.append(score)
        return np.var(excitement_scores)
    
    def _compute_writing_style_features(self, text):
        """Extract writing style features"""
        # Simple features - you could add more
        features = [
            text.count('!'),  # Exclamations
            text.count('?'),  # Questions
            text.count('...'),  # Ellipsis
            len(re.findall(r'[A-Z]{2,}', text)),  # All caps words
            len(re.findall(r'http\S+', text)),  # URLs
            len(re.findall(r'@\w+', text)),  # Mentions
            len(re.findall(r'#\w+', text)),  # Hashtags
        ]
        return torch.tensor(features, dtype=torch.float).mean()  # Simple average
    
    def _create_edges(self, news_tweets, edges_df, user_to_idx, num_users):
        """Create edge index (same as original UPFD)"""
        edges = []
        
        # News to all users
        for user_idx in range(num_users):
            edges.append([0, user_idx + 1])
            edges.append([user_idx + 1, 0])
        
        # User-user interactions
        news_edges = edges_df[edges_df['source'].str.startswith(news_tweets['news_id'].iloc[0])]
        tweet_to_user = dict(zip(news_tweets['tweet_id'], news_tweets['user_id']))
        
        for _, edge in news_edges.iterrows():
            if edge['source'] in tweet_to_user and edge['target'] in tweet_to_user:
                src_user = tweet_to_user[edge['source']]
                tgt_user = tweet_to_user[edge['target']]
                
                if src_user != tgt_user and src_user in user_to_idx and tgt_user in user_to_idx:
                    src_idx = user_to_idx[src_user] + 1
                    tgt_idx = user_to_idx[tgt_user] + 1
                    edges.append([src_idx, tgt_idx])
                    edges.append([tgt_idx, src_idx])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()