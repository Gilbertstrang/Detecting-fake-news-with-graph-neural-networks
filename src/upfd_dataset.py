# src/upfd_dataset.py
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

class UPFDStyleDataset(InMemoryDataset):
    """
    UPFD-style dataset where:
    - Center node is the news article
    - Surrounding nodes are users
    - Edges represent user interactions
    """
    def __init__(self, root, tweets_file, edges_file, transform=None):
        self.tweets_file = tweets_file
        self.edges_file = edges_file
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ['upfd_style_data.pt']
    
    def process(self):
        # Load data
        tweets_df = pd.read_csv(f'{self.root}/raw/{self.tweets_file}')
        edges_df = pd.read_csv(f'{self.root}/raw/{self.edges_file}')
        
        # Convert timestamp to datetime
        tweets_df['timestamp'] = pd.to_datetime(tweets_df['timestamp'])
        
        # Group by news
        news_groups = tweets_df.groupby('news_id')
        
        data_list = []
        
        print("Creating UPFD-style graphs...")
        for news_id, news_tweets in tqdm(news_groups):
            # Create heterogeneous graph
            data = self._create_news_user_graph(news_id, news_tweets, edges_df)
            if data is not None:
                data_list.append(data)
        
        print(f"Created {len(data_list)} news-user graphs")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def _create_news_user_graph(self, news_id, news_tweets, edges_df):
        """Create a graph with news as center and users as nodes"""
        
        # Get unique users who tweeted about this news
        users = news_tweets['user_id'].unique()
        if len(users) < 3:
            return None
        
        # Create user mapping
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        
        # News node features (aggregate from all tweets)
        news_features = self._create_news_features(news_tweets)
        
        # User node features
        user_features = []
        for user in users:
            user_tweets = news_tweets[news_tweets['user_id'] == user]
            user_feat = self._create_user_features(user, user_tweets)
            user_features.append(user_feat)
        
        user_features = torch.tensor(user_features, dtype=torch.float)
        
        # Create edges between users based on retweet relationships
        user_edges = self._create_user_interaction_edges(
            news_tweets, edges_df, user_to_idx
        )
        
        # Create edges from news to all users (star topology)
        news_user_edges = []
        for user_idx in range(len(users)):
            news_user_edges.append([0, user_idx + 1])  # News is node 0
            news_user_edges.append([user_idx + 1, 0])  # Bidirectional
        
        # Combine all nodes: [news_node, user_nodes...]
        x = torch.cat([news_features.unsqueeze(0), user_features], dim=0)
        
        # Combine edges
        all_edges = news_user_edges + user_edges
        if len(all_edges) == 0:
            # If no edges, create self-loops
            all_edges = [[0, 0]]
        edge_index = torch.tensor(all_edges, dtype=torch.long).t()
        
        # Create node type indicators
        node_type = torch.zeros(x.shape[0], dtype=torch.long)
        node_type[0] = 0  # News node
        node_type[1:] = 1  # User nodes
        
        # Label
        y = torch.tensor([news_tweets['label'].iloc[0]], dtype=torch.long)
        
        # Create PyG Data
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            node_type=node_type,
            num_nodes=len(users) + 1,  # +1 for news node
            news_idx=0  # News is always node 0
        )
        
        return data
    
    def _create_news_features(self, news_tweets):
        """Create features for the news node - MUST MATCH USER FEATURES COUNT (8)"""
        # Handle potential missing values
        likes = news_tweets['likes'].fillna(0)
        retweets = news_tweets['retweets'].fillna(0)
        
        # Calculate time span (handle if all timestamps are same)
        time_span = 0.0
        if len(news_tweets) > 1:
            try:
                time_diff = news_tweets['timestamp'].max() - news_tweets['timestamp'].min()
                time_span = time_diff.total_seconds() / 3600.0  # Convert to hours
            except:
                time_span = 0.0
        
        # Calculate tweet velocity (tweets per hour) as the 8th feature
        tweet_velocity = float(len(news_tweets)) / (time_span + 1.0)  # +1 to avoid division by zero
        
        # Aggregate statistics (8 features to match user features)
        features = [
            float(len(news_tweets)),              # Total tweets
            float(news_tweets['user_id'].nunique()),  # Unique users
            float(likes.sum()),                   # Total likes
            float(retweets.sum()),                # Total retweets
            float(likes.mean()),                  # Average likes
            float(retweets.mean()),               # Average retweets
            time_span,                            # Time span in hours
            tweet_velocity,                       # Tweet velocity (tweets/hour) - 8th feature
        ]
        
        return torch.tensor(features, dtype=torch.float)
    
    def _create_user_features(self, user_id, user_tweets):
        """Create features for a user node (8 features)"""
        # Handle potential missing columns
        likes_mean = user_tweets['likes'].fillna(0).mean()
        retweets_mean = user_tweets['retweets'].fillna(0).mean()
        
        # Check if columns exist, use defaults if not
        followers = 0
        following = 0
        verified = 0.0
        credibility = 0.5
        
        if 'followers' in user_tweets.columns:
            followers = user_tweets['followers'].fillna(0).mean()
        if 'following' in user_tweets.columns:
            following = user_tweets['following'].fillna(0).mean()
        if 'verified' in user_tweets.columns:
            verified = float(user_tweets['verified'].fillna(False).any())
        if 'credibility' in user_tweets.columns:
            credibility = user_tweets['credibility'].fillna(0.5).mean()
        
        features = [
            float(len(user_tweets)),  # Number of tweets about this news
            float(likes_mean),
            float(retweets_mean),
            float(user_tweets['is_root'].any() if 'is_root' in user_tweets.columns else 0),
            float(followers),
            float(following),
            float(verified),
            float(credibility),
        ]
        
        return features
    
    def _create_user_interaction_edges(self, news_tweets, edges_df, user_to_idx):
        """Create edges between users based on retweet patterns"""
        edges = []
        
        # Get relevant edges for this news
        news_id_str = str(news_tweets['news_id'].iloc[0])
        news_edges = edges_df[edges_df['source'].astype(str).str.startswith(news_id_str)]
        
        # Map tweet interactions to user interactions
        tweet_to_user = dict(zip(news_tweets['tweet_id'].astype(str), news_tweets['user_id']))
        
        for _, edge in news_edges.iterrows():
            source_tweet = str(edge['source'])
            target_tweet = str(edge['target'])
            
            if source_tweet in tweet_to_user and target_tweet in tweet_to_user:
                source_user = tweet_to_user[source_tweet]
                target_user = tweet_to_user[target_tweet]
                
                if source_user != target_user:  # No self-loops
                    if source_user in user_to_idx and target_user in user_to_idx:
                        src_idx = user_to_idx[source_user] + 1  # +1 because news is node 0
                        tgt_idx = user_to_idx[target_user] + 1
                        edges.append([src_idx, tgt_idx])
                        edges.append([tgt_idx, src_idx])  # Bidirectional
        
        return edges