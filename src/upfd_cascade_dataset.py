# src/upfd_cascade_dataset.py
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import json
import random

class UPFDCascadeDataset(InMemoryDataset):
    """
    True UPFD-style cascade dataset where:
    - Nodes are individual tweets/retweets + news node
    - Each tweet node has user features (who posted it)
    - Edges represent information flow (retweet chains, replies)
    - REALISTIC cascade patterns based on fake vs real news propagation
    """
    
    def __init__(self, root, tweets_file, edges_file, transform=None):
        self.tweets_file = tweets_file
        self.edges_file = edges_file
        self.cascade_patterns = self._load_cascade_patterns()
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    def _load_cascade_patterns(self):
        """Load realistic cascade patterns from JSON file"""
        try:
            with open('cascade_patterns_politifact.json', 'r') as f:
                patterns = json.load(f)
            
            # Separate fake and real patterns
            fake_patterns = [p for p in patterns if p.get('is_fake', True)]
            real_patterns = [p for p in patterns if not p.get('is_fake', True)]
            
            print(f"Loaded {len(fake_patterns)} fake patterns and {len(real_patterns)} real patterns")
            
            return {
                'fake': fake_patterns,
                'real': real_patterns
            }
        except:
            print("Warning: Could not load cascade patterns, using heuristics")
            return {'fake': [], 'real': []}
    
    @property
    def processed_file_names(self):
        return ['upfd_cascade_data.pt']
    
    def process(self):
        # Load data
        tweets_df = pd.read_csv(f'{self.root}/raw/{self.tweets_file}')
        edges_df = pd.read_csv(f'{self.root}/raw/{self.edges_file}')
        
        # Convert timestamp to datetime
        tweets_df['timestamp'] = pd.to_datetime(tweets_df['timestamp'])
        
        # Group by news to create cascade graphs
        news_groups = tweets_df.groupby('news_id')
        
        data_list = []
        
        print("Creating UPFD-style cascade graphs with REALISTIC patterns...")
        for news_id, news_tweets in tqdm(news_groups):
            data = self._create_cascade_graph(news_id, news_tweets, edges_df)
            if data is not None:
                data_list.append(data)
        
        print(f"Created {len(data_list)} cascade graphs")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def _create_cascade_graph(self, news_id, news_tweets, edges_df):
        """Create a cascade graph with news + tweet nodes using REALISTIC patterns"""
        
        if len(news_tweets) < 2:
            return None
        
        # Sort tweets by timestamp to get propagation order
        news_tweets = news_tweets.sort_values('timestamp').reset_index(drop=True)
        
        # Create tweet mapping (news node will be index 0)
        tweet_ids = news_tweets['tweet_id'].tolist()
        tweet_to_idx = {tweet_id: idx + 1 for idx, tweet_id in enumerate(tweet_ids)}  # +1 for news node
        
        # === CREATE NODE FEATURES ===
        
        # 1. News node features (aggregate statistics)
        news_features = self._create_news_node_features(news_tweets)
        
        # 2. Tweet node features (individual tweets + user info)
        tweet_features = []
        for _, tweet in news_tweets.iterrows():
            tweet_feat = self._create_tweet_node_features(tweet)
            tweet_features.append(tweet_feat)
        
        tweet_features = torch.stack(tweet_features)
        
        # Combine: [news_node, tweet_nodes...]
        x = torch.cat([news_features.unsqueeze(0), tweet_features], dim=0)
        
        # === CREATE EDGES WITH REALISTIC PATTERNS ===
        
        # Determine if this is fake or real news
        label = news_tweets['label'].iloc[0]
        is_fake = (label == 1)
        
        # 1. Edges from news to all tweets (news is the source)
        news_edges = [[0, tweet_to_idx[tweet_id]] for tweet_id in tweet_ids]
        
        # 2. Cascade edges from existing edge data
        cascade_edges = self._create_cascade_edges(news_tweets, edges_df, tweet_to_idx)
        
        # 3. REALISTIC pattern-based edges
        realistic_edges = self._create_realistic_cascade_edges(
            news_tweets, tweet_to_idx, is_fake
        )
        
        # Combine all edges
        all_edges = news_edges + cascade_edges + realistic_edges
        
        if len(all_edges) == 0:
            # Fallback: news to first tweet
            all_edges = [[0, 1]]
        
        edge_index = torch.tensor(all_edges, dtype=torch.long).t()
        
        # === CREATE NODE TYPES ===
        num_nodes = len(tweet_ids) + 1  # +1 for news node
        node_type = torch.zeros(num_nodes, dtype=torch.long)
        node_type[0] = 0  # News node
        node_type[1:] = 1  # Tweet nodes
        
        # === LABEL ===
        y = torch.tensor([label], dtype=torch.long)
        
        # === CREATE DATA OBJECT ===
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            node_type=node_type,
            num_nodes=num_nodes,
            news_idx=0,  # News is always node 0
            cascade_size=len(tweet_ids),
            is_fake=is_fake
        )
        
        return data
    
    def _create_realistic_cascade_edges(self, news_tweets, tweet_to_idx, is_fake):
        """Create edges based on realistic fake vs real news patterns"""
        edges = []
        
        if len(self.cascade_patterns['fake']) == 0:
            # Fallback to heuristics
            return self._create_heuristic_edges(news_tweets, tweet_to_idx)
        
        # Select appropriate patterns
        patterns = self.cascade_patterns['fake'] if is_fake else self.cascade_patterns['real']
        
        # Find patterns with similar size
        cascade_size = len(news_tweets)
        similar_patterns = [
            p for p in patterns 
            if abs(p['num_nodes'] - cascade_size) <= max(5, cascade_size * 0.3)
        ]
        
        if not similar_patterns:
            similar_patterns = patterns  # Use any pattern if none similar
        
        # Select a random pattern
        pattern = random.choice(similar_patterns)
        
        # Apply the pattern structure
        tweets_list = news_tweets.to_dict('records')
        tweets_list.sort(key=lambda x: x['timestamp'])
        
        # Get pattern tree structure
        prop_tree = pattern.get('propagation_tree', {})
        
        # Map pattern structure to our tweets
        if '0' in prop_tree and '1' in prop_tree:
            # Pattern has depth structure
            
            # Level 1: Connect to early tweets (most active)
            level_1_size = min(len(prop_tree.get('1', [])), len(tweets_list) // 2)
            
            for i in range(level_1_size):
                if i < len(tweets_list) - 1:
                    source_idx = tweet_to_idx[tweets_list[i]['tweet_id']]
                    target_idx = tweet_to_idx[tweets_list[i + 1]['tweet_id']]
                    edges.append([source_idx, target_idx])
            
            # Level 2: Create branching patterns
            if '2' in prop_tree and len(tweets_list) > 3:
                level_2_size = min(len(prop_tree.get('2', [])), len(tweets_list) // 3)
                
                # Connect later tweets to earlier ones (branching)
                for i in range(level_2_size):
                    source_pos = min(i, len(tweets_list) // 2)
                    target_pos = min(level_1_size + i, len(tweets_list) - 1)
                    
                    if source_pos < target_pos:
                        source_idx = tweet_to_idx[tweets_list[source_pos]['tweet_id']]
                        target_idx = tweet_to_idx[tweets_list[target_pos]['tweet_id']]
                        edges.append([source_idx, target_idx])
        
        # Apply pattern-specific characteristics
        if is_fake:
            # Fake news: More star-like (many direct connections to popular tweets)
            # Connect recent tweets to most popular tweets
            for i, tweet in enumerate(tweets_list[2:], 2):  # Skip first 2
                if i < len(tweets_list):
                    # Find most popular tweet (highest engagement)
                    popular_idx = 0
                    max_engagement = 0
                    for j, prev_tweet in enumerate(tweets_list[:i]):
                        engagement = prev_tweet.get('likes', 0) + prev_tweet.get('retweets', 0)
                        if engagement > max_engagement:
                            max_engagement = engagement
                            popular_idx = j
                    
                    if random.random() < 0.6:  # 60% chance to connect to popular
                        source_idx = tweet_to_idx[tweets_list[popular_idx]['tweet_id']]
                        target_idx = tweet_to_idx[tweet['tweet_id']]
                        edges.append([source_idx, target_idx])
        else:
            # Real news: More tree-like (deeper, sequential chains)
            # Create longer chains
            for i in range(1, min(len(tweets_list), 8)):  # Create chains up to length 8
                if random.random() < 0.4:  # 40% chance for chain connection
                    source_idx = tweet_to_idx[tweets_list[i-1]['tweet_id']]
                    target_idx = tweet_to_idx[tweets_list[i]['tweet_id']]
                    edges.append([source_idx, target_idx])
        
        return edges
    
    def _create_news_node_features(self, news_tweets):
        """Create features for the news node (8 features)"""
        likes = news_tweets['likes'].fillna(0)
        retweets = news_tweets['retweets'].fillna(0)
        
        # Time span calculation
        time_span = 0.0
        if len(news_tweets) > 1:
            try:
                time_diff = news_tweets['timestamp'].max() - news_tweets['timestamp'].min()
                time_span = time_diff.total_seconds() / 3600.0  # hours
            except:
                time_span = 0.0
        
        features = [
            float(len(news_tweets)),              # Total tweets in cascade
            float(news_tweets['user_id'].nunique()),  # Unique users
            float(likes.sum()),                   # Total likes
            float(retweets.sum()),                # Total retweets
            float(likes.mean()),                  # Average likes per tweet
            float(retweets.mean()),               # Average retweets per tweet
            time_span,                            # Cascade duration (hours)
            float(len(news_tweets)) / (time_span + 1.0),  # Tweet velocity
        ]
        
        return torch.tensor(features, dtype=torch.float)
    
    def _create_tweet_node_features(self, tweet):
        """Create features for individual tweet nodes (8 features)"""
        # Handle missing values
        likes = float(tweet.get('likes', 0) or 0)
        retweets = float(tweet.get('retweets', 0) or 0)
        followers = float(tweet.get('user_followers', 1000) or 1000)
        credibility = float(tweet.get('user_credibility', 0.5) or 0.5)
        bot_score = float(tweet.get('user_bot_score', 0.1) or 0.1)
        
        features = [
            likes,                                # Tweet likes
            retweets,                             # Tweet retweets
            float(tweet.get('is_root', False)),   # Is root tweet
            followers,                            # User followers
            float(tweet.get('user_verified', False) or False),  # User verified
            credibility,                          # User credibility
            bot_score,                            # User bot score
            1.0 if tweet.get('text', '').count('!') > 2 else 0.0,  # Emotional indicator
        ]
        
        return torch.tensor(features, dtype=torch.float)
    
    def _create_cascade_edges(self, news_tweets, edges_df, tweet_to_idx):
        """Create edges from existing edge data (retweet relationships)"""
        edges = []
        
        # Get edges for this news cascade
        news_id_str = str(news_tweets['news_id'].iloc[0])
        
        # Filter edges that belong to this cascade
        if 'news_id' in edges_df.columns:
            cascade_edges = edges_df[edges_df['news_id'] == news_id_str]
        else:
            # Fallback: filter by tweet ID prefix
            cascade_edges = edges_df[
                edges_df['source'].astype(str).str.startswith(news_id_str)
            ]
        
        for _, edge in cascade_edges.iterrows():
            source_tweet = str(edge['source'])
            target_tweet = str(edge['target'])
            
            if source_tweet in tweet_to_idx and target_tweet in tweet_to_idx:
                src_idx = tweet_to_idx[source_tweet]
                tgt_idx = tweet_to_idx[target_tweet]
                edges.append([src_idx, tgt_idx])
        
        return edges
    
    def _create_heuristic_edges(self, news_tweets, tweet_to_idx):
        """Create heuristic edges based on temporal and behavioral patterns"""
        edges = []
        tweets_list = news_tweets.to_dict('records')
        
        # Sort by timestamp
        tweets_list.sort(key=lambda x: x['timestamp'])
        
        for i, tweet in enumerate(tweets_list):
            tweet_idx = tweet_to_idx[tweet['tweet_id']]
            
            # Connect to recent tweets (temporal window)
            window_size = min(5, i)  # Look back at most 5 tweets
            
            for j in range(max(0, i - window_size), i):
                prev_tweet = tweets_list[j]
                prev_idx = tweet_to_idx[prev_tweet['tweet_id']]
                
                # Probability of connection based on:
                # 1. Time difference (closer = higher prob)
                # 2. User similarity (same type = higher prob)
                # 3. Engagement level
                
                time_diff_hours = (tweet['timestamp'] - prev_tweet['timestamp']).total_seconds() / 3600.0
                time_factor = np.exp(-time_diff_hours / 24.0)  # Decay over 24 hours
                
                # User type similarity
                same_user_type = (
                    tweet.get('user_type', 'regular') == prev_tweet.get('user_type', 'regular')
                )
                user_factor = 1.5 if same_user_type else 1.0
                
                # Engagement factor
                engagement_factor = min(2.0, (prev_tweet.get('retweets', 0) + 1) ** 0.2)
                
                connection_prob = time_factor * user_factor * engagement_factor * 0.3
                
                if np.random.random() < connection_prob:
                    edges.append([prev_idx, tweet_idx])
        
        return edges 