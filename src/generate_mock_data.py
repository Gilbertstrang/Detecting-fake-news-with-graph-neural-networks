import pandas as pd
import numpy as np
import random
import requests
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import subprocess
import os

"""This uses Ollama to generate fake news tweets, although final work used ChatGPT 4"""

class TweetGenerator:
    def __init__(self, use_ollama=True):
        self.use_ollama = use_ollama
        
        if use_ollama:
            # Check if Ollama is available
            try:
                # Check if ollama is running
                response = requests.get('http://localhost:11434/api/tags')
                if response.status_code == 200:
                    print("✓ Ollama is running")
                    # Check available models
                    models = response.json()['models']
                    print(f"Available models: {[m['name'] for m in models]}")
                    
                    # Use the first available small model
                    self.model = 'phi' if any('phi' in m['name'] for m in models) else models[0]['name']
                    print(f"Using model: {self.model}")
                else:
                    raise Exception("Ollama not responding")
            except Exception as e:
                print(f"❌ Ollama not available: {e}")
                print("Falling back to GPT-2")
                self.use_ollama = False
        
        if not self.use_ollama:
            print("Loading GPT-2...")
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            import torch
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_with_ollama(self, prompt):
        """Generate text using Ollama API"""
        try:
            response = requests.post('http://localhost:11434/api/generate', 
                json={
                    'model': self.model,
                    'prompt': prompt + " (Tweet text only, no explanation):",
                    'stream': False,
                    'options': {
                        'temperature': 0.8,
                        'max_tokens': 100,
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                text = response.json()['response'].strip()
                # Clean up the response
                text = text.replace('\n', ' ')
                # Remove any meta-text
                if ':' in text[:50]:
                    text = text.split(':', 1)[1].strip()
                return text[:280]
            else:
                print(f"Ollama error: {response.status_code}")
                return self.generate_with_gpt2(prompt)
                
        except Exception as e:
            print(f"Ollama generation failed: {e}")
            return self.generate_with_gpt2(prompt)
    
    def generate_with_gpt2(self, prompt):
        """Generate text using GPT-2"""
        import torch
        inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=50, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=100,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt):].strip()[:280]
    
    def generate_fake_tweet(self, topic):
        prompt = f"Write a sensational fake news tweet about {topic}"
        
        if self.use_ollama:
            return self.generate_with_ollama(prompt)
        else:
            return self.generate_with_gpt2(prompt)
    
    def generate_real_tweet(self, topic):
        prompt = f"Write a factual news tweet about {topic}"
        
        if self.use_ollama:
            return self.generate_with_ollama(prompt)
        else:
            return self.generate_with_gpt2(prompt)

class CascadeGenerator:
    def __init__(self, num_users=5000, num_news=200):
        self.num_users = num_users
        self.num_news = num_news
        self.tweet_gen = TweetGenerator(use_ollama=True)  # Changed to True!
        self.users = self._create_users()
        
    def _create_users(self):
        """Create realistic user profiles"""
        users = []
        
        for i in range(self.num_users):
            # Different user types
            if i < self.num_users * 0.05:  # 5% verified
                user_type = 'verified'
                followers = int(np.random.lognormal(8, 2))
                credibility = random.uniform(0.6, 0.95)
            elif i < self.num_users * 0.15:  # 10% bots
                user_type = 'bot'
                followers = random.randint(10, 1000)
                credibility = random.uniform(0, 0.3)
            else:  # Regular users
                user_type = 'regular'
                followers = int(np.random.lognormal(4, 1.5))
                credibility = random.uniform(0.2, 0.8)
            
            users.append({
                'user_id': f'user_{i:05d}',
                'user_type': user_type,
                'followers': followers,
                'following': int(np.random.lognormal(4, 1.2)),
                'verified': user_type == 'verified',
                'credibility': credibility,
            })
        
        return pd.DataFrame(users)
    
    def generate_dataset(self):
        """Generate complete cascade dataset"""
        all_tweets = []
        all_edges = []
        
        topics = [
            'COVID-19 vaccine', 'climate change', 'stock market',
            'artificial intelligence', 'election results', 'celebrity news',
            'cryptocurrency', 'healthcare reform', 'space exploration'
        ]
        
        print(f"Generating {self.num_news} news cascades...")
        
        for i in tqdm(range(self.num_news)):
            news_id = f'news_{i:05d}'
            is_fake = random.random() < 0.3  # 30% fake news
            topic = random.choice(topics)
            
            # Generate cascade
            cascade_tweets, cascade_edges = self._generate_cascade(
                news_id, is_fake, topic
            )
            
            all_tweets.extend(cascade_tweets)
            all_edges.extend(cascade_edges)
        
        return pd.DataFrame(all_tweets), pd.DataFrame(all_edges)
    
    def _generate_cascade(self, news_id, is_fake, topic):
        """Generate a single news cascade"""
        tweets = []
        edges = []
        
        # Select initial spreaders
        if is_fake:
            # Fake news: spread by less credible users
            spreaders = self.users[self.users['credibility'] < 0.4].sample(
                random.randint(2, 5)
            )
        else:
            # Real news: more organic spread
            spreaders = self.users.sample(random.randint(1, 3))
        
        base_time = datetime.now() - timedelta(hours=random.randint(1, 48))
        
        # Generate initial tweets
        for idx, (_, user) in enumerate(spreaders.iterrows()):
            tweet_text = (self.tweet_gen.generate_fake_tweet(topic) 
                         if is_fake else 
                         self.tweet_gen.generate_real_tweet(topic))
            
            tweet = {
                'tweet_id': f'{news_id}_t{len(tweets):05d}',
                'user_id': user['user_id'],
                'text': tweet_text,
                'timestamp': base_time + timedelta(minutes=random.randint(0, 30)),
                'likes': int(np.random.lognormal(2, 1.5)),
                'retweets': int(np.random.lognormal(2, 2)),
                'is_root': True,
                'label': 1 if is_fake else 0,
                'news_id': news_id,
            }
            tweets.append(tweet)
        
        # Generate retweet cascade
        for root_tweet in tweets[:]:  # Copy to avoid modification during iteration
            self._expand_tweet(root_tweet, tweets, edges, max_depth=3)
        
        return tweets, edges
    
    def _expand_tweet(self, parent_tweet, tweets, edges, depth=0, max_depth=3):
        """Recursively expand retweets"""
        if depth >= max_depth:
            return
        
        # Number of retweets decreases with depth
        num_retweets = int(parent_tweet['retweets'] * (0.5 ** depth))
        num_retweets = min(num_retweets, random.randint(0, 5))
        
        for _ in range(num_retweets):
            # Select retweeter
            retweeter = self.users.sample(1).iloc[0]
            
            # Create retweet
            retweet = {
                'tweet_id': f"{parent_tweet['news_id']}_t{len(tweets):05d}",
                'user_id': retweeter['user_id'],
                'text': f"RT @{parent_tweet['user_id']}: {parent_tweet['text'][:200]}...",
                'timestamp': parent_tweet['timestamp'] + timedelta(
                    minutes=random.randint(5, 120)
                ),
                'likes': int(np.random.lognormal(1, 1)),
                'retweets': int(np.random.lognormal(1.5, 1.5)),
                'is_root': False,
                'label': parent_tweet['label'],
                'news_id': parent_tweet['news_id'],
            }
            tweets.append(retweet)
            
            # Add edge
            edges.append({
                'source': parent_tweet['tweet_id'],
                'target': retweet['tweet_id'],
                'timestamp': retweet['timestamp'],
            })
            
            # Recursively expand
            self._expand_tweet(retweet, tweets, edges, depth + 1, max_depth)

if __name__ == "__main__":
    # Generate the data
    generator = CascadeGenerator(num_users=5000, num_news=200)
    tweets_df, edges_df = generator.generate_dataset()
    
    # Save to files
    tweets_df.to_csv('data/raw/mock_tweets.csv', index=False)
    edges_df.to_csv('data/raw/mock_edges.csv', index=False)
    
    print(f"\nGenerated {len(tweets_df)} tweets in {len(tweets_df['news_id'].unique())} cascades")
    print(f"Generated {len(edges_df)} edges")