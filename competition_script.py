import json
import pandas as pd
import numpy as np
import re
import emoji
import pickle
import sys
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob


class BotFeatureExtractor:
    """Feature extraction for bot detection"""

    def extract_user_features(self, user, user_posts):
        features = {'user_id': user['id']}
        features['tweet_count'] = user.get('tweet_count', 0)
        features['z_score'] = user.get('z_score', 0)
        
        if len(user_posts) == 0:
            features.update({
                'average_hashtags': 0, 'pct_url_posts': 0, 'avg_tweet_len': 0,
                'emoji_avg': 0, 'avg_sentiment': 0, 'std_sentiment': 0,
                'retweet_ratio': 0, 'unique_mentions_ratio': 0
            })
        else:
            features['average_hashtags'] = user_posts['text'].str.count('#').mean()
            has_url = user_posts['text'].str.contains(r"https?://", case=False, regex=True, na=False)
            features['pct_url_posts'] = has_url.sum() / len(user_posts)
            features['avg_tweet_len'] = user_posts['text'].str.len().mean()
            features['emoji_avg'] = user_posts['text'].apply(detect_emojis).mean()
            
            sentiments = []
            for text in user_posts['text']:
                try:
                    sentiments.append(TextBlob(str(text)).sentiment.polarity)
                except:
                    sentiments.append(0)
            features['avg_sentiment'] = np.mean(sentiments)
            features['std_sentiment'] = np.std(sentiments)
            
            retweet_count = user_posts['text'].str.startswith('RT @', na=False).sum()
            features['retweet_ratio'] = retweet_count / len(user_posts)
            
            all_mentions = []
            for text in user_posts['text']:
                all_mentions.extend(re.findall(r'@(\w+)', str(text)))
            features['unique_mentions_ratio'] = len(set(all_mentions)) / len(all_mentions) if all_mentions else 0
        
        if len(user_posts) > 1:
            timestamps = pd.to_datetime(user_posts['created_at']).sort_values()
            time_gaps = timestamps.diff().dt.total_seconds().dropna()
            features['avg_time_gap'] = time_gaps.mean()
            features['std_time_gap'] = time_gaps.std()
            features['cv_time_gap'] = time_gaps.std() / time_gaps.mean() if time_gaps.mean() > 0 else 0
        else:
            features.update({'avg_time_gap': 0, 'std_time_gap': 0, 'cv_time_gap': 0})
        
        username = user.get('username', '') or ''
        features['username_len'] = len(username)
        features['username_has_numbers'] = int(bool(re.search(r'\d', username)))
        features['username_ends_with_numbers'] = int(bool(re.search(r'\d{4,}$', username)))
        features['username_random_score'] = sum(c.isdigit() for c in username) / max(len(username), 1)
        
        return features
    
    def extract_all_features(self, users_df, posts_df):
        features_list = []
        for idx, user in users_df.iterrows():
            user_posts = posts_df[posts_df['author_id'] == user['id']]
            features_list.append(self.extract_user_features(user, user_posts))
        return pd.DataFrame(features_list)


def detect_emojis(text):
    return 0 if pd.isna(text) else emoji.emoji_count(text)


def train_models():
    """Train final models on all practice data"""
    print("="*60)
    print("TRAINING FINAL MODELS")
    print("="*60)
    
    print("\nLoading practice datasets...")
    datasets = {}
    for dataset_id in [30, 31, 32, 33]:
        with open(f'dataset.posts&users.{dataset_id}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(f'dataset.bots.{dataset_id}.txt', 'r', encoding='utf-8') as f:
            bot_ids = set(f.read().strip().split('\n'))
        
        datasets[dataset_id] = {
            'users': pd.DataFrame(data['users']),
            'posts': pd.DataFrame(data['posts']),
            'bot_ids': bot_ids
        }
    
    print("Extracting features...")
    extractor = BotFeatureExtractor()
    features = {}
    
    for dataset_id in [30, 31, 32, 33]:
        features[dataset_id] = extractor.extract_all_features(
            datasets[dataset_id]['users'],
            datasets[dataset_id]['posts']
        )
        features[dataset_id]['is_bot'] = features[dataset_id]['user_id'].isin(datasets[dataset_id]['bot_ids'])
    
    english_features = pd.concat([features[30], features[32]], ignore_index=True)
    french_features = pd.concat([features[31], features[33]], ignore_index=True)
    
    print("\nTraining English model...")
    X_en = english_features.drop(['user_id', 'id', 'is_bot'], axis=1, errors='ignore').fillna(0)
    y_en = english_features['is_bot']
    
    model_en = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=8,
        min_samples_leaf=3, max_features='sqrt',
        random_state=42, class_weight='balanced'
    )
    model_en.fit(X_en, y_en)
    print(f"  Trained on {len(X_en)} users ({sum(y_en)} bots)")
    
    print("\nTraining French model...")
    X_fr = french_features.drop(['user_id', 'id', 'is_bot'], axis=1, errors='ignore').fillna(0)
    y_fr = french_features['is_bot']
    
    model_fr = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=8,
        min_samples_leaf=3, max_features='sqrt',
        random_state=42, class_weight='balanced'
    )
    model_fr.fit(X_fr, y_fr)
    print(f"  Trained on {len(X_fr)} users ({sum(y_fr)} bots)")
    
    with open('model_english.pkl', 'wb') as f:
        pickle.dump({'model': model_en, 'feature_columns': X_en.columns.tolist(), 'threshold': 0.4}, f)
    
    with open('model_french.pkl', 'wb') as f:
        pickle.dump({'model': model_fr, 'feature_columns': X_fr.columns.tolist(), 'threshold': 0.55}, f)
    
    print("\n Models saved: model_english.pkl, model_french.pkl")


def predict(eval_file, lang, team_name):
    """Generate predictions for evaluation data"""
    print(f"\n{'='*60}")
    print(f"PREDICTING {lang.upper()} DATA")
    print(f"{'='*60}")
    
    print(f"\nLoading {eval_file}...")
    with open(eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    users_df = pd.DataFrame(data['users'])
    posts_df = pd.DataFrame(data['posts'])
    print(f"  {len(users_df)} users, {len(posts_df)} posts")
    
    print("Extracting features...")
    extractor = BotFeatureExtractor()
    features_df = extractor.extract_all_features(users_df, posts_df)
    
    model_file = f'model_{lang}.pkl'
    print(f"Loading {model_file}...")
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    feature_columns = model_data['feature_columns']
    threshold = model_data['threshold']
    
    X = features_df[feature_columns].fillna(0)
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    bot_users = features_df[predictions == 1]['user_id'].tolist()
    print(f"\nDetected {len(bot_users)} bots ({len(bot_users)/len(users_df)*100:.1f}%)")
    
    output_file = f"{team_name}.detections.{lang[:2]}.txt"
    with open(output_file, 'w') as f:
        for user_id in bot_users:
            f.write(f"{user_id}\n")
    
    print(f"Saved: {output_file}")
    return output_file


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_models()
    elif len(sys.argv) > 1 and sys.argv[1] == 'predict':
        if len(sys.argv) < 5:
            print("Usage: python competition_script.py predict <eval_file> <lang> <team_name>")
            sys.exit(1)
        predict(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Usage:")
        print("Train: python competition_script.py train")
        print("Predict: python competition_script.py predict <eval_file> <lang> <team_name>")