import json
import pandas as pd
import numpy as np
import re
import emoji
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from textblob import TextBlob


class BotFeatureExtractor:
    """Extracts behavioral features from social media users for bot detection"""

    def extract_user_features(self, user, user_posts):
        """Extract features for a single user"""
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
        """Extract features for all users"""
        features_list = []
        for idx, user in users_df.iterrows():
            user_posts = posts_df[posts_df['author_id'] == user['id']]
            features_list.append(self.extract_user_features(user, user_posts))
        return pd.DataFrame(features_list)


class BotDetector:
    """Random Forest based bot detector with optimized hyperparameters"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'
        )
        self.feature_columns = None
    
    def train(self, features_df):
        """Train the model on labeled data"""
        X = features_df.drop(['user_id', 'id', 'is_bot'], axis=1, errors='ignore')
        y = features_df['is_bot']
        
        self.feature_columns = X.columns.tolist()
        X = X.fillna(0)
        
        print(f"\nTraining Random Forest on {len(X)} samples ({sum(y)} bots, {len(y)-sum(y)} humans)")
        print(f"Features: {len(self.feature_columns)}")
        
        self.model.fit(X, y)
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 features:")
        for idx, row in importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self
    
    def predict(self, features_df, threshold=0.5):
        """Make predictions with adjustable threshold"""
        X = features_df[self.feature_columns].fillna(0)
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        return predictions, probabilities
    
    def evaluate(self, features_df, threshold=0.5):
        """Evaluate performance using competition scoring"""
        predictions, _ = self.predict(features_df, threshold)
        y_true = features_df['is_bot']
        
        print(f"\n{'='*60}")
        print(f"EVALUATION (threshold={threshold:.2f})")
        print(f"{'='*60}")
        
        print("\n" + classification_report(y_true, predictions, target_names=['Human', 'Bot']))
        
        cm = confusion_matrix(y_true, predictions)
        print("Confusion Matrix:")
        print(f"              Predicted")
        print(f"              Human  Bot")
        print(f"Human           {cm[0,0]:3d}   {cm[0,1]:3d}")
        print(f"Bot             {cm[1,0]:3d}   {cm[1,1]:3d}")
        
        tp, fn, fp = cm[1, 1], cm[1, 0], cm[0, 1]
        score = (4 * tp) - fn - (2 * fp)
        
        print(f"\nCompetition Score: {score}")
        print(f"  TP: {tp} (+{tp*4}), FN: {fn} (-{fn}), FP: {fp} (-{fp*2})")
        
        return score
    
    def find_optimal_threshold(self, features_df):
        """Find threshold that maximizes competition score"""
        _, probabilities = self.predict(features_df, threshold=0.5)
        y_true = features_df['is_bot']
        
        best_score = -999
        best_threshold = 0.5
        
        print("\nOptimizing threshold...")
        for threshold in np.arange(0.3, 0.7, 0.05):
            predictions = (probabilities >= threshold).astype(int)
            cm = confusion_matrix(y_true, predictions)
            tp, fn, fp = cm[1, 1], cm[1, 0], cm[0, 1]
            score = (4 * tp) - fn - (2 * fp)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        print(f"Optimal: {best_threshold:.2f} (score: {best_score})")
        return best_threshold


def cross_validate(features_df, n_splits=5):
    """K-fold cross-validation to assess generalization"""
    X = features_df.drop(['user_id', 'id', 'is_bot'], axis=1, errors='ignore').fillna(0)
    y = features_df['is_bot']
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print(f"\n{'='*60}")
    print(f"K-FOLD CROSS-VALIDATION (k={n_splits})")
    print(f"{'='*60}")
    
    scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=8,
            min_samples_leaf=3, max_features='sqrt',
            random_state=42, class_weight='balanced'
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        cm = confusion_matrix(y_test, predictions)
        tp, fn, fp = cm[1, 1], cm[1, 0], cm[0, 1]
        score = (4 * tp) - fn - (2 * fp)
        max_score = 4 * sum(y_test)
        
        scores.append((score, max_score))
        print(f"Fold {fold+1}: {score}/{max_score} ({score/max_score*100:.1f}%)")
    
    avg = np.mean([s[0]/s[1] for s in scores]) * 100
    print(f"\nAverage: {avg:.1f}%")
    
    return scores


def detect_emojis(text):
    """Count emojis in text"""
    return 0 if pd.isna(text) else emoji.emoji_count(text)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BOT DETECTION SYSTEM - TRAINING PIPELINE")
    print("="*60)
    
    print("\nLoading datasets...")
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
        lang = "EN" if dataset_id in [30, 32] else "FR"
        print(f"  Dataset {dataset_id} ({lang}): {len(datasets[dataset_id]['users'])} users, {len(bot_ids)} bots")
    
    print("\n" + "="*60)
    print("FEATURE EXTRACTION")
    print("="*60)
    
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
    
    print(f"\nEnglish: {len(english_features)} users ({sum(english_features['is_bot'])} bots)")
    print(f"French: {len(french_features)} users ({sum(french_features['is_bot'])} bots)")
    
    print("\n" + "="*60)
    print("ENGLISH MODEL")
    print("="*60)
    
    detector_en = BotDetector()
    detector_en.train(english_features)
    threshold_en = detector_en.find_optimal_threshold(english_features)
    score_en = detector_en.evaluate(english_features, threshold=threshold_en)
    cv_en = cross_validate(english_features)
    
    print("\n" + "="*60)
    print("FRENCH MODEL")
    print("="*60)
    
    detector_fr = BotDetector()
    detector_fr.train(french_features)
    threshold_fr = detector_fr.find_optimal_threshold(french_features)
    score_fr = detector_fr.evaluate(french_features, threshold=threshold_fr)
    cv_fr = cross_validate(french_features)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nEnglish: {score_en}/516 points, CV: {np.mean([s[0]/s[1] for s in cv_en])*100:.1f}%")
    print(f"French: {score_fr}/220 points, CV: {np.mean([s[0]/s[1] for s in cv_fr])*100:.1f}%")