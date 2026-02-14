## Bot Detection System


Social media bot detection for "Bot or Not" competition that uses behavioral feature engineering and Random Forest Classification

## Overview

This system detects bot accounts on social media platforms by analyzing behavioral patterns including:
- Content features (hashtag usage, URLs, sentiment)
- Temporal patterns (posting regularity, time gaps)
- Profile characteristics (username patterns)
- Engagement patterns (retweet ratio, mention diversity)

Developed for the Bot or Not Challenge (February 2026).

## Performance

**Training Results:**
- English: 508/516 points (98.5%)
- French: 213/220 points (96.8%)

**Cross-Validation:**
- English: 74.0% average
- French: 44.5% average (very small dataset for french bots)

## Installation
```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

## Usage

### Training (for pre-competition)

Train models on all practice data; note that all practice datasets must (.txt and .json) must be in the current working directory:
```bash
python competition_script.py train
```

This creates:
- `model_english.pkl` (English model, threshold=0.4)
- `model_french.pkl` (French model, threshold=0.55)

These thresholds were pre-determined from Pre-competition training using the provided example datasets.

### Prediction (for competition day)

Generate predictions for evaluation data:
```bash
python competition_script.py predict <eval_file> <language> <team_name>
```

**Example:**
```bash
python competition_script.py predict eval_en.json english TeamName
python competition_script.py predict eval_fr.json french TeamName
```

**Output:**
- `TeamName.detections.en.txt` (English predictions)
- `TeamName.detections.fr.txt` (French predictions)

## Features

The system extracts 17 behavioral features per user:

**Content Features:**
- Average hashtags per post
- Percentage of posts with URLs
- Average post length
- Emoji usage
- Sentiment (mean & std deviation)
- Retweet ratio
- Mention diversity

**Temporal Features:**
- Average time gap between posts
- Standard deviation of time gaps
- Coefficient of variation (regularity measure)

**Profile Features:**
- Username length
- Username contains numbers
- Username ends with 4+ digits
- Username randomness score

**Provided Features:**
- Tweet count
- Z-score

## Model

**Algorithm:** Random Forest Classifier

**Hyperparameters:**
- 200 trees
- Max depth: 8
- Min samples split: 8
- Min samples leaf: 3
- Max features: sqrt
- Class weight: balanced

**Threshold Optimization:**
- English: 0.4 
- French: 0.55 

## Files

- `bot_detection.py` - Full training and analysis pipeline
- `competition_script.py` - Fast competition day script
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Development

**Analyze training data:**
```bash
python bot_detection.py
```

This runs the full pipeline including:
- Feature extraction
- Model training
- Threshold optimization
- Cross-validation
- Performance analysis

## Team

Davi Guimaraes Ruas

