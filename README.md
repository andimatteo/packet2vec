# 5G Packet Classification Project

This repository implements a modular pipeline for classifying 5G network packets. It supports embedding generation, multihead binary classifiers, dynamic detection of new classes, and is designed for federated training environments.

## Features

1. **Dataset: 5G-NIDD**
   - Used for both training and testing across the pipeline.
2. **Embedding Generation**
   - **`packet2vec.py`**: Trains a Word2Vec model on packet sequences to produce custom embeddings.
   - **Pretrained BERT**: Alternative plug-and-play method to obtain embeddings without dedicated training.
3. **Classification Model**
   - **K binary classifiers** based on a Transformer with Multi-Head Attention (MHA), one per known class.
   - Each MHA classifier outputs a binary prediction: 1 for class membership, 0 otherwise.
4. **Outlier Detection & New-Class Creation**
   - Packets predicted negative by all K classifiers are labeled *outliers*.
   - Outlier embeddings are buffered and clustered using DBSCAN.
   - When at least **M** outliers fall within distance **ε**, a new class is defined, and a new MHA binary classifier is trained on that cluster.
5. **Modular & Federated Architecture**
   - **Install** or **remove** classifiers dynamically.
   - Designed for **federated training**, allowing distributed nodes to update local components and synchronize selective weights or gradients.

## Requirements

- Python 3.8 or higher  
- Install dependencies:

```bash
pip install -r requirements.txt
```

## 1. dataset preparation
Ensure the 5G-NIDD dataset nogtp layer is placed under data/raw. 
Use your preferred splitting strategy or a provided preprocessing script to generate train.pkl, test.pkl, etc.

## 2. Embedding Generation
### 2.1 Using `packet2vec.py`
```python
python src/packet2vec.py \
  --input data/raw/train_sequences.txt \
  --output data/embeddings/embeddings.csv \
  --vector-size 128 --window 5 --epochs 10
```
### 2.2 Using Pretrained BERT
```python
python src/bert_embeddings.py \
  --input data/raw/train_sequences.txt \
  --output data/embeddings/bert_embeddings.pkl \
  --model-name bert-base-uncased
```
## 3. Classification Model Training
```python
python src/train_classifier.py \
  --embeddings data/embeddings/embeddings.csv \
  --labels data/embeddings/train.pkl \
  --config config.yaml
```
![Classifier architecture](https://i.imgur.com/eL6BFwT.png)

Pipeline steps:
1. Load embeddings & labels.
2. Initialize K MHA binary classifiers.
3. Train each classifier.
4. During validation, collect embeddings of packets classified as outliers.
5. Cluster outlier embeddings using DBSCAN.
6. For each valid cluster, train a new binary MHA classifier and update active classifiers.
 

## 4. Dynamic New-Class Creation Policy
Clusters are detected by DBSCAN with parameters `epsilon` and `min_samples` defined in `config.yaml`. 
When a cluster meets the threshold, a new class `NewClass_<timestamp>` is created, and a corresponding classifier is trained on its embeddings.

## 5. Modular & Federated Architecture
- *Dynamic Classifier Management*: Install or remove classifiers via configuration.
- *Federated Training*: Each client node can:
    - Update local classifier weights.
    - Report outlier embeddings for clustering.
    - Sync selected weights or gradients with the central server.
