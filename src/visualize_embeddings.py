import pickle
import os
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

# Config
EMBEDDING_FILE = 'data/embeddings/bert_embeddings.pkl'
LOG_DIR = 'runs/embeddings_vis'
SAMPLE_SIZE = 10000  # numero di punti da visualizzare

# Carica embeddings
with open(EMBEDDING_FILE, 'rb') as f:
    data = pickle.load(f)

# Organizza per label
label_to_items = defaultdict(list)
for item in data:
    label = item['meta'].get('label', 'unknown')
    label_to_items[label].append(item)

# Campiona proporzionalmente
sampled = []
for label, items in label_to_items.items():
    k = min(int(SAMPLE_SIZE * len(items) / len(data)), len(items))
    sampled.extend(random.sample(items, k))

# Shuffle finale
random.shuffle(sampled)

# Estrai embedding e metadati
embeddings = np.stack([x['embedding'] for x in sampled])
metadata = [x['meta']['label'] for x in sampled]

# Scrivi su TensorBoard
writer = SummaryWriter(LOG_DIR)
writer.add_embedding(mat=embeddings, metadata=metadata, tag='Sampled_Embeddings')
writer.close()

print(f"Scritti {len(sampled)} embeddings in TensorBoard ({LOG_DIR})")

