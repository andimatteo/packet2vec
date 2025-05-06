from scapy.all import Ether, IP, TCP, UDP, ICMP, Raw, Dot1Q
import random
import time
import os
import glob
import pickle
import argparse
import warnings
import multiprocessing
import pyshark

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import re

# Suppress pyshark warnings
warnings.filterwarnings('ignore')

# --- Configurazione tokenizer e regex ANSI ---
DEFAULT_MODEL = 'prajjwal1/bert-tiny'
ansi_escape = re.compile(r'\x1b\[[0-9;]*[mK]')

# Subsampling rate globale (fra 0 e 1); verrà sovrascritta da args
SUBSAMPLING_RATE = 1.0

# Mapping dei source IP per identificare traffico per tipo di attacco
MAL_MAP = {
    'UDPFlood':   ['10.155.15.4'],
    'Goldeneye':  ['10.155.15.3'],
    'ICMPFlood':  ['10.155.15.9'],
    'SYNScan':    ['10.155.15.1'],
    'Torshammer': ['10.155.15.4'],
    'UDPScan':    ['10.155.15.9'],
    'Slowloris':  ['10.155.15.0'],
    'SYNFlood':   ['10.155.15.4'],
    'TCPConnect': ['10.155.15.1']
}

def extract_meta(pkt, fn):
    src = dst = sport = dport = None
    label = 'benign'
    if hasattr(pkt, 'ip'):
        ip = pkt.ip
        src = ip.get_field_value('ip.src')
        dst = ip.get_field_value('ip.dst')
        for attack_name, ips in MAL_MAP.items():
            if attack_name in fn and src in ips:
                label = attack_name
                break
    if hasattr(pkt, 'tcp'):
        sport = pkt.tcp.get_field_value('tcp.srcport')
        dport = pkt.tcp.get_field_value('tcp.dstport')
    elif hasattr(pkt, 'udp'):
        sport = pkt.udp.get_field_value('udp.srcport')
        dport = pkt.udp.get_field_value('udp.dstport')
    return {'srcaddr': src, 'dstaddr': dst, 'sport': sport, 'dport': dport, 'label': label}


def process_file_fast(fn):
    texts, metas = [], []
    capture = pyshark.FileCapture(fn, keep_packets=False)
    for pkt in capture:
        if random.random() > SUBSAMPLING_RATE:
            continue
        try:
            raw = str(pkt)
        except Exception:
            continue
        clean = ansi_escape.sub('', raw).strip()
        if not clean:
            continue
        texts.append(" ".join(clean.split()))
        metas.append(extract_meta(pkt, fn))
    capture.close()
    return texts, metas


def load_packets(paths):
    workers = min(len(paths), os.cpu_count())
    with multiprocessing.Pool(workers) as pool:
        results = pool.map(process_file_fast, paths)
    texts, metas = [], []
    for t, m in results:
        texts.extend(t)
        metas.extend(m)
    return texts, metas


def compute_bert_embeddings(texts, model, tokenizer, device, max_length):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def process_and_save(texts, metas, model, tokenizer, output_path, batch_size, max_length):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_meta = metas[i:i+batch_size]
        embeddings = compute_bert_embeddings(batch, model, tokenizer, device, max_length)
        for emb, md in zip(embeddings, batch_meta):
            data.append({'meta': md, 'embedding': emb})
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} vectors to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estrai embeddings da pcap con BERT')
    parser.add_argument('--pcap-dir', default='data/raw', help='Directory contenente file pcap')
    parser.add_argument('--output', default='data/embeddings/bert_embeddings.pkl', help='File di output pickle')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per BERT')
    parser.add_argument('--sampling-rate', type=float, default=1.0, help='Frazione di pacchetti (0.0–1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Seed per sampling riproducibile')
    parser.add_argument('--model-name', default=DEFAULT_MODEL, help='Modello HuggingFace da utilizzare')
    parser.add_argument('--max-length', type=int, default=512, help='Max token length')
    args = parser.parse_args()

    SUBSAMPLING_RATE = args.sampling_rate
    random.seed(args.seed)

    paths = sorted(glob.glob(os.path.join(args.pcap_dir, '*.pcap*')))
    texts, metas = load_packets(paths)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.model_max_length = args.max_length
    model = AutoModel.from_pretrained(args.model_name)

    process_and_save(texts, metas, model, tokenizer, args.output, args.batch_size, args.max_length)

