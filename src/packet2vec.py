import os
import glob
import re
import pickle
import argparse
import warnings
import multiprocessing
import random

import numpy as np
import pyshark
from transformers import BertTokenizerFast
from gensim.models import Word2Vec
from torch.utils.tensorboard import SummaryWriter

# Suppress pyshark warnings
warnings.filterwarnings('ignore')

# --- Configurazione tokenizzatore e regex ANSI ---
TOKENIZER_MODEL = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_MODEL)
ansi_escape = re.compile(r'\x1b\[[0-9;]*[mK]')

# Subsampling rate globale (fra 0 e 1); verrà sovrascritta da args
SUBSAMPLING_RATE = 1.0

# Mapping dei source IP per identificare traffico 'malicious'
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
    """Estrae src/dst IP, porte e label da un pacchetto pyshark."""
    src = dst = sport = dport = None
    label = 'benign'
    if hasattr(pkt, 'ip'):
        ip = pkt.ip
        src = ip.get_field_value('ip.src')
        dst = ip.get_field_value('ip.dst')
        for lbl, ips in MAL_MAP.items():
            if lbl in fn and src in ips:
                label = lbl
                break
    if hasattr(pkt, 'tcp'):
        sport = pkt.tcp.get_field_value('tcp.srcport')
        dport = pkt.tcp.get_field_value('tcp.dstport')
    elif hasattr(pkt, 'udp'):
        sport = pkt.udp.get_field_value('udp.srcport')
        dport = pkt.udp.get_field_value('udp.dstport')
    return {'srcaddr': src, 'dstaddr': dst, 'sport': sport, 'dport': dport, 'label': label}

def process_file_fast(fn):
    """
    Processa un file pcap:
    - subsampling casuale: processa solo ~SUBSAMPLING_RATE dei pacchetti
    - raccoglie raw_text = str(pkt)
    - pulisce ANSI e normalizza spazi
    - tokenizza in batch
    - restituisce liste di token e metadati
    """
    texts, metas = [], []
    capture = pyshark.FileCapture(fn, keep_packets=False)
    for pkt in capture:
        # subsampling: skip con probabilità 1-SUBSAMPLING_RATE
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

    if not texts:
        return [], []

    # tokenizzazione in batch
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=tokenizer.model_max_length
    )
    token_lists = [
        tokenizer.convert_ids_to_tokens(ids)
        for ids in enc['input_ids']
    ]
    return token_lists, metas

def load_packets(paths):
    """Carica e tokenizza in parallelo tutti i file pcap."""
    workers = min(len(paths), os.cpu_count())
    with multiprocessing.Pool(workers) as pool:
        results = pool.map(process_file_fast, paths)
    sentences, meta = [], []
    for toks, m in results:
        sentences.extend(toks)
        meta.extend(m)
    return sentences, meta

def process_and_save(sentences, meta, model, output_path):
    """Calcola embeddings medi da Word2Vec e salva in pickle."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = []
    for sent, md in zip(sentences, meta):
        vecs = [model.wv[t] for t in sent if t in model.wv]
        if not vecs:
            continue
        emb = np.mean(vecs, axis=0)
        data.append({'meta': md, 'embedding': emb})
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} vectors to {output_path}")

def train(args):
    global SUBSAMPLING_RATE
    SUBSAMPLING_RATE = args.sampling_rate
    random.seed(args.seed)

    paths = sorted(glob.glob(os.path.join(args.raw_dir, '*.pcap*')))
    sentences, meta = load_packets(paths)

    model = Word2Vec(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=1, hs=1, negative=0,
        compute_loss=True
    )
    model.build_vocab(sentences)

    writer = SummaryWriter(args.log_dir)
    total_iters = 0
    num_sentences = len(sentences)
    checkpoint_interval = args.checkpoint_interval

    for epoch in range(1, args.epochs + 1):
        print(f"Starting epoch {epoch}/{args.epochs}")
        for i in range(0, num_sentences, args.batch_size):
            batch = sentences[i : i + args.batch_size]
            model.train(batch, total_examples=len(batch), epochs=1)
            scores = model.score(batch, total_sentences=len(batch))
            batch_loss = -np.mean(scores)

            total_iters += 1
            writer.add_scalar('Loss/train', batch_loss, total_iters)

            if total_iters % 100 == 0:
                processed = min((i + args.batch_size), num_sentences)
                print(f"Iter {total_iters}: loss={batch_loss:.4f} | Packets {processed}/{num_sentences}")

            if total_iters % checkpoint_interval == 0:
                ckpt = os.path.splitext(args.model_path)[0] + f"_iter{total_iters}.model"
                os.makedirs(os.path.dirname(ckpt), exist_ok=True)
                model.save(ckpt)
                print(f"Checkpoint saved to {ckpt}")

        epoch_ckpt = os.path.splitext(args.model_path)[0] + f"_epoch{epoch}.model"
        os.makedirs(os.path.dirname(epoch_ckpt), exist_ok=True)
        model.save(epoch_ckpt)
        print(f"Epoch {epoch} complete. Saved to {epoch_ckpt}")

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model.save(args.model_path)
    print(f"Training complete. Model saved to {args.model_path}")

    process_and_save(sentences, meta, model, args.train_output)

def test(args):
    global SUBSAMPLING_RATE
    SUBSAMPLING_RATE = args.sampling_rate
    random.seed(args.seed)

    paths = sorted(glob.glob(os.path.join(args.test_dir, '*.pcap*')))
    sentences, meta = load_packets(paths)

    model = Word2Vec(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=1, hs=1, negative=0,
        compute_loss=True
    )
    model.build_vocab(sentences)

    writer = SummaryWriter(args.test_log_dir)
    total_iters = 0
    num_sentences = len(sentences)

    for epoch in range(1, args.epochs + 1):
        print(f"Starting test epoch {epoch}/{args.epochs}")
        for i in range(0, num_sentences, args.batch_size):
            batch = sentences[i : i + args.batch_size]
            model.train(batch, total_examples=len(batch), epochs=1)
            scores = model.score(batch, total_sentences=len(batch))
            batch_loss = -np.mean(scores)

            total_iters += 1
            writer.add_scalar('Loss/test', batch_loss, total_iters)

            if total_iters % 100 == 0:
                processed = min((i + args.batch_size), num_sentences)
                print(f"Test iter {total_iters}: loss={batch_loss:.4f} | Packets {processed}/{num_sentences}")

    os.makedirs(os.path.dirname(args.test_model_path), exist_ok=True)
    model.save(args.test_model_path)
    print(f"Testing complete. Model saved to {args.test_model_path}")

    process_and_save(sentences, meta, model, args.test_output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    # Train
    p1 = sub.add_parser('train')
    p1.add_argument('--raw-dir', default='data/raw')
    p1.add_argument('--vector-size', type=int, default=100)
    p1.add_argument('--window', type=int, default=5)
    p1.add_argument('--min-count', type=int, default=1)
    p1.add_argument('--workers', type=int, default=os.cpu_count())
    p1.add_argument('--epochs', type=int, default=5)
    p1.add_argument('--batch-size', type=int, default=100)
    p1.add_argument('--checkpoint-interval', type=int, default=1000)
    p1.add_argument('--sampling-rate', type=float, default=0.1,
                    help='Frazione di pacchetti da processare (0.0–1.0)')
    p1.add_argument('--seed', type=int, default=42,
                    help='Seed per il sampling riproducibile')
    p1.add_argument('--log-dir', default='runs/word2vec')
    p1.add_argument('--model-path', default='models/word2vec_pcap.model')
    p1.add_argument('--train-output', default='data/embeddings/train.pkl')
    p1.set_defaults(func=train)

    # Test
    p2 = sub.add_parser('test')
    p2.add_argument('--test-dir', default='data/test')
    p2.add_argument('--vector-size', type=int, default=100)
    p2.add_argument('--window', type=int, default=5)
    p2.add_argument('--min-count', type=int, default=1)
    p2.add_argument('--workers', type=int, default=os.cpu_count())
    p2.add_argument('--epochs', type=int, default=5)
    p2.add_argument('--batch-size', type=int, default=100)
    p2.add_argument('--sampling-rate', type=float, default=1,
                    help='Frazione di pacchetti da processare (0.0–1.0)')
    p2.add_argument('--seed', type=int, default=42,
                    help='Seed per il sampling riproducibile')
    p2.add_argument('--test-log-dir', default='runs/word2vec_test')
    p2.add_argument('--test-model-path', default='models/word2vec_pcap_test.model')
    p2.add_argument('--test-output', default='data/embeddings/test.pkl')
    p2.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)

