import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import miditok
from pathlib import Path
import json
import math
import random
import logging
from tqdm import tqdm
import os
import time
import sys
from functools import partial
from symusic import Score
import re
import numpy as np
import argparse
from tokenize_metadata import tokenize_metadata
import config

# --- USAGE: ---
# python training.py --data_dir PATH/TO/DATASET --model_save_dir PATH/TO/SAVE/MODELS
#
# --- TO RESUME TRAINING: ---
# python training.py --data_dir PATH/TO/DATASET --model_save_dir PATH/TO/SAVE/MODELS --resume_from_checkpoint PATH/TO/transformer_best.pt

# --- Configurazione del logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configurazione / Costanti ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Iperparametri del Modello e Addestramento (Esempi!)
EPOCHS = 100
BATCH_SIZE = 64 # Riduci se hai poca memoria GPU
ACCUMULATION_STEPS = 4  # Definisce quanti "micro-batch" elaborare prima di un aggiornamento dei pesi.
LEARNING_RATE = 0.0001
EMB_SIZE = 256 # Dimensione embedding
NHEAD = 4 # Numero di head nell'attention (deve dividere EMB_SIZE)
FFN_HID_DIM = 1024 # Dimensione layer nascosto FeedForward
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DROPOUT = 0.1

# Setup Logging (opzionale ma utile)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#------------------------
# Tokenizer e Vocabolario
#------------------------

def build_or_load_tokenizer(midi_file_paths=None, force_build=False):
    """
    Costruisce o carica il tokenizer MIDI e la sua configurazione/vocabolario,
    utilizzando i parametri centralizzati da config.py.
    """
    if VOCAB_PATH.exists() and not force_build:
        logging.info(f"Caricamento configurazione tokenizer MIDI da {VOCAB_PATH}")
        try:
            tokenizer = config.MIDI_TOKENIZER_STRATEGY(params=str(VOCAB_PATH))
            logging.info(f"Tokenizer caricato con successo da {VOCAB_PATH}")
        except Exception as e:
             logging.error(f"Errore nel caricare parametri tokenizer da {VOCAB_PATH}. Errore: {e}", exc_info=True)
             logging.info("Tentativo di ricostruire il tokenizer da zero.")
             return build_or_load_tokenizer(midi_file_paths=midi_file_paths, force_build=True)
    else:
        logging.info("Creazione nuova configurazione tokenizer MIDI utilizzando config.py...")
        tokenizer = config.MIDI_TOKENIZER_STRATEGY(tokenizer_config=config.TOKENIZER_PARAMS)
        logging.info(f"Tokenizer {config.MIDI_TOKENIZER_STRATEGY.__name__} inizializzato con i parametri da config.py.")

        if midi_file_paths and hasattr(tokenizer, 'train'):
            logging.info(f"Addestramento del tokenizer (es. BPE) con {len(midi_file_paths)} file.")
            tokenizer.train(
                vocab_size=20000,
                model="BPE",
                files_paths=midi_file_paths
            )
            logging.info("Addestramento del tokenizer completato.")
        
        logging.info(f"Salvataggio configurazione tokenizer MIDI in {VOCAB_PATH}")
        VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(VOCAB_PATH))

    logging.info(f"Dimensione vocabolario MIDI (dopo caricamento/costruzione): {len(tokenizer)}")

    try:
        assert tokenizer[config.MIDI_PAD_TOKEN_NAME] is not None
        assert tokenizer[config.MIDI_SOS_TOKEN_NAME] is not None
        assert tokenizer[config.MIDI_EOS_TOKEN_NAME] is not None
    except AssertionError:
        logging.error("CRITICO: Token speciali MIDI mancanti nel tokenizer.")
        sys.exit(1)
        
    return tokenizer


def load_metadata_vocab(vocab_path):
    """Carica un vocabolario di metadati esistente e verifica i token speciali."""
    if not vocab_path.exists():
        logging.error(f"ERRORE CRITICO: File vocabolario metadati non trovato in {vocab_path}. Esegui prima dataset_creator.py.")
        sys.exit(1)
    
    logging.info(f"Caricamento vocabolario Metadati da {vocab_path}")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    token_to_id = vocab_data['token_to_id']
    
    required_specials = [config.META_PAD_TOKEN_NAME, config.META_UNK_TOKEN_NAME, config.META_SOS_TOKEN_NAME, config.META_EOS_TOKEN_NAME]
    missing = [t for t in required_specials if t not in token_to_id]
    if missing:
        logging.error(f"ERRORE CRITICO: Token speciali metadati mancanti nel file caricato: {missing}.")
        sys.exit(1)
        
    logging.info(f"Vocabolario metadati caricato. Dimensione: {len(token_to_id)}")
    return token_to_id

#------------------------
# Dataset e DataLoader
#------------------------

class MutopiaDataset(Dataset):
    def __init__(self, jsonl_path, midi_tokenizer, metadata_vocab_map,
                 max_len_midi_padded, max_len_meta, splits_dir, filter_key=None):
        self.splits_dir = Path(splits_dir)
        self.midi_tokenizer = midi_tokenizer
        self.metadata_vocab_map = metadata_vocab_map
        self.max_len_midi_padded = max_len_midi_padded
        self.max_len_meta = max_len_meta
        self.filter_key = filter_key

        try:
            self.meta_pad_id = metadata_vocab_map[config.META_PAD_TOKEN_NAME]
            self.sos_meta_id = metadata_vocab_map[config.META_SOS_TOKEN_NAME]
            self.eos_meta_id = metadata_vocab_map[config.META_EOS_TOKEN_NAME]
            self.unk_meta_id = metadata_vocab_map[config.META_UNK_TOKEN_NAME]
            
            self.sos_midi_id = midi_tokenizer[config.MIDI_SOS_TOKEN_NAME]
            self.eos_midi_id = midi_tokenizer[config.MIDI_EOS_TOKEN_NAME]
            self.midi_pad_id = midi_tokenizer[config.MIDI_PAD_TOKEN_NAME]
            if None in [self.sos_midi_id, self.eos_midi_id, self.midi_pad_id,
                        self.meta_pad_id, self.sos_meta_id, self.eos_meta_id, self.unk_meta_id]:
                raise ValueError("Uno o più ID di token speciali non sono stati trovati.")
        except (KeyError, ValueError) as e:
            logging.error(f"ERRORE CRITICO in Dataset __init__: Token speciale non trovato: {e}")
            raise

        logging.info(f"Inizializzazione Dataset da {jsonl_path} (assumendo dati pre-chunked).")
        self.samples = [] 

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    
                    meta_tokens_str = tokenize_metadata(entry.get('metadata', {}))
                    meta_token_ids = [self.metadata_vocab_map.get(token, self.unk_meta_id) for token in meta_tokens_str]
                    src_seq_list = [self.sos_meta_id] + meta_token_ids[:self.max_len_meta-2] + [self.eos_meta_id]
                    src_tensor = torch.tensor(src_seq_list, dtype=torch.long)

                    token_ids_relative_path = entry.get('token_ids_path')
                    if token_ids_relative_path:
                        self.samples.append((src_tensor, token_ids_relative_path))
                except (json.JSONDecodeError, KeyError) as e:
                    logging.warning(f"Skipping malformed line in {jsonl_path}: {e}")

        if not self.samples:
            logging.error(f"Nessun campione valido caricato da {jsonl_path}. Controllare il file.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):        
        src_tensor, token_ids_relative_path = self.samples[idx]
        try:
            full_token_path = self.splits_dir / token_ids_relative_path
            raw_midi_ids_from_file = np.load(full_token_path)
        except Exception as e:
            logging.error(f"Errore nel caricare il file di token binario: {full_token_path}. Errore: {e}")
            return None

        max_data_tokens = self.max_len_midi_padded - 2
        truncated_midi_ids = raw_midi_ids_from_file[:max_data_tokens]
        
        tgt_seq_list = [self.sos_midi_id] + truncated_midi_ids.tolist() + [self.eos_midi_id]
        tgt_tensor = torch.tensor(tgt_seq_list, dtype=torch.long)
        return src_tensor, tgt_tensor

def pad_collate_fn(batch, meta_pad_id, midi_pad_id):
    batch = [item for item in batch if item is not None]
    if not batch: return None, None, None, None
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=meta_pad_id)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=midi_pad_id)
    src_padding_mask = (src_padded == meta_pad_id)
    tgt_padding_mask = (tgt_padded == midi_pad_id)
    return src_padded, tgt_padded, src_padding_mask, tgt_padding_mask

#------------------------
# Modello Transformer
#------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead,
                 src_vocab_size, tgt_vocab_size, max_pe_len,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=max_pe_len)

        self.transformer = nn.Transformer(
            d_model=emb_size, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, memory_key_padding_mask=None, tgt_mask=None):
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size))

        if tgt_mask is None:
             tgt_len = tgt.size(1)
             tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool), diagonal=1)
        
        if memory_key_padding_mask is None:
             memory_key_padding_mask = src_padding_mask

        outs = self.transformer(src_emb, tgt_emb,
                                tgt_mask=tgt_mask,
                                src_key_padding_mask=src_padding_mask,
                                tgt_key_padding_mask=tgt_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)
        return self.generator(outs)

#------------------------
# Ciclo di Addestramento e Valutazione
#------------------------

def train_epoch(model, optimizer, criterion, train_dataloader):
    model.train()
    total_loss = 0
    processed_batches = 0
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    optimizer.zero_grad()
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training Epoch", leave=False)

    for i, batch_data in progress_bar:
        if batch_data[0] is None: continue
        
        src, tgt, src_padding_mask, tgt_padding_mask = batch_data
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        src_padding_mask = src_padding_mask.to(DEVICE)
        tgt_padding_mask = tgt_padding_mask.to(DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_input_padding_mask = tgt_padding_mask[:, :-1]
        tgt_out = tgt[:, 1:]

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            logits = model(src=src, tgt=tgt_input,
                           src_padding_mask=src_padding_mask,
                           tgt_padding_mask=tgt_input_padding_mask,
                           memory_key_padding_mask=src_padding_mask)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss = loss / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUMULATION_STEPS
        processed_batches += 1
        progress_bar.set_postfix({'train_loss': f'{total_loss / processed_batches:.4f}'})

    return total_loss / processed_batches

def evaluate(model, criterion, dataloader):
    model.eval()
    total_loss = 0
    processed_batches = 0
    progress_bar = tqdm(dataloader, desc="Evaluation", leave=False)

    with torch.no_grad():
        for batch_data in progress_bar:
            if batch_data[0] is None: continue
            src, tgt, src_padding_mask, tgt_padding_mask = batch_data
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            src_padding_mask = src_padding_mask.to(DEVICE)
            tgt_padding_mask = tgt_padding_mask.to(DEVICE)

            tgt_input = tgt[:, :-1]
            tgt_input_padding_mask = tgt_padding_mask[:, :-1]
            tgt_out = tgt[:, 1:]

            logits = model(src=src, tgt=tgt_input,
                           src_padding_mask=src_padding_mask,
                           tgt_padding_mask=tgt_input_padding_mask,
                           memory_key_padding_mask=src_padding_mask)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            total_loss += loss.item()
            processed_batches += 1
            progress_bar.set_postfix({'eval_loss': f'{total_loss / processed_batches:.4f}'})
            
    if processed_batches == 0: return float('inf')
    return total_loss / processed_batches

#------------------------ 
# Esecuzione Principale
#------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Addestra un modello Transformer per la generazione musicale.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Percorso della cartella base del dataset.")
    parser.add_argument("--model_save_dir", type=Path, required=True, help="Percorso per salvare i checkpoint del modello.")
    ### NUOVO ###
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None, help="Percorso opzionale a un checkpoint per riprendere l'addestramento.")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    MODEL_SAVE_DIR = args.model_save_dir
    SPLITS_DIR = DATA_DIR / "dataset_splits"
    VOCAB_PATH = DATA_DIR / "midi_vocab.json"
    METADATA_VOCAB_PATH = DATA_DIR / "metadata_vocab.json"
    
    ### NUOVO ###
    # Inizializzazione delle variabili per la ripresa
    checkpoint = None
    if args.resume_from_checkpoint and args.resume_from_checkpoint.exists():
        logging.info(f"Trovato checkpoint: {args.resume_from_checkpoint}. Caricamento in corso...")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=DEVICE)
        logging.info("Checkpoint caricato.")
    elif args.resume_from_checkpoint:
        logging.warning(f"Checkpoint specificato ma non trovato in: {args.resume_from_checkpoint}. Avvio di un nuovo addestramento.")


    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Utilizzo del dataset da: {DATA_DIR}")
    logging.info(f"I modelli verranno salvati in: {MODEL_SAVE_DIR}")
    
    logging.info("--- Preparazione Tokenizer e Vocabolari ---")
    midi_tokenizer = build_or_load_tokenizer(force_build=False)
    metadata_vocab_map = load_metadata_vocab(METADATA_VOCAB_PATH)

    MIDI_VOCAB_SIZE = len(midi_tokenizer)
    META_VOCAB_SIZE = len(metadata_vocab_map)
    try:
        MIDI_PAD_ID = midi_tokenizer[config.MIDI_PAD_TOKEN_NAME]
        META_PAD_ID = metadata_vocab_map[config.META_PAD_TOKEN_NAME]
    except KeyError as e:
        logging.error(f"ERRORE CRITICO: Token PAD non trovato: {e}")
        sys.exit(1)

    logging.info(f"MIDI Vocab Size: {MIDI_VOCAB_SIZE}, MIDI PAD ID: {MIDI_PAD_ID}")
    logging.info(f"Meta Vocab Size: {META_VOCAB_SIZE}, Meta PAD ID: {META_PAD_ID}")

    logging.info("--- Creazione Dataset e DataLoader ---")
    train_dataset = MutopiaDataset(SPLITS_DIR / "train.jsonl", midi_tokenizer, metadata_vocab_map, 
                                     config.MAX_SEQ_LEN_MIDI, config.MAX_SEQ_LEN_META, splits_dir=SPLITS_DIR)
    val_dataset = MutopiaDataset(SPLITS_DIR / "validation.jsonl", midi_tokenizer, metadata_vocab_map,
                                   config.MAX_SEQ_LEN_MIDI, config.MAX_SEQ_LEN_META, splits_dir=SPLITS_DIR)

    collate_fn_with_padding_ids = partial(pad_collate_fn, meta_pad_id=META_PAD_ID, midi_pad_id=MIDI_PAD_ID)
    
    num_dataloader_workers = 4 if os.name == 'nt' else os.cpu_count() or 2
    use_persistent_workers = bool(num_dataloader_workers > 0)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                collate_fn=collate_fn_with_padding_ids, 
                                num_workers=num_dataloader_workers,
                                persistent_workers=use_persistent_workers)
    val_dataloader = None
    if len(val_dataset) > 0:
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                collate_fn=collate_fn_with_padding_ids, 
                                num_workers=num_dataloader_workers,
                                persistent_workers=use_persistent_workers)

    logging.info("--- Inizializzazione Modello ---")
    
    ### NUOVO / MODIFICATO ###
    # Se stiamo riprendendo, usiamo i parametri del checkpoint. Altrimenti, usiamo quelli globali.
    if checkpoint and 'model_params' in checkpoint:
        logging.info("Inizializzazione del modello con i parametri salvati nel checkpoint.")
        model_params_to_save = checkpoint['model_params']
        # Assicurati che tutti i vocabolari siano della dimensione corretta
        model_params_to_save['src_vocab_size'] = META_VOCAB_SIZE
        model_params_to_save['tgt_vocab_size'] = MIDI_VOCAB_SIZE
    else:
        logging.info("Inizializzazione di un nuovo modello con i parametri di default.")
        max_pe_len_calculated = max(config.MAX_SEQ_LEN_MIDI, config.MAX_SEQ_LEN_META) + 100 
        model_params_to_save = {
            'num_encoder_layers': NUM_ENCODER_LAYERS, 'num_decoder_layers': NUM_DECODER_LAYERS,
            'emb_size': EMB_SIZE, 'nhead': NHEAD, 'src_vocab_size': META_VOCAB_SIZE,
            'tgt_vocab_size': MIDI_VOCAB_SIZE, 'dim_feedforward': FFN_HID_DIM, 'dropout': DROPOUT,
            'max_pe_len': max_pe_len_calculated
        }
        
    model = Seq2SeqTransformer(**model_params_to_save).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=MIDI_PAD_ID)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    ### NUOVO ###
    # Variabili per gestire lo stato dell'addestramento
    start_epoch = 1
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience_early_stopping = 10 

    if checkpoint:
        logging.info("Ripristino dello stato del modello e dell'ottimizzatore.")
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Le altre variabili di stato vengono ripristinate per continuare correttamente
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logging.info(f"Ripresa dall'epoca {start_epoch} con best_val_loss = {best_val_loss:.4f}")

    logging.info("Learning rate scheduler 'ReduceLROnPlateau' attivato con pazienza=3.")
    logging.info(f"Numero parametri: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logging.info(f"Device: {DEVICE}")
    
    vocab_info_to_save = {
        'midi_vocab_path': str(VOCAB_PATH), 'metadata_vocab_path': str(METADATA_VOCAB_PATH),
        'midi_tokenizer_strategy': config.MIDI_TOKENIZER_STRATEGY.__name__
    }

    logging.info("--- Inizio Addestramento ---")
    
    ### MODIFICATO ###
    for epoch in range(start_epoch, EPOCHS + 1):
        logging.info(f"--- Epoch {epoch}/{EPOCHS} ---")
        start_time_epoch = time.time()
        train_loss_epoch = train_epoch(model, optimizer, criterion, train_dataloader)
        
        current_val_loss = float('inf')
        if val_dataloader:
            current_val_loss = evaluate(model, criterion, val_dataloader)
        
        epoch_duration_total = time.time() - start_time_epoch
        logging.info(f"Epoch {epoch}: Train Loss = {train_loss_epoch:.4f}, Val Loss = {current_val_loss:.4f} (Durata: {epoch_duration_total:.2f}s)")

        if val_dataloader:
            scheduler.step(current_val_loss)

        if val_dataloader and current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            epochs_no_improve = 0
            best_model_path = MODEL_SAVE_DIR / "transformer_best.pt"
            logging.info(f"Nuova best validation loss: {best_val_loss:.4f}. Salvataggio modello in {best_model_path}")
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
                'model_params': model_params_to_save, 'vocab_info': vocab_info_to_save
            }, best_model_path)
        elif val_dataloader:
            epochs_no_improve += 1
        
        if epoch % 10 == 0:
            periodic_model_path = MODEL_SAVE_DIR / f"transformer_epoch_{epoch}.pt"
            logging.info(f"Salvataggio checkpoint periodico (epoch {epoch}): {periodic_model_path}")
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
                'model_params': model_params_to_save, 'vocab_info': vocab_info_to_save
            }, periodic_model_path)
        
        if val_dataloader and epochs_no_improve >= patience_early_stopping:
            logging.info(f"Nessun miglioramento per {patience_early_stopping} epoche. Early stopping.")
            break 
            
    final_model_path = MODEL_SAVE_DIR / "transformer_final.pt"
    logging.info(f"Salvataggio checkpoint finale: {final_model_path}")
    torch.save({
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
        'model_params': model_params_to_save, 'vocab_info': vocab_info_to_save
    }, final_model_path)

    logging.info("--- Addestramento Terminato ---")