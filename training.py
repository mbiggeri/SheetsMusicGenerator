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


# ### MODIFICA TPU: Import necessari e setup ambiente ###
# Cerca di importare torch_xla. Se fallisce, assumiamo di non essere in un ambiente TPU.
try:
    import torch_xla.core.xla_model as xm # type: ignore
    import torch_xla.distributed.parallel_loader as pl # type: ignore
    import torch_xla.distributed.xla_multiprocessing as xmp # type: ignore
    IS_TPU = True
except ImportError:
    IS_TPU = False

# --- Configurazione del logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configurazione / Costanti ---
# ### MODIFICA TPU: Il device verrà impostato dinamicamente in seguito ###
DEVICE = None 

# Iperparametri del Modello e Addestramento (Esempi!)
EPOCHS = 100
# Nota: Su TPU, il BATCH_SIZE è per *singolo core*. Il batch size effettivo sarà BATCH_SIZE * 8.
BATCH_SIZE = 256 
# Nota: L'accumulo di gradienti è generalmente SCONSIGLIATO su TPU. Impostalo a 1.
ACCUMULATION_STEPS = 1
LEARNING_RATE = 0.0003
EMB_SIZE = 256
NHEAD = 4
FFN_HID_DIM = 1024
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DROPOUT = 0.1

# ... (il resto delle funzioni `build_or_load_tokenizer`, `load_metadata_vocab`, `MutopiaDataset`, `pad_collate_fn`, `PositionalEncoding` e `Seq2SeqTransformer` rimangono INVARIATE) ...

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
             tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=DEVICE, dtype=torch.bool), diagonal=1)
        
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

# ### MODIFICA TPU: Il ciclo di train/eval è stato modificato per supportare entrambi gli ambienti ###
def train_epoch(model, optimizer, criterion, train_dataloader):
    model.train()
    total_loss = 0
    processed_batches = 0
    
    # Usa GradScaler solo per CUDA, non per TPU
    use_amp = (not IS_TPU) and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    
    optimizer.zero_grad()
    
    # Per TPU, usiamo ParallelLoader che gestisce la distribuzione dei dati
    if IS_TPU:
        # L'enumerazione avviene direttamente sul loader di XLA
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Training Epoch (Core {xm.get_ordinal()})", leave=False)
    else:
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training Epoch", leave=False)

    for i, batch_data in (enumerate(progress_bar) if not IS_TPU else progress_bar):
        if batch_data[0] is None: continue
        
        # Su GPU, i dati vanno spostati manualmente. Su TPU no, ParallelLoader lo gestisce.
        if not IS_TPU:
            src, tgt, src_padding_mask, tgt_padding_mask = [d.to(DEVICE) for d in batch_data]
        else:
            src, tgt, src_padding_mask, tgt_padding_mask = batch_data

        tgt_input = tgt[:, :-1]
        tgt_input_padding_mask = tgt_padding_mask[:, :-1]
        tgt_out = tgt[:, 1:]

        with autocast(device_type="cuda" if not IS_TPU else "cpu", dtype=torch.float16 if not IS_TPU else torch.bfloat16, enabled=use_amp or IS_TPU):
            logits = model(src=src, tgt=tgt_input,
                           src_padding_mask=src_padding_mask,
                           tgt_padding_mask=tgt_input_padding_mask,
                           memory_key_padding_mask=src_padding_mask)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            if ACCUMULATION_STEPS > 1:
                loss = loss / ACCUMULATION_STEPS

        if use_amp: # Logica GPU con AMP
            scaler.scale(loss).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else: # Logica TPU o CPU
            loss.backward()
            # xm.optimizer_step esegue la sincronizzazione dei gradienti tra i core TPU
            if IS_TPU:
                xm.optimizer_step(optimizer)
            else: # CPU
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * (ACCUMULATION_STEPS if ACCUMULATION_STEPS > 1 else 1)
        processed_batches += 1
        progress_bar.set_postfix({'train_loss': f'{total_loss / processed_batches:.4f}'})

    return total_loss / processed_batches

def evaluate(model, criterion, dataloader):
    model.eval()
    total_loss = 0
    processed_batches = 0
    
    desc = f"Evaluation (Core {xm.get_ordinal()})" if IS_TPU else "Evaluation"
    progress_bar = tqdm(dataloader, desc=desc, leave=False)

    with torch.no_grad():
        for batch_data in progress_bar:
            if batch_data[0] is None: continue
            
            if not IS_TPU:
                src, tgt, src_padding_mask, tgt_padding_mask = [d.to(DEVICE) for d in batch_data]
            else:
                src, tgt, src_padding_mask, tgt_padding_mask = batch_data

            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_input_padding_mask = tgt_padding_mask[:, :-1]
            
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

# ### MODIFICA TPU: La logica principale è incapsulata in una funzione ###
# Questo è il pattern richiesto da torch_xla.distributed.xla_multiprocessing
def main_training_loop(rank, args):
    global DEVICE, VOCAB_PATH, METADATA_VOCAB_PATH # Rendi le variabili globali accessibili
    
    # Imposta il device corretto per il processo corrente
    if IS_TPU:
        DEVICE = xm.xla_device()
        # Stampa solo dal processo master per evitare output duplicati
        log_fn = xm.master_print
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_fn = logging.info
    
    # Inizializza le variabili globali del path basate sugli argomenti
    DATA_DIR = args.data_dir
    MODEL_SAVE_DIR = args.model_save_dir
    SPLITS_DIR = DATA_DIR / "dataset_splits"
    VOCAB_PATH = DATA_DIR / "midi_vocab.json"
    METADATA_VOCAB_PATH = DATA_DIR / "metadata_vocab.json"
    
    # Il master si occupa di creare le directory
    if not IS_TPU or xm.is_master_ordinal():
        MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        log_fn(f"Ambiente rilevato: {'TPU' if IS_TPU else 'GPU/CPU'}")
        log_fn(f"Device in uso: {DEVICE}")

    # Attendi che il master abbia finito prima di procedere (per TPU)
    if IS_TPU:
        xm.rendezvous('setup_directories')

    checkpoint = None
    if args.resume_from_checkpoint and args.resume_from_checkpoint.exists():
        log_fn(f"Trovato checkpoint: {args.resume_from_checkpoint}. Caricamento in corso...")
        # Carica sul dispositivo corrente (CPU per poi spostare)
        checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu')
        log_fn("Checkpoint caricato.")
    elif args.resume_from_checkpoint:
        logging.warning(f"Checkpoint specificato ma non trovato in: {args.resume_from_checkpoint}. Avvio di un nuovo addestramento.")

    log_fn(f"Utilizzo del dataset da: {DATA_DIR}")
    log_fn(f"I modelli verranno salvati in: {MODEL_SAVE_DIR}")
    
    log_fn("--- Preparazione Tokenizer e Vocabolari ---")
    midi_tokenizer = build_or_load_tokenizer(force_build=False)
    metadata_vocab_map = load_metadata_vocab(METADATA_VOCAB_PATH)

    MIDI_VOCAB_SIZE = len(midi_tokenizer)
    META_VOCAB_SIZE = len(metadata_vocab_map)
    MIDI_PAD_ID = midi_tokenizer[config.MIDI_PAD_TOKEN_NAME]
    META_PAD_ID = metadata_vocab_map[config.META_PAD_TOKEN_NAME]
    
    log_fn(f"MIDI Vocab Size: {MIDI_VOCAB_SIZE}, MIDI PAD ID: {MIDI_PAD_ID}")
    log_fn(f"Meta Vocab Size: {META_VOCAB_SIZE}, Meta PAD ID: {META_PAD_ID}")

    log_fn("--- Creazione Dataset e DataLoader ---")
    train_dataset = MutopiaDataset(SPLITS_DIR / "train.jsonl", midi_tokenizer, metadata_vocab_map, 
                                     config.MAX_SEQ_LEN_MIDI, config.MAX_SEQ_LEN_META, splits_dir=SPLITS_DIR)
    val_dataset = MutopiaDataset(SPLITS_DIR / "validation.jsonl", midi_tokenizer, metadata_vocab_map,
                                   config.MAX_SEQ_LEN_MIDI, config.MAX_SEQ_LEN_META, splits_dir=SPLITS_DIR)

    collate_fn_with_padding_ids = partial(pad_collate_fn, meta_pad_id=META_PAD_ID, midi_pad_id=MIDI_PAD_ID)
    
    # Crea un sampler distribuito per TPU
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=xm.xrt_world_size() if IS_TPU else 1,
        rank=xm.get_ordinal() if IS_TPU else 0, shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=xm.xrt_world_size() if IS_TPU else 1,
        rank=xm.get_ordinal() if IS_TPU else 0, shuffle=False
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, 
                                  collate_fn=collate_fn_with_padding_ids, num_workers=2, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
                                collate_fn=collate_fn_with_padding_ids, num_workers=2, drop_last=True)

    # Usa ParallelLoader per TPU per pre-caricare i dati su ogni core
    if IS_TPU:
        train_dataloader = pl.ParallelLoader(train_dataloader, [DEVICE]).per_device_loader(DEVICE)
        val_dataloader = pl.ParallelLoader(val_dataloader, [DEVICE]).per_device_loader(DEVICE)

    log_fn("--- Inizializzazione Modello ---")
    if checkpoint and 'model_params' in checkpoint:
        log_fn("Inizializzazione del modello con i parametri salvati nel checkpoint.")
        model_params_to_save = checkpoint['model_params']
    else:
        log_fn("Inizializzazione di un nuovo modello con i parametri di default.")
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

    start_epoch = 1
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience_early_stopping = 10

    if checkpoint:
        log_fn("Ripristino dello stato del modello e dell'ottimizzatore.")
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        log_fn(f"Ripresa dall'epoca {start_epoch} con best_val_loss = {best_val_loss:.4f}")

    log_fn(f"Numero parametri: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    vocab_info_to_save = {
        'midi_vocab_path': str(VOCAB_PATH), 'metadata_vocab_path': str(METADATA_VOCAB_PATH),
        'midi_tokenizer_strategy': config.MIDI_TOKENIZER_STRATEGY.__name__
    }

    log_fn("--- Inizio Addestramento ---")
    
    epoch = 0
    for epoch in range(start_epoch, EPOCHS + 1):
        # Imposta l'epoca nel sampler per garantire un rimescolamento corretto
        if IS_TPU:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            
        log_fn(f"--- Epoch {epoch}/{EPOCHS} ---")
        start_time_epoch = time.time()
        
        train_loss_epoch = train_epoch(model, optimizer, criterion, train_dataloader)
        
        # ### MODIFICA TPU: La loss viene raccolta da tutti i core
        if IS_TPU:
            train_loss_epoch = xm.mesh_reduce('train_loss_reduce', train_loss_epoch, np.mean)

        current_val_loss = float('inf')
        if len(val_dataset) > 0:
            current_val_loss = evaluate(model, criterion, val_dataloader)
            if IS_TPU:
                current_val_loss = xm.mesh_reduce('val_loss_reduce', current_val_loss, np.mean)
        
        epoch_duration_total = time.time() - start_time_epoch
        log_fn(f"Epoch {epoch}: Train Loss = {train_loss_epoch:.4f}, Val Loss = {current_val_loss:.4f} (Durata: {epoch_duration_total:.2f}s)")

        scheduler.step(current_val_loss)

        # Il salvataggio viene fatto solo dal processo master
        if (not IS_TPU or xm.is_master_ordinal()):
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                epochs_no_improve = 0
                best_model_path = MODEL_SAVE_DIR / "transformer_best.pt"
                log_fn(f"Nuova best validation loss: {best_val_loss:.4f}. Salvataggio modello in {best_model_path}")
                # xm.save gestisce correttamente il salvataggio da un device XLA
                save_payload = {
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
                    'model_params': model_params_to_save, 'vocab_info': vocab_info_to_save
                }
                if IS_TPU: xm.save(save_payload, best_model_path)
                else: torch.save(save_payload, best_model_path)
            else:
                epochs_no_improve += 1
            
            if epoch % 10 == 0:
                periodic_model_path = MODEL_SAVE_DIR / f"transformer_epoch_{epoch}.pt"
                log_fn(f"Salvataggio checkpoint periodico (epoch {epoch}): {periodic_model_path}")
                save_payload = {
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
                    'model_params': model_params_to_save, 'vocab_info': vocab_info_to_save
                }
                if IS_TPU: xm.save(save_payload, periodic_model_path)
                else: torch.save(save_payload, periodic_model_path)
        
        if epochs_no_improve >= patience_early_stopping:
            log_fn(f"Nessun miglioramento per {patience_early_stopping} epoche. Early stopping.")
            break 
            
    if not IS_TPU or xm.is_master_ordinal():
        final_model_path = MODEL_SAVE_DIR / "transformer_final.pt"
        log_fn(f"Salvataggio checkpoint finale: {final_model_path}")
        save_payload = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
            'model_params': model_params_to_save, 'vocab_info': vocab_info_to_save
        }
        if IS_TPU: xm.save(save_payload, final_model_path)
        else: torch.save(save_payload, final_model_path)

    log_fn("--- Addestramento Terminato ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Addestra un modello Transformer per la generazione musicale.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Percorso della cartella base del dataset.")
    parser.add_argument("--model_save_dir", type=Path, required=True, help="Percorso per salvare i checkpoint del modello.")
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None, help="Percorso opzionale a un checkpoint per riprendere l'addestramento.")
    args = parser.parse_args()

    if IS_TPU:
        # Avvia 8 processi, uno per ogni core TPU, eseguendo la funzione `main_training_loop`
        xmp.spawn(main_training_loop, args=(args,), nprocs=8, start_method='fork')
    else:
        # Se non è una TPU, esegui la funzione normalmente in un singolo processo
        # il rank 0 è fittizio, non viene usato
        main_training_loop(0, args)