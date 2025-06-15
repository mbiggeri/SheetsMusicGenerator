import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt
# Rimosso GradScaler e autocast da torch.cuda.amp per importarli direttamente dove servono
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


# --- USAGE (GPU/CPU): ---
# python training.py --data_dir PATH/TO/DATASET --model_save_dir PATH/TO/SAVE/MODELS
#
# --- TO RESUME TRAINING (GPU/CPU): ---
# python training.py --data_dir PATH/TO/DATASET --model_save_dir PATH/TO/SAVE/MODELS --resume_from_checkpoint PATH/TO/transformer_best.pt

# EXAMPLE: python training.py --data_dir "C:\Users\Michael\Desktop\MusicDatasets\Datasets\adl_piano_midi_octuple" --model_save_dir "C:\Users\Michael\Desktop\MusicDatasets\Datasets"

# --- Configurazione del logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configurazione / Costanti ---
# MODIFICA: Il device viene impostato in modo standard per PyTorch (GPU o CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Iperparametri del Modello e Addestramento
EPOCHS = 100
BATCH_SIZE = 64 # Adatta il batch size alla memoria della tua GPU
ACCUMULATION_STEPS = 1 # Puoi aumentarlo se il BATCH_SIZE è troppo grande per la memoria
LEARNING_RATE = 0.0001
EMB_SIZE = 128
NHEAD = 4
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DROPOUT = 0.1

# ... (le funzioni `build_or_load_tokenizer`, `load_metadata_vocab`, `MutopiaDataset`, `pad_collate_fn`, `PositionalEncoding` e `Seq2SeqTransformer` rimangono INVARIATE) ...
# Assicurati che le definizioni di queste funzioni siano presenti qui nel tuo file finale.
# Per brevità, non le ripeto, ma sono necessarie per l'esecuzione.

#------------------------
# Tokenizer e Vocabolario
#------------------------

def build_or_load_tokenizer(midi_file_paths=None, force_build=False):
    """
    Costruisce o carica il tokenizer MIDI e la sua configurazione/vocabolario,
    utilizzando i parametri centralizzati da config.py.
    """
    # Assumiamo che VOCAB_PATH sia definita globalmente più avanti
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

# --- Funzione per caricare le dimensioni dei vocabolari ---
def load_octuple_vocab_sizes(path: Path) -> dict:
    """Carica le dimensioni dei vocabolari componenti per Octuple."""
    if not path.exists():
        logging.error(f"File delle dimensioni del vocabolario Octuple non trovato in {path}.")
        raise FileNotFoundError
    with open(path, 'r') as f:
        return json.load(f)

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
                 max_len_midi_padded, max_len_meta, splits_dir):
        self.splits_dir = Path(splits_dir)
        self.midi_tokenizer = midi_tokenizer
        self.metadata_vocab_map = metadata_vocab_map
        self.max_len_midi_padded = max_len_midi_padded
        self.max_len_meta = max_len_meta

        # --- CONTROLLO STRATEGIA: OCTUPLE ---
        self.is_octuple = isinstance(self.midi_tokenizer, miditok.Octuple)
        
        # Gestione token speciali
        self.meta_pad_id = metadata_vocab_map[config.META_PAD_TOKEN_NAME]
        self.sos_meta_id = metadata_vocab_map[config.META_SOS_TOKEN_NAME]
        self.eos_meta_id = metadata_vocab_map[config.META_EOS_TOKEN_NAME]
        self.unk_meta_id = metadata_vocab_map[config.META_UNK_TOKEN_NAME]

        # Per Octuple, SOS/EOS devono essere tuple di ID.
        if self.is_octuple:
            self.sos_midi = tuple(voc[config.MIDI_SOS_TOKEN_NAME] for voc in self.midi_tokenizer.vocab)
            self.eos_midi = tuple(voc[config.MIDI_EOS_TOKEN_NAME] for voc in self.midi_tokenizer.vocab)
        else: # Per i tokenizer standard, sono singoli int
            self.sos_midi = self.midi_tokenizer[config.MIDI_SOS_TOKEN_NAME]
            self.eos_midi = self.midi_tokenizer[config.MIDI_EOS_TOKEN_NAME]


        logging.info(f"Inizializzazione Dataset. Rilevato tokenizer Octuple: {self.is_octuple}")
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    meta_tokens_str = tokenize_metadata(entry.get('metadata', {}))
                    meta_token_ids = [self.metadata_vocab_map.get(token, self.unk_meta_id) for token in meta_tokens_str]
                    src_seq_list = [self.sos_meta_id] + meta_token_ids[:self.max_len_meta-2] + [self.eos_meta_id]
                    src_tensor = torch.tensor(src_seq_list, dtype=torch.long)
                    self.samples.append((src_tensor, entry.get('token_ids_path')))
                except (json.JSONDecodeError, KeyError) as e:
                    logging.warning(f"Skipping malformed line in {jsonl_path}: {e}")

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

        # Tronca alla lunghezza massima
        truncated_midi_ids = raw_midi_ids_from_file[:self.max_len_midi_padded-2]
        
        # --- CONTROLLO STRATEGIA: OCTUPLE ---
        if self.is_octuple:
            sos_array = np.array([self.sos_midi], dtype=np.int64)
            eos_array = np.array([self.eos_midi], dtype=np.int64)
            tgt_seq_array = np.vstack([sos_array, truncated_midi_ids, eos_array])
            tgt_tensor = torch.tensor(tgt_seq_array, dtype=torch.long)
        else: # Percorso per tokenizer standard (REMI, etc.)
            tgt_seq_list = [self.sos_midi] + truncated_midi_ids.tolist() + [self.eos_midi]
            tgt_tensor = torch.tensor(tgt_seq_list, dtype=torch.long)
            
        return src_tensor, tgt_tensor

def pad_collate_fn_standard(batch, meta_pad_id, midi_pad_id):
    batch = [item for item in batch if item is not None]
    if not batch: return None, None, None, None
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=meta_pad_id)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=midi_pad_id)
    src_padding_mask = (src_padded == meta_pad_id)
    tgt_padding_mask = (tgt_padded == midi_pad_id)
    return src_padded, tgt_padded, src_padding_mask, tgt_padding_mask

def pad_collate_fn_octuple(batch, meta_pad_id, midi_pad_tuple):
    batch = [item for item in batch if item is not None]
    if not batch: return None, None, None, None
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=meta_pad_id)
    src_padding_mask = (src_padded == meta_pad_id)
    max_len = max(t.size(0) for t in tgt_batch)
    padded_tgt_tensors = []
    for t in tgt_batch:
        num_pads = max_len - t.size(0)
        padding = torch.tensor([midi_pad_tuple] * num_pads, dtype=torch.long) if num_pads > 0 else []
        padded_t = torch.cat([t, padding], dim=0) if num_pads > 0 else t
        padded_tgt_tensors.append(padded_t)
    tgt_padded = torch.stack(padded_tgt_tensors)
    tgt_padding_mask = (tgt_padded[:, :, 0] == midi_pad_tuple[0])
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

# Modello specifico per Octuple
class Seq2SeqTransformerOctuple(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead,
                 src_vocab_size, tgt_vocab_sizes: dict, max_pe_len,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.vocab_names = list(tgt_vocab_sizes.keys())
        self.num_components = len(self.vocab_names)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.ModuleList([nn.Embedding(tgt_vocab_sizes[name], emb_size) for name in self.vocab_names])
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=max_pe_len)
        self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.generator = nn.ModuleList([nn.Linear(emb_size, tgt_vocab_sizes[name]) for name in self.vocab_names])

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, memory_key_padding_mask=None, tgt_mask=None):
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        tgt_emb_sum = torch.zeros(tgt.size(0), tgt.size(1), self.emb_size, device=tgt.device)
        for i in range(self.num_components):
            tgt_emb_sum += self.tgt_tok_emb[i](tgt[:, :, i])
        tgt_emb = self.positional_encoding(tgt_emb_sum * math.sqrt(self.emb_size))
        if tgt_mask is None:
            tgt_len = tgt.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool), diagonal=1)
        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_padding_mask
        outs = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        logits = [gen(outs) for gen in self.generator]
        return logits

#------------------------
# Ciclo di Addestramento e Valutazione
#------------------------
def train_epoch(model, optimizer, criterions, train_dataloader, is_octuple):
    model.train()
    total_loss = 0
    # Rimuovi optimizer.zero_grad() da qui
    progress_bar = tqdm(train_dataloader, desc="Training Epoch", leave=False)

    for i, batch_data in enumerate(progress_bar): # Aggiungi enumerate
        if batch_data[0] is None: continue
        
        src, tgt, src_padding_mask, tgt_padding_mask = [d.to(DEVICE) for d in batch_data]

        # La logica della loss rimane invariata...
        if is_octuple:
            # ... (codice per la loss di octuple)
            tgt_input = tgt[:, :-1, :]
            tgt_out = tgt[:, 1:, :]
            tgt_input_padding_mask = tgt_padding_mask[:, :-1]
            logits_list = model(src=src, tgt=tgt_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_input_padding_mask, memory_key_padding_mask=src_padding_mask)
            
            batch_loss = 0
            for j in range(len(logits_list)): # Usa 'j' per non confondere con l'indice del batch 'i'
                loss_j = criterions[j](logits_list[j].reshape(-1, logits_list[j].shape[-1]), tgt_out[:, :, j].reshape(-1))
                batch_loss += loss_j
            batch_loss = batch_loss / len(logits_list)
        else:
            # ... (codice per la loss standard)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_input_padding_mask = tgt_padding_mask[:, :-1]
            logits = model(src=src, tgt=tgt_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_input_padding_mask, memory_key_padding_mask=src_padding_mask)
            batch_loss = criterions(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        # --- NUOVA LOGICA DI ACCUMULO ---
        # Normalizza la loss per lo step di accumulo
        batch_loss = batch_loss / ACCUMULATION_STEPS
        
        batch_loss.backward()
        
        # Limita la norma dei gradienti per prevenire l'esplosione e stabilizzare il training.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Esegui l'aggiornamento dei pesi solo ogni ACCUMULATION_STEPS
        if (i + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Aggiorna la loss totale (rimuovi la normalizzazione per il logging)
        total_loss += batch_loss.item() * ACCUMULATION_STEPS
        progress_bar.set_postfix({'train_loss': f'{total_loss / (progress_bar.n + 1):.4f}'})

    # --- AGGIUNTA OPZIONALE: Gestisci l'ultimo step se non è un multiplo ---
    if (i + 1) % ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(train_dataloader)

def evaluate(model, criterions, val_dataloader, is_octuple):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_data in tqdm(val_dataloader, desc="Evaluation", leave=False):
            if batch_data[0] is None: continue
            src, tgt, src_padding_mask, tgt_padding_mask = [d.to(DEVICE) for d in batch_data]
            
            # --- CONTROLLO STRATEGIA: OCTUPLE ---
            if is_octuple:
                tgt_input = tgt[:, :-1, :]
                tgt_out = tgt[:, 1:, :]
                tgt_input_padding_mask = tgt_padding_mask[:, :-1]
                logits_list = model(src=src, tgt=tgt_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_input_padding_mask, memory_key_padding_mask=src_padding_mask)
                batch_loss = 0
                for i in range(len(logits_list)):
                    loss_i = criterions[i](logits_list[i].reshape(-1, logits_list[i].shape[-1]), tgt_out[:, :, i].reshape(-1))
                    batch_loss += loss_i
                batch_loss = batch_loss / len(logits_list)
            else: # --- PERCORSO PER TOKENIZER STANDARD (REMI, etc.) ---
                tgt_input = tgt[:, :-1]
                tgt_out = tgt[:, 1:]
                tgt_input_padding_mask = tgt_padding_mask[:, :-1]
                logits = model(src=src, tgt=tgt_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_input_padding_mask, memory_key_padding_mask=src_padding_mask)
                batch_loss = criterions(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            
            total_loss += batch_loss.item()
            
    return total_loss / len(val_dataloader)

# ------------------------
# Learning Rate Finder
# ------------------------
class _LRFModelWrapper(torch.nn.Module):
    """Wraps the model to be compatible with LRFinder's model(inputs) call."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_dict):
        # Unpack the dictionary and call the real model with the correct arguments.
        return self.model(
            src=inputs_dict["src"],
            tgt=inputs_dict["tgt"],
            src_padding_mask=inputs_dict["src_padding_mask"],
            tgt_padding_mask=inputs_dict["tgt_padding_mask"],
            memory_key_padding_mask=inputs_dict["src_padding_mask"]
        )

class _LRFCriterionWrapper(torch.nn.Module):
    """Wraps the criterion to be compatible with LRFinder's criterion(output, labels) call."""
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, model_output, labels):
        # model_output is 'logits', labels is 'tgt_out'.
        # Reshape them as expected by the original criterion.
        return self.criterion(model_output.reshape(-1, model_output.shape[-1]), labels.reshape(-1))

def _lr_finder_collate_fn(batch, meta_pad_id, midi_pad_id):
    """
    Custom collate_fn to package a batch into the (inputs, labels) format
    that LRFinder expects. Tensors are kept on the CPU.
    """
    # Use the original padding logic
    src_padded, tgt_padded, src_padding_mask, tgt_padding_mask = pad_collate_fn_standard(batch, meta_pad_id, midi_pad_id)

    if src_padded is None:
        return None, None

    # Tensors remain on the CPU. The LRFinder object will move them to the device.

    # Prepare model inputs and loss targets from the batch
    tgt_input = tgt_padded[:, :-1]
    tgt_out = tgt_padded[:, 1:]
    tgt_input_padding_mask = tgt_padding_mask[:, :-1]

    # 'inputs' is a dictionary containing everything the model's forward pass needs.
    inputs = {
        "src": src_padded,
        "tgt": tgt_input,
        "src_padding_mask": src_padding_mask,
        "tgt_padding_mask": tgt_input_padding_mask,
    }
    # 'labels' is what the criterion will compare against the model's output.
    labels = tgt_out

    return inputs, labels

def find_best_lr(model, optimizer, criterion, train_dataloader, device, model_save_dir, meta_pad_id, midi_pad_id):
    """
    Esegue il test del range del learning rate e salva il grafico.
    This version is adapted for a Seq2Seq model by using wrapper classes.
    """
    # 1. Wrap the model and criterion for compatibility
    wrapped_model = _LRFModelWrapper(model)
    wrapped_criterion = _LRFCriterionWrapper(criterion)

    # 2. Create a new DataLoader with the custom collate function
    train_dataset = train_dataloader.dataset
    batch_size = train_dataloader.batch_size
    lr_finder_collate = partial(_lr_finder_collate_fn, meta_pad_id=meta_pad_id, midi_pad_id=midi_pad_id)
    
    finder_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   collate_fn=lr_finder_collate, num_workers=0, drop_last=True)

    # 3. Initialize LRFinder with the wrapped components
    # The optimizer's parameters are already those of the original model, which is correct.
    lr_finder = LRFinder(wrapped_model, optimizer, wrapped_criterion, device=device)
    
    logging.info("Avvio del test del range del learning rate...")
    lr_finder.range_test(finder_dataloader, end_lr=1, num_iter=len(finder_dataloader), step_mode="exp")

    # The rest of the function remains the same
    min_loss_lr = lr_finder.history['lr'][lr_finder.history['loss'].index(lr_finder.best_loss)]
    suggested_lr = min_loss_lr / 10

    logging.info(f"Test del learning rate completato.")
    logging.info(f"Learning rate con la perdita minima: {min_loss_lr:.2e}")
    logging.info(f"Learning rate suggerito (un ordine di grandezza in meno): {suggested_lr:.2e}")

    plot_path = model_save_dir / "lr_finder_plot.png"
    logging.info(f"Salvataggio del grafico del learning rate finder in: {plot_path}")
    lr_finder.plot()
    plt.savefig(plot_path)
    plt.close()

    lr_finder.reset()

    return suggested_lr

#------------------------ 
# Esecuzione Principale
#------------------------
def main_training_loop(args):
    """
    Funzione principale che orchestra l'intero processo di addestramento.
    È resa adattabile alla strategia di tokenizzazione scelta in config.py.
    """
    # Rendi le variabili globali del path accessibili se necessario, anche se è meglio passarle
    global VOCAB_PATH, METADATA_VOCAB_PATH

    # 1. IMPOSTAZIONE DEI PERCORSI
    DATA_DIR = args.data_dir
    MODEL_SAVE_DIR = args.model_save_dir
    SPLITS_DIR = DATA_DIR / "dataset_splits"
    VOCAB_PATH = DATA_DIR / "midi_vocab.json"
    METADATA_VOCAB_PATH = DATA_DIR / "metadata_vocab.json"
    
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Device in uso: {DEVICE}")
    logging.info(f"I modelli verranno salvati in: {MODEL_SAVE_DIR}")

    # 2. CARICAMENTO TOKENIZER E DETERMINAZIONE DELLA STRATEGIA
    logging.info("--- Preparazione Tokenizer e Vocabolari ---")
    midi_tokenizer = build_or_load_tokenizer(VOCAB_PATH)
    metadata_vocab_map = load_metadata_vocab(METADATA_VOCAB_PATH)
    
    # Questo è il nostro "selettore" di strategia. Controlliamo il tipo di tokenizer.
    is_octuple = isinstance(midi_tokenizer, miditok.Octuple)
    logging.info(f"Strategia Tokenizer Rilevata: {'Octuple' if is_octuple else 'Standard (es. REMI)'}")

    # 3. SETUP CONDIZIONALE BASATO SULLA STRATEGIA
    # Prepariamo tutti gli "ingredienti" specifici per la strategia scelta.
    if is_octuple:
        logging.info("Configurazione per la strategia Octuple...")
        # Carica le dimensioni dei sotto-vocabolari di Octuple
        MIDI_VOCAB_SIZES_PATH = config.get_project_paths(DATA_DIR)["midi_vocab_sizes"]
        tgt_vocab_sizes = load_octuple_vocab_sizes(MIDI_VOCAB_SIZES_PATH)
        vocab_names_ordered = list(tgt_vocab_sizes.keys())
        
        # --- CORREZIONE APPLICATA QUI ---
        # Costruisci manualmente la tupla degli ID di padding iterando su ogni sotto-vocabolario.
        # midi_tokenizer.vocab per Octuple è una lista di dizionari (uno per ogni dimensione).
        MIDI_PAD_ID = tuple(voc[config.MIDI_PAD_TOKEN_NAME] for voc in midi_tokenizer.vocab)
        logging.info(f"ID di padding per Octuple costruito correttamente: {MIDI_PAD_ID}")
        
        # Scegli la classe del modello e la funzione collate corrette
        ModelClass = Seq2SeqTransformerOctuple
        collate_fn = partial(pad_collate_fn_octuple, meta_pad_id=metadata_vocab_map[config.META_PAD_TOKEN_NAME], midi_pad_tuple=MIDI_PAD_ID)
        
        # Prepara i parametri specifici per il modello Octuple
        model_params_specific = {'tgt_vocab_sizes': tgt_vocab_sizes}
        
        # Il criterio di loss è una lista di loss, una per ogni componente della tupla
        criterions = [nn.CrossEntropyLoss(ignore_index=MIDI_PAD_ID[i]) for i in range(len(vocab_names_ordered))]
    
    else: # --- Percorso per tokenizer Standard (REMI, etc.) ---
        logging.info("Configurazione per la strategia Standard (REMI, Structured, etc.)...")
        # L'ID di PAD è un singolo intero
        MIDI_PAD_ID = midi_tokenizer[config.MIDI_PAD_TOKEN_NAME]
        
        # Scegli la classe del modello e la funzione collate standard
        ModelClass = Seq2SeqTransformer
        collate_fn = partial(pad_collate_fn_standard, meta_pad_id=metadata_vocab_map[config.META_PAD_TOKEN_NAME], midi_pad_id=MIDI_PAD_ID)
        
        # Prepara i parametri specifici per il modello standard
        model_params_specific = {'tgt_vocab_size': len(midi_tokenizer)}
        
        # Il criterio di loss è uno solo
        criterions = nn.CrossEntropyLoss(ignore_index=MIDI_PAD_ID)

    # 4. CREAZIONE DATASET E DATALOADER (ora usano la collate_fn corretta)
    logging.info("--- Creazione Dataset e DataLoader ---")
    train_dataset = MutopiaDataset(SPLITS_DIR / "train.jsonl", midi_tokenizer, metadata_vocab_map, config.MAX_SEQ_LEN_MIDI, config.MAX_SEQ_LEN_META, splits_dir=SPLITS_DIR)
    val_dataset = MutopiaDataset(SPLITS_DIR / "validation.jsonl", midi_tokenizer, metadata_vocab_map, config.MAX_SEQ_LEN_MIDI, config.MAX_SEQ_LEN_META, splits_dir=SPLITS_DIR)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 5. INIZIALIZZAZIONE MODELLO E OTTIMIZZATORE
    logging.info("--- Inizializzazione Modello ---")
    # Parametri di base, comuni a entrambi i modelli
    max_pe_len_calculated = max(config.MAX_SEQ_LEN_MIDI, config.MAX_SEQ_LEN_META) + 100
    model_params_base = {
        'num_encoder_layers': NUM_ENCODER_LAYERS, 'num_decoder_layers': NUM_DECODER_LAYERS,
        'emb_size': EMB_SIZE, 'nhead': NHEAD, 'src_vocab_size': len(metadata_vocab_map),
        'dim_feedforward': FFN_HID_DIM, 'dropout': DROPOUT,
        'max_pe_len': max_pe_len_calculated
    }
    # Uniamo i parametri di base con quelli specifici della strategia
    model_params_to_save = {**model_params_base, **model_params_specific}
    
    model = ModelClass(**model_params_to_save).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # Logica per riprendere da un checkpoint (invariata)
    start_epoch = 1
    best_val_loss = float('inf')
    if args.resume_from_checkpoint and args.resume_from_checkpoint.exists():
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logging.info(f"Ripresa dall'epoca {start_epoch} con best_val_loss = {best_val_loss:.4f}")

    logging.info(f"Numero parametri: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 6. CICLO DI TRAINING PRINCIPALE
    logging.info("--- Inizio Addestramento ---")
    for epoch in range(start_epoch, EPOCHS + 1):
        logging.info(f"--- Epoch {epoch}/{EPOCHS} ---")
        
        # Le funzioni di training e valutazione ora ricevono il flag 'is_octuple'
        # per sapere quale logica interna applicare.
        train_loss = train_epoch(model, optimizer, criterions, train_dataloader, is_octuple)
        val_loss = evaluate(model, criterions, val_dataloader, is_octuple)
        
        scheduler.step(val_loss)
        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Logica di salvataggio (invariata)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = MODEL_SAVE_DIR / "transformer_best.pt"
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
                'model_params': model_params_to_save
            }, save_path)
            logging.info(f"Nuova best validation loss. Modello salvato in {save_path}")

    logging.info("--- Addestramento Terminato ---")

if __name__ == "__main__":
    # Add this block to handle CUDA multiprocessing
    import torch.multiprocessing as mp
    if torch.cuda.is_available():
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    parser = argparse.ArgumentParser(description="Addestra un modello Transformer per la generazione musicale.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Percorso della cartella base del dataset.")
    parser.add_argument("--model_save_dir", type=Path, required=True, help="Percorso per salvare i checkpoint del modello.")
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None, help="Percorso opzionale a un checkpoint per riprendere l'addestramento.")
    parser.add_argument("--find_lr", action="store_true", help="Esegui il test del range del learning rate e termina.")
    args = parser.parse_args()

    # MODIFICA: Esecuzione diretta della funzione di training, senza logica TPU
    main_training_loop(args)