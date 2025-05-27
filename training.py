import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import gc # Importa il modulo garbage collection
import miditok # Assicurati di installare miditok
from pathlib import Path
import json
import math
import random
import logging
from tqdm import tqdm
import os
import time
import sys
from functools import partial # IMPORTANTE: Aggiunto per DataLoader collate_fn
from symusic import Score

# --- Configurazione del logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configurazione / Costanti ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Percorsi base per i dataset
DATA_DIR_MUTOPIA = Path("/content/SheetsMusicGenerator/mutopia_data")
DATA_DIR_MAGICMIDI = Path("/content/SheetsMusicGenerator/The_Magic_Of_MIDI") # Come specificato

# Imposta DATA_DIR di default (mutopia)
DATA_DIR = DATA_DIR_MUTOPIA
dataset_name_chosen = "mutopia (default)"

# Controlla gli argomenti da riga di comando
if "-magicmidi" in sys.argv:
    DATA_DIR = DATA_DIR_MAGICMIDI
    dataset_name_chosen = "The_Magic_Of_MIDI"
elif "-mutopia" in sys.argv:
    DATA_DIR = DATA_DIR_MUTOPIA # Esplicito, anche se è il default
    dataset_name_chosen = "mutopia"

logging.info(f"Utilizzo del dataset: {dataset_name_chosen} ({DATA_DIR})")

SPLITS_DIR = DATA_DIR / "dataset_splits" # Directory con train/validation/test.jsonl
MIDI_BASE_DIR = DATA_DIR # Directory radice dove cercare i midi_relative_path

# Definisci il percorso su Google Drive dove vuoi salvare i modelli
# Assicurati che la cartella "IlMioProgettoModelli" (o come vuoi chiamarla)
# esista nel tuo Google Drive, oppure MODEL_SAVE_DIR.mkdir la creerà.
DRIVE_MOUNT_POINT = Path("/content/drive/MyDrive/")
MODEL_SAVE_DIR = DRIVE_MOUNT_POINT / "SheetsMusicGenerator_Models" # Esempio: Salva in una cartella "SheetsMusicGenerator_Models" su Drive
# --- FINE MODIFICA PER MODEL_SAVE_DIR ---

# Crea directory se non esistono
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Configurazioni Tokenizer MIDI (scegliere una strategia)
MIDI_TOKENIZER_STRATEGY = miditok.REMI # Esempio scelto
MIDI_VOCAB_TARGET_SIZE = 50000 # Esempio: Dimensione target per il vocabolario MIDI se addestrato

VOCAB_PATH = DATA_DIR / "midi_vocab.json" # Dove salvare/caricare il vocabolario MIDI
METADATA_VOCAB_PATH = DATA_DIR / "metadata_vocab.json" # Vocabolario per i token metadati

# Token Speciali MIDI (allineati con le convenzioni di miditok)
MIDI_PAD_TOKEN_NAME = "PAD_None"
MIDI_SOS_TOKEN_NAME = "SOS_None"
MIDI_EOS_TOKEN_NAME = "EOS_None"
MIDI_UNK_TOKEN_NAME = "UNK_None"

# Token Speciali per Metadati (possono rimanere custom)
META_PAD_TOKEN_NAME = "<pad_meta>"
META_UNK_TOKEN_NAME = "<unk_meta>"
META_SOS_TOKEN_NAME = "<sos_meta>"
META_EOS_TOKEN_NAME = "<eos_meta>"

# Iperparametri del Modello e Addestramento (Esempi!)
EPOCHS = 25
BATCH_SIZE = 256 # Riduci se hai poca memoria GPU
LEARNING_RATE = 0.0001
EMB_SIZE = 256 # Dimensione embedding
NHEAD = 4 # Numero di head nell'attention (deve dividere EMB_SIZE)
FFN_HID_DIM = 512 # Dimensione layer nascosto FeedForward
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT = 0.1
MAX_SEQ_LEN_MIDI = 1024 # Lunghezza massima sequenza MIDI (tronca/scarta se più lunga)
MAX_SEQ_LEN_META = 128 # Aumentata per includere potenziale titolo lungo

# Programmi MIDI considerati come "pianoforte"
PIANO_PROGRAMS = list(range(0, 8))

# --- NUOVI IPERPARAMETRI PER MODALITÀ DI PROCESSSAMENTO ---
# PROCESSING_MODE = "piano_only"
PROCESSING_MODE = "multi_instrument_stream"

# Setup Logging (opzionale ma utile)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#------------------------
# Tokenizer e Vocabolario
#------------------------

def build_or_load_tokenizer(midi_file_paths=None, force_build=False):
    """Costruisce o carica il tokenizer MIDI e la sua configurazione/vocabolario."""
    special_tokens = [MIDI_PAD_TOKEN_NAME, MIDI_SOS_TOKEN_NAME, MIDI_EOS_TOKEN_NAME, MIDI_UNK_TOKEN_NAME]

    if VOCAB_PATH.exists() and not force_build:
        logging.info(f"Caricamento configurazione tokenizer MIDI da {VOCAB_PATH}")
        try:
            tokenizer = MIDI_TOKENIZER_STRATEGY(params=str(VOCAB_PATH))
            logging.info(f"Tokenizer caricato con successo da {VOCAB_PATH}")
        except Exception as e:
             logging.error(f"Errore nel caricare parametri tokenizer da {VOCAB_PATH}. Errore: {e}", exc_info=True)
             logging.info("Tentativo di ricostruire il tokenizer da zero.")
             return build_or_load_tokenizer(midi_file_paths=midi_file_paths, force_build=True)
    else:
        logging.info("Creazione nuova configurazione tokenizer MIDI...")
        tokenizer_config_special_tokens = [MIDI_PAD_TOKEN_NAME, MIDI_SOS_TOKEN_NAME, MIDI_EOS_TOKEN_NAME, MIDI_UNK_TOKEN_NAME]
        
        tokenizer_params = miditok.TokenizerConfig(
            special_tokens=tokenizer_config_special_tokens,
            use_programs=True,
            one_token_stream_for_programs=True,
            program_changes=True,
            use_chords=True,
            use_rests=True,
            use_tempos=True,
            use_time_signatures=True,
            use_velocities=True,
        )
        
        tokenizer = MIDI_TOKENIZER_STRATEGY(tokenizer_config=tokenizer_params)
        logging.info(f"Tokenizer {MIDI_TOKENIZER_STRATEGY.__name__} inizializzato con use_programs=True, one_token_stream_for_programs=True.")

        if midi_file_paths:
            # if hasattr(tokenizer, 'train'):      Esegui questa se vuoi che venga costruito un vocabolario finale più piccolo con BPE o simili a partire da quello generato con la strategy
            logging.info(f"Numero di file MIDI forniti per l'addestramento: {len(midi_file_paths)}")    
            if MIDI_TOKENIZER_STRATEGY != miditok.TSD and MIDI_TOKENIZER_STRATEGY != miditok.REMI and hasattr(tokenizer, 'train'):          
                logging.info(f"Entro nel blocco if hasattr(tokenizer, 'train')") # Conferma
                try:
                    logging.info(f"Dimensione vocabolario PRIMA di tokenizer.train: {len(tokenizer)}")
                    tokenizer.train(vocab_size=MIDI_VOCAB_TARGET_SIZE, files_paths=midi_file_paths)
                    logging.info(f"Dimensione vocabolario DOPO tokenizer.train: {len(tokenizer)}")
                except Exception as e:
                    logging.error(f"Errore durante tokenizer.train: {e}", exc_info=True)
            else:
                logging.info(f"Il tokenizer {MIDI_TOKENIZER_STRATEGY.__name__} non ha il metodo .train()")
        else:
            logging.warning("Nessun file MIDI fornito per l'addestramento del tokenizer durante la costruzione. "
                            "Il vocabolario potrebbe essere subottimale se l'addestramento (es. BPE) è necessario per la strategia scelta.")

        logging.info(f"Salvataggio configurazione tokenizer MIDI in {VOCAB_PATH}")
        VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(VOCAB_PATH))

    logging.info(f"Dimensione vocabolario MIDI (dopo caricamento/costruzione): {len(tokenizer)}")

    missing_ids_info = {}
    try:
        pad_id_check = tokenizer[MIDI_PAD_TOKEN_NAME]
        sos_id_check = tokenizer[MIDI_SOS_TOKEN_NAME]
        eos_id_check = tokenizer[MIDI_EOS_TOKEN_NAME]
        unk_id_check = tokenizer[MIDI_UNK_TOKEN_NAME]
        logging.info(f"ID Token Speciali MIDI recuperati - PAD: {pad_id_check}, SOS: {sos_id_check}, EOS: {eos_id_check}, UNK: {unk_id_check}")

        if pad_id_check is None: missing_ids_info[MIDI_PAD_TOKEN_NAME] = "Non trovato (None)"
        # Continua per altri token speciali...
    except Exception as e:
        logging.error(f"Errore durante l'accesso agli ID dei token speciali MIDI: {e}", exc_info=True)
        missing_ids_info["ERRORE GENERALE RECUPERO ID MIDI"] = str(e)

    if missing_ids_info:
         logging.error(f"ERRORE CRITICO: Impossibile recuperare gli ID per i seguenti token speciali MIDI: {missing_ids_info}")
         sys.exit(1)
    return tokenizer

def tokenize_metadata(metadata_dict):
    tokens = []
    if 'style' in metadata_dict and metadata_dict['style']:
        tokens.append(f"Style={metadata_dict['style'].replace(' ', '_')}")
    if 'key' in metadata_dict and metadata_dict['key']:
        tokens.append(f"Key={metadata_dict['key'].replace(' ', '_')}")
    if 'time_signature' in metadata_dict and metadata_dict['time_signature']:
        tokens.append(f"TimeSig={metadata_dict['time_signature']}")
    if 'title' in metadata_dict and metadata_dict['title']:
        clean_title = metadata_dict['title'].replace(' ', '_').lower()
        clean_title = ''.join(c for c in clean_title if c.isalnum() or c == '_')
        tokens.append(f"Title={clean_title[:30]}")
    return tokens

def build_or_load_metadata_vocab(all_metadata_examples, force_build=False):
    if METADATA_VOCAB_PATH.exists() and not force_build:
        logging.info(f"Caricamento vocabolario Metadati da {METADATA_VOCAB_PATH}")
        with open(METADATA_VOCAB_PATH, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        token_to_id = vocab_data['token_to_id']
        required_specials = [META_PAD_TOKEN_NAME, META_UNK_TOKEN_NAME, META_SOS_TOKEN_NAME, META_EOS_TOKEN_NAME]
        missing = [t for t in required_specials if t not in token_to_id]
        if missing:
             logging.warning(f"Token speciali metadati mancanti nel file caricato: {missing}. Ricostruisco.")
             return build_or_load_metadata_vocab(all_metadata_examples, force_build=True)
        id_to_token = {i: t for t, i in token_to_id.items()}
        return token_to_id, id_to_token
    else:
        logging.info("Creazione nuovo vocabolario Metadati...")
        metadata_tokens = set()
        for meta_dict in all_metadata_examples:
            tokens = tokenize_metadata(meta_dict)
            metadata_tokens.update(tokens)

        all_tokens_list = [META_PAD_TOKEN_NAME, META_UNK_TOKEN_NAME, META_SOS_TOKEN_NAME, META_EOS_TOKEN_NAME] + sorted(list(metadata_tokens))
        token_to_id = {token: i for i, token in enumerate(all_tokens_list)}
        id_to_token = {i: token for token, i in token_to_id.items()}

        if token_to_id[META_PAD_TOKEN_NAME] != 0:
            logging.warning(f"ID del META_PAD_TOKEN_NAME ({META_PAD_TOKEN_NAME}) non è 0. Riassegno gli ID per coerenza con ignore_index.")
            # Riassegna per avere PAD = 0
            pad_tok = META_PAD_TOKEN_NAME
            other_specials = [t for t in [META_UNK_TOKEN_NAME, META_SOS_TOKEN_NAME, META_EOS_TOKEN_NAME] if t in token_to_id] # quelli già presenti
            unique_metadata_tokens_sorted = sorted(list(metadata_tokens))
            
            all_tokens_reordered = [pad_tok] + \
                                   [s for s in other_specials if s != pad_tok] + \
                                   [mt for mt in unique_metadata_tokens_sorted if mt not in [pad_tok] + other_specials]
            # Assicura che tutti i token originali siano presenti, specialmente se un token metadato avesse lo stesso nome di uno speciale
            final_token_set = set(all_tokens_reordered)
            for special_tok_defined in [META_UNK_TOKEN_NAME, META_SOS_TOKEN_NAME, META_EOS_TOKEN_NAME]:
                if special_tok_defined not in final_token_set:
                    all_tokens_reordered.append(special_tok_defined)
            
            # Rimuovi duplicati mantenendo l'ordine (prima occorrenza)
            seen = set()
            all_tokens_final_unique_ordered = []
            for item in all_tokens_reordered:
                if item not in seen:
                    seen.add(item)
                    all_tokens_final_unique_ordered.append(item)

            token_to_id = {token: i for i, token in enumerate(all_tokens_final_unique_ordered)}
            id_to_token = {i: token for token, i in token_to_id.items()}
            logging.info(f"Nuovo ID META_PAD_TOKEN_NAME: {token_to_id.get(META_PAD_TOKEN_NAME, 'NON TROVATO DOPO RIORDINO')}")


        vocab_data = {'token_to_id': token_to_id, 'id_to_token': id_to_token}
        logging.info(f"Salvataggio vocabolario Metadati in {METADATA_VOCAB_PATH}")
        METADATA_VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(METADATA_VOCAB_PATH, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        logging.info(f"Dimensione vocabolario Metadati (incl. speciali): {len(token_to_id)}")
        return token_to_id, id_to_token

#------------------------
# Dataset e DataLoader
#------------------------

class MutopiaDataset(Dataset):
    def __init__(self, jsonl_path, midi_base_dir, midi_tokenizer, metadata_vocab_map,
                 max_len_midi, max_len_meta, filter_key=None):
        self.midi_base_dir = Path(midi_base_dir)
        self.midi_tokenizer = midi_tokenizer
        self.metadata_vocab_map = metadata_vocab_map
        self.max_len_midi = max_len_midi
        self.max_len_meta = max_len_meta
        self.filter_key = filter_key

        try:
            self.meta_pad_id = metadata_vocab_map[META_PAD_TOKEN_NAME]
            self.sos_meta_id = metadata_vocab_map[META_SOS_TOKEN_NAME]
            self.eos_meta_id = metadata_vocab_map[META_EOS_TOKEN_NAME]
            self.unk_meta_id = metadata_vocab_map[META_UNK_TOKEN_NAME]
            self.sos_midi_id = midi_tokenizer[MIDI_SOS_TOKEN_NAME]
            self.eos_midi_id = midi_tokenizer[MIDI_EOS_TOKEN_NAME]
            self.midi_pad_id = midi_tokenizer[MIDI_PAD_TOKEN_NAME]
            if None in [self.sos_midi_id, self.eos_midi_id, self.midi_pad_id,
                        self.meta_pad_id, self.sos_meta_id, self.eos_meta_id, self.unk_meta_id]:
                raise ValueError("Uno o più ID di token speciali non sono stati trovati.")
        except (KeyError, ValueError) as e:
            logging.error(f"ERRORE CRITICO in Dataset __init__: Token speciale non trovato: {e}")
            raise

        logging.info(f"Caricamento dati da {jsonl_path} per Dataset...")
        self.data = []
        skipped_missing_midi_file = 0
        # ... (altri contatori di skip)

        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        entry = json.loads(line)
                        current_metadata = entry.get('metadata', {})
                        if self.filter_key and current_metadata.get('key') != self.filter_key:
                            continue # skipped_key_filter += 1
                        midi_relative_path_str = current_metadata.get('midi_relative_path')
                        if not midi_relative_path_str:
                            continue # skipped_no_relative_path += 1
                        midi_relative_path_str = midi_relative_path_str.replace("\\", "/") # Assicura che sia in formato Unix
                        midi_path_check = self.midi_base_dir / midi_relative_path_str
                        if midi_path_check.exists() and midi_path_check.is_file():
                            self.data.append(entry)
                        else:
                            skipped_missing_midi_file += 1
                    except json.JSONDecodeError:
                        pass # skipped_json_decode += 1
                    except Exception:
                         pass # skipped_other_entry_error += 1
        except FileNotFoundError:
             logging.error(f"File dataset non trovato: {jsonl_path}")
             raise
        logging.info(f"Caricati {len(self.data)} campioni validi. Saltati per MIDI mancante: {skipped_missing_midi_file}")
        if not self.data:
            logging.error(f"Nessun dato caricato da {jsonl_path}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        max_retries = 2
        entry = self.data[idx]
        actual_metadata = entry.get('metadata', {})
        midi_relative_path_str = actual_metadata.get('midi_relative_path')
        if not midi_relative_path_str: return None
        midi_relative_path_str = midi_relative_path_str.replace("\\", "/") # Assicura che sia in formato Unix
        midi_full_path = self.midi_base_dir / midi_relative_path_str

        for attempt in range(max_retries):
            try:
                score = Score(str(midi_full_path))
                if PROCESSING_MODE == "piano_only":
                    piano_tracks_presenti = [track for track in score.tracks if track.program in PIANO_PROGRAMS]
                    if not piano_tracks_presenti: return None
                    score.tracks = piano_tracks_presenti
                    if len(score.tracks) == 0: return None
                
                meta_tokens_str = tokenize_metadata(actual_metadata)
                meta_token_ids = [self.metadata_vocab_map.get(token, self.unk_meta_id) for token in meta_tokens_str]
                src_seq = [self.sos_meta_id] + meta_token_ids[:self.max_len_meta-2] + [self.eos_meta_id]
                
                midi_tokens_output = self.midi_tokenizer(score) 
                if not hasattr(midi_tokens_output, 'ids'): raise RuntimeError("Output tokenizer MIDI malformato")
                raw_midi_ids = midi_tokens_output.ids
                if raw_midi_ids is None or not isinstance(raw_midi_ids, list): raise RuntimeError("raw_midi_ids non validi")

                processed_midi_ids = []
                if len(raw_midi_ids) > 0:
                    if isinstance(raw_midi_ids[0], int): processed_midi_ids = raw_midi_ids
                    elif isinstance(raw_midi_ids[0], list): # Caso inatteso per REMI
                        processed_midi_ids = [item for sublist in raw_midi_ids for item in sublist if isinstance(sublist, list)]
                    else: raise RuntimeError("Formato ID MIDI inatteso.")
                
                if not processed_midi_ids: return None 
                tgt_seq = [self.sos_midi_id] + processed_midi_ids[:self.max_len_midi-2] + [self.eos_midi_id]
                return torch.tensor(src_seq, dtype=torch.long), torch.tensor(tgt_seq, dtype=torch.long)
            except Exception:
                if attempt + 1 == max_retries: return None
        return None

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
    def __init__(self, d_model, dropout=0.1, max_len=5000): # max_len qui è il default della classe
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead,
                 src_vocab_size, tgt_vocab_size, max_pe_len, # Aggiunto max_pe_len
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size # Salva per la scalatura
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        # Usa max_pe_len passato al costruttore
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
    progress_bar = tqdm(train_dataloader, desc="Training Epoch", leave=False)

    for batch_data in progress_bar:
        if batch_data[0] is None: continue
        src, tgt, src_padding_mask, tgt_padding_mask = batch_data
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        src_padding_mask, tgt_padding_mask = src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_input_padding_mask = tgt_padding_mask[:, :-1]
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src=src, tgt=tgt_input,
                       src_padding_mask=src_padding_mask,
                       tgt_padding_mask=tgt_input_padding_mask,
                       memory_key_padding_mask=src_padding_mask)
        
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Svuota la cache CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Forzare la garbage collection di Python (per liberare anche RAM CPU)
        gc.collect()

        total_loss += loss.item()
        processed_batches += 1
        progress_bar.set_postfix({'train_loss': f'{total_loss / processed_batches:.4f}'})

    if processed_batches == 0: return 0.0
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
            src_padding_mask, tgt_padding_mask = src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE)

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
    logging.info("--- Preparazione Tokenizer e Vocabolari ---")
    train_jsonl_path = SPLITS_DIR / "train.jsonl"
    all_train_metadata = []
    midi_files_for_vocab_build = []

    logging.info(f"Inizio lettura {train_jsonl_path} per costruire vocabolari...")
    try:
        if not train_jsonl_path.is_file(): raise FileNotFoundError(f"JSONL non trovato: {train_jsonl_path}")
        with open(train_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    metadata_dict = entry.get('metadata', {})
                    relative_path_str = metadata_dict.get('midi_relative_path')
                    if relative_path_str:
                        relative_path_str = relative_path_str.replace("\\", "/") # Assicura che sia in formato Unix
                        midi_full_path_str = str(MIDI_BASE_DIR / relative_path_str)
                        if Path(midi_full_path_str).is_file():
                            midi_files_for_vocab_build.append(midi_full_path_str)
                            all_train_metadata.append(metadata_dict)
                except json.JSONDecodeError: pass
                except Exception: pass
    except Exception as e:
        logging.error(f"ERRORE CRITICO lettura {train_jsonl_path}: {e}", exc_info=True)
        sys.exit(1)

    if not midi_files_for_vocab_build:
        logging.error("ERRORE CRITICO: Nessun file MIDI per tokenizer.")
        sys.exit(1)

    force_rebuild_vocabs = False # Reimposta a False dopo il primo run di successo!
    midi_tokenizer = build_or_load_tokenizer(midi_file_paths=midi_files_for_vocab_build, force_build=force_rebuild_vocabs)
    metadata_vocab_map, _ = build_or_load_metadata_vocab(all_train_metadata, force_build=force_rebuild_vocabs)

    MIDI_VOCAB_SIZE = len(midi_tokenizer)
    META_VOCAB_SIZE = len(metadata_vocab_map)
    try:
        MIDI_PAD_ID = midi_tokenizer[MIDI_PAD_TOKEN_NAME]
        META_PAD_ID = metadata_vocab_map[META_PAD_TOKEN_NAME]
    except KeyError as e:
        logging.error(f"ERRORE CRITICO: Token PAD non trovato: {e}")
        sys.exit(1)

    logging.info(f"MIDI Vocab Size: {MIDI_VOCAB_SIZE}, MIDI PAD ID: {MIDI_PAD_ID}")
    logging.info(f"Meta Vocab Size: {META_VOCAB_SIZE}, Meta PAD ID: {META_PAD_ID}")

    logging.info("--- Creazione Dataset e DataLoader ---")
    try:
        train_dataset = MutopiaDataset(SPLITS_DIR / "train.jsonl", MIDI_BASE_DIR, midi_tokenizer, metadata_vocab_map, MAX_SEQ_LEN_MIDI, MAX_SEQ_LEN_META)
        val_dataset = MutopiaDataset(SPLITS_DIR / "validation.jsonl", MIDI_BASE_DIR, midi_tokenizer, metadata_vocab_map, MAX_SEQ_LEN_MIDI, MAX_SEQ_LEN_META)
    except Exception as e:
        logging.error(f"Errore creazione Dataset: {e}", exc_info=True)
        sys.exit(1)

    if len(train_dataset) == 0:
         logging.error("Dataset di training vuoto.")
         sys.exit(1)

    collate_fn_with_padding_ids = partial(pad_collate_fn, meta_pad_id=META_PAD_ID, midi_pad_id=MIDI_PAD_ID)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_with_padding_ids, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_with_padding_ids, num_workers=0)

    logging.info("--- Inizializzazione Modello ---")
    # Calcola max_pe_len da usare
    max_pe_len_calculated = max(MAX_SEQ_LEN_MIDI, MAX_SEQ_LEN_META) + 100 # +100 come buffer (o un valore che sai essere sufficiente)

    model = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD,
        META_VOCAB_SIZE, MIDI_VOCAB_SIZE, 
        max_pe_len=max_pe_len_calculated, # Passa il max_pe_len calcolato
        dim_feedforward=FFN_HID_DIM, dropout=DROPOUT
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=MIDI_PAD_ID)
    logging.info(f"Numero parametri: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logging.info(f"Device: {DEVICE}")
    
    model_params_to_save = {
    'num_encoder_layers': NUM_ENCODER_LAYERS,
    'num_decoder_layers': NUM_DECODER_LAYERS,
    'emb_size': EMB_SIZE,
    'nhead': NHEAD,
    'src_vocab_size': META_VOCAB_SIZE,
    'tgt_vocab_size': MIDI_VOCAB_SIZE,
    'dim_feedforward': FFN_HID_DIM,
    'dropout': DROPOUT,
    'max_pe_len': max_pe_len_calculated
    }
    
    vocab_info_to_save = {
        'midi_vocab_path': str(VOCAB_PATH),
        'metadata_vocab_path': str(METADATA_VOCAB_PATH),
        'midi_pad_id': MIDI_PAD_ID,
        'meta_pad_id': META_PAD_ID,
        'MAX_SEQ_LEN_MIDI': MAX_SEQ_LEN_MIDI,
        'MAX_SEQ_LEN_META': MAX_SEQ_LEN_META
    }

    logging.info("--- Inizio Addestramento ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5
    last_saved_epoch = 0 # Per tenere traccia dell'ultima epoca in cui è stato salvato un checkpoint periodico

    for epoch in range(1, EPOCHS + 1):
        logging.info(f"--- Epoch {epoch}/{EPOCHS} ---")
        start_time_epoch = time.time()
        train_loss_epoch = train_epoch(model, optimizer, criterion, train_dataloader)
        val_loss_epoch = evaluate(model, criterion, val_dataloader) # val_loss_epoch è la loss di validazione dell'epoca corrente
        epoch_duration_total = time.time() - start_time_epoch
        logging.info(f"Epoch {epoch}: Train Loss = {train_loss_epoch:.4f}, Val Loss = {val_loss_epoch:.4f} (Durata: {epoch_duration_total:.2f}s)")

        # --- Logica di salvataggio checkpoint ---
        # Salva un checkpoint ogni 10 epoche
        if epoch % 10 == 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            periodic_model_filename = f"transformer_mutopia_epoch{epoch}_valloss{val_loss_epoch:.4f}_{timestamp}.pt"
            periodic_model_path = MODEL_SAVE_DIR / periodic_model_filename
            logging.info(f"Salvataggio checkpoint periodico (epoch {epoch}): {periodic_model_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'current_val_loss': val_loss_epoch, # Validation loss di questa epoca
                'best_val_loss': best_val_loss,     # Migliore validation loss vista finora (per early stopping)
                'model_params': model_params_to_save,
                'vocab_info': vocab_info_to_save
            }, periodic_model_path)
            last_saved_epoch = epoch

        # --- Logica di Early stopping ---
        if val_loss_epoch < best_val_loss:
            logging.info(f"Validation loss migliorata da {best_val_loss:.4f} a {val_loss_epoch:.4f}.")
            best_val_loss = val_loss_epoch
            epochs_no_improve = 0
            # Non salviamo più il "best_model" qui ad ogni miglioramento,
            # ma continuiamo a tracciare best_val_loss per l'early stopping.
        else:
            epochs_no_improve += 1
            logging.info(f"Validation loss non migliorata ({val_loss_epoch:.4f} vs best {best_val_loss:.4f}). Epoche senza miglioramento: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            logging.info(f"Nessun miglioramento per {patience} epoche consecutive. Early stopping.")
            break # Esce dal ciclo di addestramento

    # --- Salvataggio del modello finale ---
    # Salva lo stato finale del modello dopo il completamento di tutte le epoche o l'early stopping,
    # specialmente se l'ultima epoca non era un multiplo di 10.
    if epoch != last_saved_epoch : # Controlla se l'ultimo stato è già stato salvato periodicamente
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        final_model_filename = f"transformer_mutopia_final_epoch{epoch}_valloss{val_loss_epoch:.4f}_{timestamp}.pt"
        final_model_path = MODEL_SAVE_DIR / final_model_filename
        logging.info(f"Salvataggio checkpoint finale (dopo epoch {epoch}): {final_model_path}")
        torch.save({
            'epoch': epoch, # Ultima epoca completata
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'current_val_loss': val_loss_epoch, # Validation loss dell'ultima epoca
            'best_val_loss': best_val_loss,     # Migliore validation loss generale
            'model_params': model_params_to_save,
            'vocab_info': vocab_info_to_save
        }, final_model_path)
    else:
        logging.info(f"Lo stato finale del modello (epoch {epoch}) è già stato salvato come checkpoint periodico.")


    logging.info("--- Addestramento Terminato ---")
    logging.info("--- Script Terminato ---")