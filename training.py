import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F # Importa F per softmax
import miditok # Assicurati di installare miditok
# from miditoolkit import MidiFile # Necessario per salvare midi generato con miditok < 2.0
from pathlib import Path
import json
import math
import random
import logging
from tqdm import tqdm # Per le barre di progresso
import os
import time # Per timestamp salvataggio

# --- Configurazione / Costanti ---
# NOTA: Questi sono solo esempi, da adattare!
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("./mutopia_data") # Directory base dei dati scaricati
SPLITS_DIR = DATA_DIR / "dataset_splits" # Directory con train/validation/test.jsonl
MIDI_BASE_DIR = DATA_DIR # Directory radice dove cercare i midi_relative_path
MODEL_SAVE_DIR = DATA_DIR / "model_checkpoints" # Directory per salvare i modelli
GENERATION_OUTPUT_DIR = DATA_DIR / "generated_midi" # Directory per output generati

# Crea directory se non esistono
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
GENERATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configurazioni Tokenizer MIDI (scegliere una strategia)
# MIDI_TOKENIZER_STRATEGY = miditok.REMI # Esempio
# MIDI_TOKENIZER_STRATEGY = miditok.TSD # Altro esempio
MIDI_TOKENIZER_STRATEGY = miditok.CPWord # Esempio scelto

VOCAB_PATH = DATA_DIR / "midi_vocab.json" # Dove salvare/caricare il vocabolario MIDI
METADATA_VOCAB_PATH = DATA_DIR / "metadata_vocab.json" # Vocabolario per i token metadati

# Token Speciali (assicurati che non collidano con quelli di miditok)
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>" # Start of Sequence (MIDI Target)
EOS_TOKEN = "<eos>" # End of Sequence (MIDI Target)
UNK_TOKEN = "<unk>" # Unknown
SOS_META_TOKEN = "<sos_meta>" # Start of Metadata Sequence (Source)
EOS_META_TOKEN = "<eos_meta>" # End of Metadata Sequence (Source)
# Assicurati che gli ID siano gestiti correttamente (di solito 0 per PAD)

# Iperparametri del Modello e Addestramento (Esempi!)
EPOCHS = 20
BATCH_SIZE = 16 # Riduci se hai poca memoria GPU
LEARNING_RATE = 0.0001
EMB_SIZE = 256 # Dimensione embedding
NHEAD = 4 # Numero di head nell'attention (deve dividere EMB_SIZE)
FFN_HID_DIM = 512 # Dimensione layer nascosto FeedForward
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT = 0.1
MAX_SEQ_LEN_MIDI = 1024 # Lunghezza massima sequenza MIDI (tronca/scarta se più lunga)
MAX_SEQ_LEN_META = 128 # Aumentata per includere potenziale titolo lungo

# Setup Logging (opzionale ma utile)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Tokenizer e Vocabolario ---

def build_or_load_tokenizer(midi_files_for_vocab, force_build=False):
    """Costruisce o carica il tokenizer MIDI e il suo vocabolario."""
    special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SOS_META_TOKEN, EOS_META_TOKEN] # Includi tutti i token speciali

    if VOCAB_PATH.exists() and not force_build:
        logging.info(f"Caricamento vocabolario MIDI da {VOCAB_PATH}")
        # miditok >= 2.0 salva parametri incluso vocabolario con add_special_tokens
        # Se usi miditok < 2.0, potresti dover aggiungere token speciali manualmente
        try:
            # Miditok >= 2.1.0
            tokenizer = MIDI_TOKENIZER_STRATEGY(params=VOCAB_PATH)
             # Verifica se i token speciali sono presenti, aggiungili se necessario
            existing_tokens = set(tokenizer.vocab)
            missing_tokens = [t for t in special_tokens if t not in existing_tokens]
            if missing_tokens:
                 logging.warning(f"Token speciali mancanti nel vocabolario caricato: {missing_tokens}. Aggiungo...")
                 tokenizer.add_tokens(missing_tokens)

        except TypeError: # Potrebbe essere un vecchio formato o versione < 2.1.0
             try:
                # Miditok 2.0.x
                tokenizer = MIDI_TOKENIZER_STRATEGY(vocab_path=VOCAB_PATH)
                existing_tokens = set(tokenizer.vocab)
                missing_tokens = [t for t in special_tokens if t not in existing_tokens]
                if missing_tokens:
                    logging.warning(f"Token speciali mancanti nel vocabolario caricato: {missing_tokens}. Aggiungo...")
                    tokenizer.add_tokens(missing_tokens)
             except Exception as e:
                 logging.error(f"Errore nel caricare tokenizer da {VOCAB_PATH}, provo a ricostruire. Errore: {e}")
                 return build_or_load_tokenizer(midi_files_for_vocab, force_build=True) # Forza ricostruzione


    else:
        logging.info("Creazione nuovo vocabolario MIDI...")
        # Aggiungi parametri specifici se necessario, es: pitch_range, beat_res, etc.
        tokenizer_params = miditok.TokenizerConfig(
            special_tokens=special_tokens, # Passa token speciali alla config (miditok >= 2.0)
            use_programs=False # Esempio: non usare program changes se non servono
            # Aggiungi altri parametri di configurazione qui
        )
        tokenizer = MIDI_TOKENIZER_STRATEGY(tokenizer_config=tokenizer_params)


        logging.info(f"Apprendimento vocabolario da {len(midi_files_for_vocab)} file MIDI...")
        if not midi_files_for_vocab:
            raise ValueError("Nessun file MIDI fornito per costruire il vocabolario.")

        # Nota: learn_vocabulary potrebbe richiedere molti file/tempo
        tokenizer.learn_vocabulary(midi_files_for_vocab)

        logging.info(f"Salvataggio vocabolario MIDI in {VOCAB_PATH}")
        # Assicurati che la directory esista
        VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save_params(VOCAB_PATH) # Metodo standard per miditok >= 2.0

    logging.info(f"Dimensione vocabolario MIDI (incl. speciali): {len(tokenizer)}")
    # Stampa ID token speciali per verifica
    try:
         logging.info(f"ID Token Speciali MIDI - PAD: {tokenizer[PAD_TOKEN]}, SOS: {tokenizer[SOS_TOKEN]}, EOS: {tokenizer[EOS_TOKEN]}, UNK: {tokenizer.unk_token_id}") # UNK gestito internamente
    except KeyError as e:
         logging.error(f"Errore: Uno dei token speciali non trovato nel tokenizer MIDI dopo caricamento/build: {e}")
         # Potrebbe essere necessario gestire questo caso critico
    except AttributeError:
         logging.warning("Attributo ID non standard per token speciali (vecchia versione miditok?).")


    return tokenizer

def tokenize_metadata(metadata_dict):
    """
    Converte il dizionario di metadati in una lista di token stringa.
    Include Style, Key, TimeSig e Title (troncato).
    """
    tokens = []
    # Metti solo alcuni metadati chiave come esempio
    if 'style' in metadata_dict and metadata_dict['style']:
        tokens.append(f"Style={metadata_dict['style'].replace(' ', '_')}")
    if 'key' in metadata_dict and metadata_dict['key']:
        tokens.append(f"Key={metadata_dict['key'].replace(' ', '_')}")
    if 'time_signature' in metadata_dict and metadata_dict['time_signature']:
        tokens.append(f"TimeSig={metadata_dict['time_signature']}")
    # Aggiungi token per il titolo
    if 'title' in metadata_dict and metadata_dict['title']:
        # Pulisci e limita lunghezza
        clean_title = metadata_dict['title'].replace(' ', '_').lower()
        # Rimuovi caratteri non alfanumerici se necessario
        clean_title = ''.join(c for c in clean_title if c.isalnum() or c == '_')
        tokens.append(f"Title={clean_title[:30]}") # Limita lunghezza

    # Aggiungi altri metadati se necessario...
    # Esempio: Strumento (se disponibile e utile)
    # if 'mutopiainstrument' in metadata_dict and metadata_dict['mutopiainstrument']:
    #    tokens.append(f"Instrument={metadata_dict['mutopiainstrument'].replace(' ', '_')}")

    return tokens

def build_or_load_metadata_vocab(all_metadata_examples, force_build=False):
    """Costruisce o carica un vocabolario per i token metadati."""
    if METADATA_VOCAB_PATH.exists() and not force_build:
        logging.info(f"Caricamento vocabolario Metadati da {METADATA_VOCAB_PATH}")
        with open(METADATA_VOCAB_PATH, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        token_to_id = vocab_data['token_to_id']
        # Assicurati che i token speciali siano presenti
        required_specials = [PAD_TOKEN, UNK_TOKEN, SOS_META_TOKEN, EOS_META_TOKEN]
        missing = [t for t in required_specials if t not in token_to_id]
        if missing:
             logging.warning(f"Token speciali metadati mancanti nel file caricato: {missing}. Ricostruisco.")
             return build_or_load_metadata_vocab(all_metadata_examples, force_build=True)
        id_to_token = {i: t for t, i in token_to_id.items()} # Ricrea id_to_token
        return token_to_id, id_to_token
    else:
        logging.info("Creazione nuovo vocabolario Metadati...")
        metadata_tokens = set()
        for meta_dict in all_metadata_examples:
            tokens = tokenize_metadata(meta_dict) # Usa la funzione aggiornata
            metadata_tokens.update(tokens)

        # Costruisci vocabolario con token speciali per metadati
        all_tokens = [PAD_TOKEN, UNK_TOKEN, SOS_META_TOKEN, EOS_META_TOKEN] + sorted(list(metadata_tokens))
        token_to_id = {token: i for i, token in enumerate(all_tokens)}
        id_to_token = {i: token for token, i in token_to_id.items()}

        # Verifica ID token speciali (PAD dovrebbe essere 0 se CrossEntropyLoss usa ignore_index=0)
        if token_to_id[PAD_TOKEN] != 0:
            logging.warning(f"ID del PAD_TOKEN ({PAD_TOKEN}) non è 0 ({token_to_id[PAD_TOKEN]}). "
                            "Potrebbe causare problemi con CrossEntropyLoss(ignore_index=0). Riassegno gli ID.")
            # Riassegna mettendo PAD a 0
            pad_tok = PAD_TOKEN
            other_specials = [t for t in [UNK_TOKEN, SOS_META_TOKEN, EOS_META_TOKEN] if t in token_to_id]
            actual_tokens = sorted(list(metadata_tokens))
            # Ordine: PAD, altri speciali, token reali
            all_tokens_reordered = [pad_tok] + other_specials + actual_tokens
            token_to_id = {token: i for i, token in enumerate(all_tokens_reordered)}
            # Aggiungi UNK se non presente nei dati originali ma definito
            if UNK_TOKEN not in token_to_id:
                 unk_id = len(token_to_id)
                 token_to_id[UNK_TOKEN] = unk_id
                 logging.info(f"Aggiunto UNK_TOKEN con ID {unk_id}")

            id_to_token = {i: token for token, i in token_to_id.items()}
            logging.info(f"Nuovo ID PAD_TOKEN: {token_to_id[PAD_TOKEN]}")


        vocab_data = {'token_to_id': token_to_id, 'id_to_token': id_to_token}
        logging.info(f"Salvataggio vocabolario Metadati in {METADATA_VOCAB_PATH}")
        METADATA_VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(METADATA_VOCAB_PATH, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        logging.info(f"Dimensione vocabolario Metadati (incl. speciali): {len(token_to_id)}")
        return token_to_id, id_to_token


# --- Dataset e DataLoader ---

class MutopiaDataset(Dataset):
    def __init__(self, jsonl_path, midi_base_dir, midi_tokenizer, metadata_vocab_map,
                 max_len_midi, max_len_meta, filter_key=None): # Aggiunto filter_key
        self.midi_base_dir = Path(midi_base_dir)
        self.midi_tokenizer = midi_tokenizer
        self.metadata_vocab_map = metadata_vocab_map # token_to_id map
        self.max_len_midi = max_len_midi
        self.max_len_meta = max_len_meta
        self.filter_key = filter_key # Es: "A minor"

        # ID Token Speciali dal vocabolario metadati (assumendo condivisione)
        # Se i vocabolari sono separati, prendere ID da midi_tokenizer per SOS/EOS MIDI
        self.pad_id = metadata_vocab_map[PAD_TOKEN]
        self.sos_id = midi_tokenizer[SOS_TOKEN] # ID SOS MIDI dal tokenizer MIDI
        self.eos_id = midi_tokenizer[EOS_TOKEN] # ID EOS MIDI dal tokenizer MIDI
        self.sos_meta_id = metadata_vocab_map[SOS_META_TOKEN]
        self.eos_meta_id = metadata_vocab_map[EOS_META_TOKEN]
        self.unk_meta_id = metadata_vocab_map[UNK_TOKEN]

        logging.info(f"Caricamento dati da {jsonl_path}...")
        self.data = []
        skipped_missing_midi = 0
        skipped_key_filter = 0
        skipped_other = 0
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        entry = json.loads(line)
                        # 1. Filtra per chiave (se specificato)
                        if self.filter_key and entry.get('metadata', {}).get('key') != self.filter_key:
                            skipped_key_filter += 1
                            continue

                        # 2. Verifica che il file MIDI esista prima di aggiungerlo
                        midi_path_check = self.midi_base_dir / entry.get('midi_relative_path', '')
                        if 'midi_relative_path' in entry and midi_path_check.exists():
                            self.data.append(entry)
                        else:
                            # logging.warning(f"File MIDI non trovato o path mancante, salto: {entry.get('midi_relative_path', 'N/A')}")
                            skipped_missing_midi += 1
                    except json.JSONDecodeError:
                        logging.warning(f"Riga JSON malformata in {jsonl_path} (linea ~{line_num+1}), salto.")
                        skipped_other += 1
                    except Exception as e:
                         logging.warning(f"Errore caricando riga {line_num+1}: {e}")
                         skipped_other += 1

        except FileNotFoundError:
             logging.error(f"File dataset non trovato: {jsonl_path}")
             raise

        logging.info(f"Caricati {len(self.data)} campioni da {jsonl_path}.")
        if self.filter_key:
            logging.info(f"   Filtrati per chiave '{self.filter_key}'.")
        logging.info(f"   Saltati per MIDI mancante: {skipped_missing_midi}")
        logging.info(f"   Saltati per filtro chiave: {skipped_key_filter}")
        logging.info(f"   Saltati per altri errori: {skipped_other}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retry logic in case of transient errors like MIDI tokenization failure
        max_retries = 2
        for attempt in range(max_retries):
            try:
                entry = self.data[idx]
                metadata = entry['metadata']
                midi_relative_path = entry['midi_relative_path']
                midi_full_path = self.midi_base_dir / midi_relative_path

                # 1. Tokenizza Metadati (Source Sequence)
                meta_tokens_str = tokenize_metadata(metadata) # Usa la funzione aggiornata
                meta_token_ids = [self.metadata_vocab_map.get(token, self.unk_meta_id) for token in meta_tokens_str]
                src_seq = [self.sos_meta_id] + meta_token_ids[:self.max_len_meta-2] + [self.eos_meta_id]

                # 2. Tokenizza MIDI (Target Sequence)
                # Usa miditok per ottenere la lista di ID interi direttamente
                # Gestione errore più robusta: ritorna None se fallisce dopo retries
                midi_token_ids = self.midi_tokenizer(midi_full_path)

                # Miditok può ritornare una lista di liste (per traccia) o una lista singola
                if isinstance(midi_token_ids, list) and len(midi_token_ids) > 0 and isinstance(midi_token_ids[0], list):
                     # Caso multi-traccia: prendi la prima traccia o unisci? Per ora prendi la prima.
                     # Potresti voler implementare una logica di unione se necessario.
                     if not midi_token_ids[0]: # Traccia vuota?
                         logging.warning(f"Prima traccia vuota per MIDI {midi_full_path}. Salto campione.")
                         raise RuntimeError("Traccia MIDI vuota")
                     midi_token_ids = midi_token_ids[0]
                elif isinstance(midi_token_ids, list) and len(midi_token_ids) > 0 and isinstance(midi_token_ids[0], int):
                     # Caso mono-traccia o output già appiattito
                     pass
                else:
                     logging.warning(f"Formato output tokenizer MIDI non atteso per {midi_full_path}: {type(midi_token_ids)}. Salto campione.")
                     raise RuntimeError("Output tokenizer MIDI non valido")


                # Aggiungi SOS/EOS MIDI e tronca
                tgt_seq = [self.sos_id] + midi_token_ids[:self.max_len_midi-2] + [self.eos_id]

                return torch.tensor(src_seq, dtype=torch.long), torch.tensor(tgt_seq, dtype=torch.long)

            except Exception as e:
                logging.warning(f"Tentativo {attempt+1}/{max_retries} fallito per idx {idx} ({midi_full_path}): {e}")
                if attempt + 1 == max_retries:
                    logging.error(f"Errore definitivo processando idx {idx} ({midi_full_path}): {e}. Ritorno None.")
                    return None # Ritorna None per indicare fallimento

        return None # Non dovrebbe arrivare qui, ma per sicurezza

def pad_collate_fn(batch):
    """
    Collate function per DataLoader. Esegue il padding, crea le maschere
    e filtra eventuali elementi None restituiti dal Dataset.
    """
    # Filtra campioni che hanno ritornato None
    batch = [item for item in batch if item is not None]
    if not batch: # Se il batch diventa vuoto dopo il filtraggio
        return None, None, None, None # Ritorna None per segnalare batch vuoto

    # Separa sorgenti e target
    src_batch, tgt_batch = zip(*batch)

    # Trova l'ID del PAD token (assumendo sia lo stesso per entrambi i vocabolari)
    # Meglio prenderlo da una fonte certa, ma 0 è comune con ignore_index
    pad_id = 0 # Assumiamo che PAD_TOKEN sia mappato a 0 in entrambi i vocabolari

    # Padding delle sequenze sorgente (metadati)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    # Padding delle sequenze target (MIDI)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)

    # Creazione maschere di padding (True dove c'è padding)
    # Nota: nn.Transformer si aspetta maschere dove True indica posizioni da ignorare
    src_padding_mask = (src_padded == pad_id)
    tgt_padding_mask = (tgt_padded == pad_id)

    return src_padded, tgt_padded, src_padding_mask, tgt_padding_mask


# --- Modello Transformer (invariato) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model) # Rimosso dim intermedia
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Args: x: Tensor, shape [seq_len, batch_size, embedding_dim]"""
        # print("Input shape to PE:", x.shape)
        # print("PE shape:", self.pe.shape)
        # print("PE slice shape:", self.pe[:x.size(0), :].shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead,
                 src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=max(MAX_SEQ_LEN_MIDI, MAX_SEQ_LEN_META)+10) # Usa max_len adeguato

        # Transformer Layer
        self.transformer = nn.Transformer(
            d_model=emb_size, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True # Usa batch_first=True per coerenza
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size) # Proietta output decoder a dimensione vocabolario target

        # Inizializzazione pesi (opzionale ma spesso utile)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, memory_key_padding_mask=None, tgt_mask=None):
        """ Forward pass. Input/Output shape (batch, seq_len) """
        # Embedding + Positional Encoding
        # Transpose per nn.Transformer che vuole (seq_len, batch, emb) se batch_first=False
        # Se batch_first=True, nn.Transformer accetta (batch, seq_len, emb)
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(EMB_SIZE)) # Scalatura embedding
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(EMB_SIZE)) # Scalatura embedding

        # Genera maschera causale per il decoder se non fornita
        if tgt_mask is None:
             tgt_len = tgt.size(1) # seq_len è la seconda dimensione con batch_first=True
             tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt.device)

        # Passaggio attraverso il Transformer
        # Maschere: src_padding_mask (batch, src_len), tgt_padding_mask (batch, tgt_len)
        #           tgt_mask (tgt_len, tgt_len)
        #           memory_key_padding_mask (batch, src_len) - passata al decoder per ignorare pad nell'output encoder
        # Nota: nn.Transformer(batch_first=True) si aspetta maschere nelle stesse shape
        if memory_key_padding_mask is None:
             memory_key_padding_mask = src_padding_mask # Usa la maschera src per la memoria encoder

        outs = self.transformer(src_emb, tgt_emb,
                                tgt_mask=tgt_mask, # Maschera causale per decoder
                                src_key_padding_mask=src_padding_mask, # Maschera padding per encoder self-attention
                                tgt_key_padding_mask=tgt_padding_mask, # Maschera padding per decoder self-attention
                                memory_key_padding_mask=memory_key_padding_mask) # Maschera padding per encoder-decoder attention

        # Proietta l'output sulla dimensione del vocabolario target
        return self.generator(outs) # Output shape: (batch, seq_len, tgt_vocab_size)

    def encode(self, src, src_mask):
        """Funzione per usare solo l'encoder (utile per inferenza). Input (batch, seq_len)"""
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(EMB_SIZE))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)

    def decode(self, tgt, memory, tgt_mask, tgt_padding_mask=None, memory_key_padding_mask=None):
        """Funzione per usare solo il decoder (utile per inferenza). Input tgt (batch, seq_len)"""
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(EMB_SIZE))
        return self.transformer.decoder(tgt_emb, memory,
                                         tgt_mask=tgt_mask,
                                         tgt_key_padding_mask=tgt_padding_mask,
                                         memory_key_padding_mask=memory_key_padding_mask)


# --- Ciclo di Addestramento e Valutazione ---

def train_epoch(model, optimizer, criterion, train_dataloader, pad_id):
    model.train() # Imposta modalità training
    total_loss = 0
    processed_batches = 0

    progress_bar = tqdm(train_dataloader, desc="Training Epoch")
    for batch_data in progress_bar:
        # Gestisci batch potenzialmente vuoti da collate_fn
        if batch_data[0] is None:
            logging.warning("Batch vuoto ricevuto dal dataloader, salto.")
            continue

        src, tgt, src_padding_mask, tgt_padding_mask = batch_data
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src_padding_mask = src_padding_mask.to(DEVICE)
        tgt_padding_mask = tgt_padding_mask.to(DEVICE)

        # Prepara input e output per il modello
        # Input per il decoder è la sequenza target senza l'ultimo token (<EOS>)
        tgt_input = tgt[:, :-1]
        tgt_input_padding_mask = tgt_padding_mask[:, :-1]

        # Target reale per la loss è la sequenza target senza il primo token (<SOS>)
        tgt_out = tgt[:, 1:]
        # La maschera per la loss non è direttamente necessaria se criterion ha ignore_index

        optimizer.zero_grad()

        # Forward pass
        # La memory_key_padding_mask per il decoder deve avere shape (batch, src_len)
        logits = model(src=src,
                       tgt=tgt_input,
                       src_padding_mask=src_padding_mask,
                       tgt_padding_mask=tgt_input_padding_mask, # Maschera per decoder self-attention
                       memory_key_padding_mask=src_padding_mask) # Maschera per encoder-decoder attention

        # Calcola loss
        # CrossEntropyLoss si aspetta (N, C) e (N), dove N=batch*seq_len, C=vocab_size
        # Usiamo ignore_index=pad_id per ignorare i token di padding nel calcolo della loss
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradient
        optimizer.step()

        total_loss += loss.item()
        processed_batches += 1
        progress_bar.set_postfix({'train_loss': total_loss / processed_batches})

    if processed_batches == 0:
        return 0.0 # Evita divisione per zero se nessun batch è stato processato
    return total_loss / processed_batches


def evaluate(model, criterion, dataloader, pad_id):
    model.eval() # Imposta modalità valutazione
    total_loss = 0
    processed_batches = 0

    progress_bar = tqdm(dataloader, desc="Evaluation")
    with torch.no_grad(): # Disabilita calcolo gradienti
        for batch_data in progress_bar:
             # Gestisci batch potenzialmente vuoti da collate_fn
            if batch_data[0] is None:
                continue

            src, tgt, src_padding_mask, tgt_padding_mask = batch_data
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            src_padding_mask = src_padding_mask.to(DEVICE)
            tgt_padding_mask = tgt_padding_mask.to(DEVICE)

            tgt_input = tgt[:, :-1]
            tgt_input_padding_mask = tgt_padding_mask[:, :-1]
            tgt_out = tgt[:, 1:]

            logits = model(src=src,
                           tgt=tgt_input,
                           src_padding_mask=src_padding_mask,
                           tgt_padding_mask=tgt_input_padding_mask,
                           memory_key_padding_mask=src_padding_mask)

            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            total_loss += loss.item()
            processed_batches += 1
            progress_bar.set_postfix({'eval_loss': total_loss / processed_batches})

    if processed_batches == 0:
        return float('inf') # Ritorna infinito se nessun batch è stato processato
    return total_loss / processed_batches


# --- Funzione di Generazione ---

def generate_sequence(model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
                      max_len=500, temperature=0.7, top_k=None, device=DEVICE):
    """
    Genera una sequenza MIDI dato un prompt di metadati.
    Args:
        model: Il modello Seq2SeqTransformer addestrato.
        midi_tokenizer: Il tokenizer MIDI usato per l'addestramento.
        metadata_vocab_map: Mappa token->id per i metadati.
        metadata_prompt: Lista di stringhe token per i metadati (es. ['Style=Classical', 'Key=C_Major']).
        max_len: Lunghezza massima della sequenza MIDI da generare.
        temperature: Controlla la casualità (valori bassi -> più prevedibile, alti -> più casuale).
        top_k: Considera solo i k token più probabili (opzionale).
        device: Dispositivo su cui eseguire la generazione.
    Returns:
        Lista di ID token MIDI generati.
    """
    model.eval()

    # Token ID speciali necessari
    sos_meta_id = metadata_vocab_map[SOS_META_TOKEN]
    eos_meta_id = metadata_vocab_map[EOS_META_TOKEN]
    unk_meta_id = metadata_vocab_map[UNK_TOKEN]
    sos_midi_id = midi_tokenizer[SOS_TOKEN]
    eos_midi_id = midi_tokenizer[EOS_TOKEN]
    pad_id = metadata_vocab_map[PAD_TOKEN] # Assume PAD ID comune

    # 1. Prepara input metadati
    meta_token_ids = [metadata_vocab_map.get(token, unk_meta_id) for token in metadata_prompt]
    src_seq = torch.tensor([[sos_meta_id] + meta_token_ids + [eos_meta_id]], dtype=torch.long, device=device)
    src_padding_mask = (src_seq == pad_id) # Shape: (1, src_len)

    with torch.no_grad():
        # 2. Codifica i metadati
        memory = model.encode(src_seq, src_padding_mask) # Shape: (1, src_len, emb_size)
        # Maschera per la memoria usata dal decoder (True dove c'è PAD nell'input src)
        memory_key_padding_mask = src_padding_mask # Shape: (1, src_len)

        # 3. Inizia generazione decoder
        # Inizia con il token SOS MIDI. Shape: (1, 1)
        tgt_tokens = torch.tensor([[sos_midi_id]], dtype=torch.long, device=device)

        for _ in range(max_len - 1): # Genera fino a max_len token (incluso SOS iniziale)
            tgt_len = tgt_tokens.size(1)
            # Maschera causale per il decoder. Shape: (tgt_len, tgt_len)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)
            # Maschera padding per l'input del decoder (non dovrebbe esserci PAD qui durante la generazione step-by-step)
            tgt_padding_mask = torch.zeros_like(tgt_tokens, dtype=torch.bool, device=device) # Shape: (1, tgt_len)

            # Ottieni output del decoder. Shape: (1, tgt_len, emb_size)
            decoder_output = model.decode(tgt=tgt_tokens, memory=memory, tgt_mask=tgt_mask,
                                          tgt_padding_mask=tgt_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)

            # Proietta sull'output del vocabolario. Shape: (1, tgt_len, midi_vocab_size)
            logits = model.generator(decoder_output)

            # Considera solo l'ultimo token predetto. Shape: (1, midi_vocab_size)
            last_logits = logits[:, -1, :]

            # Applica temperature scaling
            if temperature > 0:
                last_logits = last_logits / temperature

            # Applica top-k filtering (opzionale)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)), dim=-1)
                # Imposta a -inf i logits non nel top-k
                last_logits[last_logits < v[:, [-1]]] = -float('Inf')

            # Ottieni probabilità e campiona il prossimo token
            probs = F.softmax(last_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1) # Shape: (1, 1)

            # Se viene generato EOS, termina
            if next_token_id.item() == eos_midi_id:
                break

            # Aggiungi il token generato alla sequenza target
            tgt_tokens = torch.cat((tgt_tokens, next_token_id), dim=1)

    # Ritorna la sequenza generata (escludendo SOS iniziale)
    return tgt_tokens[0, 1:].tolist()


# --- Esecuzione Principale ---

if __name__ == "__main__":

    # 1. Prepara Tokenizer e Vocabolari
    logging.info("--- Preparazione Tokenizer e Vocabolari ---")
    train_jsonl_path = SPLITS_DIR / "train.jsonl"
    all_train_metadata = []
    midi_files_for_vocab_build = []
    try:
         with open(train_jsonl_path, 'r', encoding='utf-8') as f:
              for line in f:
                   try:
                        entry = json.loads(line)
                        # Applica qui filtro per chiave se vuoi usarlo per costruire vocabolari
                        # if entry.get('metadata', {}).get('key') != "A minor": continue
                        midi_path = MIDI_BASE_DIR / entry['midi_relative_path']
                        if midi_path.exists():
                            midi_files_for_vocab_build.append(str(midi_path))
                            all_train_metadata.append(entry['metadata']) # Usa metadati per vocabolario meta
                   except Exception as e:
                       logging.debug(f"Ignoro riga per costruzione vocabolario: {e}")
         logging.info(f"Trovati {len(midi_files_for_vocab_build)} file MIDI nel training set per il vocabolario MIDI.")
         logging.info(f"Trovati {len(all_train_metadata)} record metadati per il vocabolario Meta.")
    except FileNotFoundError:
         logging.error(f"File di training non trovato: {train_jsonl_path}. Impossibile costruire vocabolari.")
         exit()

    # Costruisci/Carica tokenizer e vocabolari
    # Forza ricostruzione se hai cambiato tokenize_metadata o token speciali
    force_rebuild_vocabs = False # Imposta a True se necessario
    midi_tokenizer = build_or_load_tokenizer(midi_files_for_vocab_build, force_build=force_rebuild_vocabs)
    metadata_vocab_map, metadata_id_to_token = build_or_load_metadata_vocab(all_train_metadata, force_build=force_rebuild_vocabs)

    # Ottieni dimensioni vocabolario e ID padding
    MIDI_VOCAB_SIZE = len(midi_tokenizer)
    META_VOCAB_SIZE = len(metadata_vocab_map)
    # Assumi che PAD_ID sia 0 in entrambi, verifica!
    PAD_ID = metadata_vocab_map[PAD_TOKEN]
    if midi_tokenizer[PAD_TOKEN] != PAD_ID:
         logging.warning(f"PAD ID Mismatch! Meta: {PAD_ID}, MIDI: {midi_tokenizer[PAD_TOKEN]}. Usa ID Meta ({PAD_ID}).")
    if PAD_ID != 0:
         logging.warning(f"PAD ID non è 0 ({PAD_ID}). Assicurati che CrossEntropyLoss(ignore_index={PAD_ID}) sia corretto.")

    logging.info(f"Vocabolario MIDI: {MIDI_VOCAB_SIZE}, Vocabolario Meta: {META_VOCAB_SIZE}, PAD_ID: {PAD_ID}")


    # 2. Crea Dataset e DataLoader
    logging.info("--- Creazione Dataset e DataLoader ---")
    # Scegli se filtrare per chiave
    KEY_FILTER = None # Imposta a "A minor" o altro se vuoi filtrare
    # KEY_FILTER = "A minor"

    try:
        train_dataset = MutopiaDataset(SPLITS_DIR / "train.jsonl", MIDI_BASE_DIR, midi_tokenizer, metadata_vocab_map,
                                     MAX_SEQ_LEN_MIDI, MAX_SEQ_LEN_META, filter_key=KEY_FILTER)
        val_dataset = MutopiaDataset(SPLITS_DIR / "validation.jsonl", MIDI_BASE_DIR, midi_tokenizer, metadata_vocab_map,
                                   MAX_SEQ_LEN_MIDI, MAX_SEQ_LEN_META, filter_key=KEY_FILTER)
        test_dataset = MutopiaDataset(SPLITS_DIR / "test.jsonl", MIDI_BASE_DIR, midi_tokenizer, metadata_vocab_map,
                                    MAX_SEQ_LEN_MIDI, MAX_SEQ_LEN_META, filter_key=KEY_FILTER)
    except FileNotFoundError as e:
        logging.error(f"File .jsonl non trovato: {e}. Interruzione.")
        exit()
    except Exception as e:
        logging.error(f"Errore durante la creazione dei Dataset: {e}", exc_info=True)
        exit()

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
         logging.error("Uno o più dataset (train/val/test) sono vuoti. Controllare filtri, percorsi o file .jsonl.")
         exit()

    # Nota: num_workers > 0 può causare problemi su Windows o con alcuni debugger. Inizia con 0.
    # pin_memory=True può velocizzare trasferimenti a GPU se usi GPU.
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn, num_workers=0, pin_memory=True if DEVICE == 'cuda' else False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn, num_workers=0, pin_memory=True if DEVICE == 'cuda' else False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn, num_workers=0, pin_memory=True if DEVICE == 'cuda' else False)


    # 3. Inizializza Modello, Ottimizzatore, Loss
    logging.info("--- Inizializzazione Modello ---")
    model = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD,
        META_VOCAB_SIZE, MIDI_VOCAB_SIZE, FFN_HID_DIM, DROPOUT
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Assicurati che ignore_index sia l'ID corretto per il tuo PAD_TOKEN
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # Conta parametri (opzionale)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Numero totale parametri addestrabili: {total_params:,}")
    logging.info(f"Modello eseguito su: {DEVICE}")

    # 4. Ciclo di Addestramento
    logging.info("--- Inizio Addestramento ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5 # Numero di epoche senza miglioramenti prima di fermarsi (early stopping)
    best_model_path = None

    for epoch in range(1, EPOCHS + 1):
        logging.info(f"--- Epoch {epoch}/{EPOCHS} ---")

        start_time = time.time()
        train_loss = train_epoch(model, optimizer, criterion, train_dataloader, PAD_ID)
        epoch_duration = time.time() - start_time

        start_time_eval = time.time()
        val_loss = evaluate(model, criterion, val_dataloader, PAD_ID)
        eval_duration = time.time() - start_time_eval

        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f} (took {epoch_duration:.2f}s), Val Loss = {val_loss:.4f} (took {eval_duration:.2f}s)")

        # Salva il modello migliore e implementa early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Salva il checkpoint del modello migliore (TODO implementato)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_save_path = MODEL_SAVE_DIR / f"transformer_mutopia_best_epoch{epoch}_{timestamp}.pt"
            logging.info(f"Miglioramento validation loss. Salvataggio modello in {model_save_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'midi_vocab_size': MIDI_VOCAB_SIZE,
                'meta_vocab_size': META_VOCAB_SIZE,
                'emb_size': EMB_SIZE,
                'nhead': NHEAD,
                'num_encoder_layers': NUM_ENCODER_LAYERS,
                'num_decoder_layers': NUM_DECODER_LAYERS,
                'ffn_hid_dim': FFN_HID_DIM,
                # Salva anche i path dei vocabolari per riferimento
                'midi_vocab_path': str(VOCAB_PATH),
                'metadata_vocab_path': str(METADATA_VOCAB_PATH),
            }, model_save_path)
            best_model_path = model_save_path # Tieni traccia del percorso migliore
        else:
            epochs_no_improve += 1
            logging.info(f"Validation loss non migliorata ({val_loss:.4f} vs best {best_val_loss:.4f}). Epoche senza miglioramento: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            logging.info(f"Nessun miglioramento per {patience} epoche consecutive. Early stopping.")
            break

    logging.info("--- Addestramento Terminato ---")

    # 5. Valutazione finale sul Test Set (TODO implementato)
    if best_model_path and best_model_path.exists():
        logging.info(f"--- Valutazione su Test Set usando il modello migliore: {best_model_path} ---")
        # Carica il modello migliore
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        # Ricrea il modello con i parametri salvati (importante se diversi dagli attuali)
        loaded_model = Seq2SeqTransformer(
            num_encoder_layers=checkpoint.get('num_encoder_layers', NUM_ENCODER_LAYERS),
            num_decoder_layers=checkpoint.get('num_decoder_layers', NUM_DECODER_LAYERS),
            emb_size=checkpoint.get('emb_size', EMB_SIZE),
            nhead=checkpoint.get('nhead', NHEAD),
            src_vocab_size=checkpoint.get('meta_vocab_size', META_VOCAB_SIZE),
            tgt_vocab_size=checkpoint.get('midi_vocab_size', MIDI_VOCAB_SIZE),
            dim_feedforward=checkpoint.get('ffn_hid_dim', FFN_HID_DIM)
            # Dropout non salvato, usa quello di default
        ).to(DEVICE)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])

        test_loss = evaluate(loaded_model, criterion, test_dataloader, PAD_ID)
        logging.info(f"Test Loss Finale: {test_loss:.4f}")
    else:
        logging.warning("Nessun modello migliore salvato trovato. Salto valutazione su Test Set.")
        loaded_model = model # Usa l'ultimo modello se non c'è un best salvato


    # 6. Generazione di esempio (TODO implementato)
    logging.info("--- Esempio di Generazione ---")
    # Usa l'ultimo modello caricato/addestrato
    example_metadata_prompt = ["Style=Folk", "Key=A_minor", "TimeSig=4/4", "Title=kayo_kami"] # Esempio basato sul JSON iniziale
    # Potresti voler provare altri prompt
    # example_metadata_prompt = ["Style=Classical", "Key=C_Major", "TimeSig=3/4", "Title=minuet"]

    logging.info(f"Prompt metadati per generazione: {example_metadata_prompt}")

    try:
        generated_token_ids = generate_sequence(
            loaded_model, midi_tokenizer, metadata_vocab_map, example_metadata_prompt,
            max_len=MAX_SEQ_LEN_MIDI, # Genera fino alla lunghezza massima usata in training
            temperature=0.75,       # Sperimenta con la temperatura
            top_k=40,               # Sperimenta con top-k
            device=DEVICE
        )

        if generated_token_ids:
            logging.info(f"Generati {len(generated_token_ids)} token MIDI.")
            # Decodifica i token in un oggetto MIDI
            # Nota: il metodo esatto può dipendere dalla versione di miditok
            try:
                # miditok >= 2.0: tokens_to_midi restituisce un oggetto miditoolkit.MidiFile
                 generated_midi_object = midi_tokenizer.tokens_to_midi([generated_token_ids]) # Passa come lista di sequenze
            except Exception as e:
                 logging.error(f"Errore durante tokens_to_midi: {e}")
                 generated_midi_object = None


            # Salva l'oggetto MIDI in un file
            if generated_midi_object:
                 timestamp = time.strftime("%Y%m%d-%H%M%S")
                 output_filename = GENERATION_OUTPUT_DIR / f"generated_{'_'.join(example_metadata_prompt)}_{timestamp}.mid"
                 try:
                     # Usa il metodo dump dell'oggetto MidiFile (miditoolkit)
                     generated_midi_object.dump(output_filename)
                     logging.info(f"File MIDI generato salvato in: {output_filename}")
                 except Exception as e:
                      logging.error(f"Errore durante il salvataggio del file MIDI generato: {e}")

        else:
            logging.warning("Generazione fallita o ha prodotto una sequenza vuota.")

    except Exception as e:
        logging.error(f"Errore durante la generazione: {e}", exc_info=True)

    logging.info("--- Script Terminato ---")