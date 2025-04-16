import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import miditok # Assicurati di installare miditok
import pandas as pd
from pathlib import Path
import json
import math
import random
import logging
from tqdm import tqdm # Per le barre di progresso
import os

# --- Configurazione / Costanti ---
# NOTA: Questi sono solo esempi, da adattare!
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("./mutopia_data") # Directory base dei dati scaricati
SPLITS_DIR = DATA_DIR / "dataset_splits" # Directory con train/validation/test.jsonl
MIDI_BASE_DIR = DATA_DIR # Directory radice dove cercare i midi_relative_path

# Configurazioni Tokenizer MIDI (scegliere una strategia)
# MIDI_TOKENIZER_STRATEGY = miditok.REMI # Esempio
MIDI_TOKENIZER_STRATEGY = miditok.CPWord # Altro esempio
# Potrebbe essere necessario specificare parametri aggiuntivi per il tokenizer
# Vedi documentazione miditok: https://miditok.readthedocs.io/

VOCAB_PATH = DATA_DIR / "midi_vocab.json" # Dove salvare/caricare il vocabolario MIDI
METADATA_VOCAB_PATH = DATA_DIR / "metadata_vocab.json" # Vocabolario per i token metadati

# Token Speciali (assicurati che non collidano con quelli di miditok)
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>" # Start of Sequence (MIDI Target)
EOS_TOKEN = "<eos>" # End of Sequence (MIDI Target)
UNK_TOKEN = "<unk>" # Unknown (opzionale)
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
MAX_SEQ_LEN_META = 64 # Lunghezza massima sequenza metadati

# Setup Logging (opzionale ma utile)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Tokenizer e Vocabolario ---

def build_or_load_tokenizer(midi_files_for_vocab, force_build=False):
    """Costruisce o carica il tokenizer MIDI e il suo vocabolario."""
    if VOCAB_PATH.exists() and not force_build:
        logging.info(f"Caricamento vocabolario MIDI da {VOCAB_PATH}")
        tokenizer = MIDI_TOKENIZER_STRATEGY(vocab_path=VOCAB_PATH)
        # Aggiungi manualmente token speciali se non salvati correttamente con miditok
        # (miditok >= 2.0 dovrebbe gestirli meglio)
        special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SOS_META_TOKEN, EOS_META_TOKEN]
        tokenizer.add_tokens(special_tokens)

    else:
        logging.info("Creazione nuovo vocabolario MIDI...")
        # Aggiungi parametri specifici se necessario, es: pitch_range, beat_res, etc.
        tokenizer = MIDI_TOKENIZER_STRATEGY()
        # Aggiungi token speciali PRIMA di imparare il vocabolario dai file
        special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SOS_META_TOKEN, EOS_META_TOKEN]
        tokenizer.add_tokens(special_tokens)

        logging.info(f"Apprendimento vocabolario da {len(midi_files_for_vocab)} file MIDI...")
        # Nota: learn_vocabulary potrebbe richiedere molti file/tempo
        tokenizer.learn_vocabulary(midi_files_for_vocab)

        logging.info(f"Salvataggio vocabolario MIDI in {VOCAB_PATH}")
        # Assicurati che la directory esista
        VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
         # Salva il vocabolario (miditok > 2.0 usa save_params)
        try:
             tokenizer.save_params(VOCAB_PATH)
        except AttributeError: # Compatibilità versioni precedenti miditok
             tokenizer.save_vocabulary(VOCAB_PATH)
             logging.warning("Usato save_vocabulary (vecchio metodo miditok). Considera aggiornamento.")


    logging.info(f"Dimensione vocabolario MIDI (incl. speciali): {len(tokenizer)}")
    return tokenizer

def build_or_load_metadata_vocab(all_metadata_examples, force_build=False):
    """Costruisce o carica un vocabolario per i token metadati."""
    if METADATA_VOCAB_PATH.exists() and not force_build:
        logging.info(f"Caricamento vocabolario Metadati da {METADATA_VOCAB_PATH}")
        with open(METADATA_VOCAB_PATH, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        return vocab_data['token_to_id'], vocab_data['id_to_token']
    else:
        logging.info("Creazione nuovo vocabolario Metadati...")
        metadata_tokens = set()
        for meta_dict in all_metadata_examples:
            tokens = tokenize_metadata(meta_dict) # Usa la funzione definita sotto
            metadata_tokens.update(tokens)

        # Costruisci vocabolario con token speciali per metadati
        # (Nota: riutilizziamo PAD, SOS, EOS, UNK definiti per MIDI per semplicità,
        #  ma potresti volerli separati se i vocabolari sono gestiti diversamente)
        all_tokens = [PAD_TOKEN, UNK_TOKEN, SOS_META_TOKEN, EOS_META_TOKEN] + sorted(list(metadata_tokens))
        token_to_id = {token: i for i, token in enumerate(all_tokens)}
        id_to_token = {i: token for token, i in token_to_id.items()}

        vocab_data = {'token_to_id': token_to_id, 'id_to_token': id_to_token}
        logging.info(f"Salvataggio vocabolario Metadati in {METADATA_VOCAB_PATH}")
        METADATA_VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(METADATA_VOCAB_PATH, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        logging.info(f"Dimensione vocabolario Metadati (incl. speciali): {len(token_to_id)}")
        return token_to_id, id_to_token

def tokenize_metadata(metadata_dict):
    """
    TODO: Implementa la logica per convertire il dizionario di metadati
          in una lista di token stringa.
    Esempio semplice:
    """
    tokens = []
    # Metti solo alcuni metadati chiave come esempio
    if 'style' in metadata_dict and metadata_dict['style']:
        tokens.append(f"Style={metadata_dict['style'].replace(' ', '_')}")
    if 'key' in metadata_dict and metadata_dict['key']:
        tokens.append(f"Key={metadata_dict['key'].replace(' ', '_')}")
    if 'time_signature' in metadata_dict and metadata_dict['time_signature']:
        tokens.append(f"TimeSig={metadata_dict['time_signature']}")
    # Aggiungi altri metadati se necessario...
    return tokens


# --- Dataset e DataLoader ---

class MutopiaDataset(Dataset):
    def __init__(self, jsonl_path, midi_base_dir, midi_tokenizer, metadata_vocab, max_len_midi, max_len_meta):
        self.midi_base_dir = Path(midi_base_dir)
        self.midi_tokenizer = midi_tokenizer
        self.metadata_vocab = metadata_vocab # token_to_id map
        self.max_len_midi = max_len_midi
        self.max_len_meta = max_len_meta

        # ID Token Speciali
        self.pad_id = metadata_vocab[PAD_TOKEN] # Assumiamo ID condivisi per PAD
        self.sos_id = metadata_vocab[SOS_TOKEN]
        self.eos_id = metadata_vocab[EOS_TOKEN]
        self.sos_meta_id = metadata_vocab[SOS_META_TOKEN]
        self.eos_meta_id = metadata_vocab[EOS_META_TOKEN]
        self.unk_meta_id = metadata_vocab[UNK_TOKEN]

        logging.info(f"Caricamento dati da {jsonl_path}...")
        self.data = []
        skipped_count = 0
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        # Verifica che il file MIDI esista prima di aggiungerlo
                        midi_path_check = self.midi_base_dir / entry.get('midi_relative_path', '')
                        if 'midi_relative_path' in entry and midi_path_check.exists():
                            self.data.append(entry)
                        else:
                            logging.warning(f"File MIDI non trovato o path mancante, salto: {entry.get('midi_relative_path', 'N/A')}")
                            skipped_count += 1
                    except json.JSONDecodeError:
                        logging.warning(f"Riga JSON malformata in {jsonl_path}, salto.")
                    except Exception as e:
                         logging.warning(f"Errore caricando riga: {e}")
                         skipped_count += 1

        except FileNotFoundError:
             logging.error(f"File dataset non trovato: {jsonl_path}")
             raise

        logging.info(f"Caricati {len(self.data)} campioni validi da {jsonl_path}. Saltati: {skipped_count}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        metadata = entry['metadata']
        midi_relative_path = entry['midi_relative_path']
        midi_full_path = self.midi_base_dir / midi_relative_path

        # 1. Tokenizza Metadati (Source Sequence)
        meta_tokens_str = tokenize_metadata(metadata) # Lista di stringhe token
        # Converte in ID, gestendo token non noti
        meta_token_ids = [self.metadata_vocab.get(token, self.unk_meta_id) for token in meta_tokens_str]
        # Aggiungi SOS/EOS e tronca se necessario
        src_seq = [self.sos_meta_id] + meta_token_ids[:self.max_len_meta-2] + [self.eos_meta_id]

        # 2. Tokenizza MIDI (Target Sequence)
        try:
            # Usa miditok per ottenere la lista di ID interi direttamente
            midi_token_ids = self.midi_tokenizer(midi_full_path)[0] # Prende la prima traccia (o unico output)
        except Exception as e:
            # Se un MIDI è corrotto o dà errore, ritorna None o un valore speciale
            logging.warning(f"Errore tokenizzazione MIDI {midi_full_path}: {e}. Salto campione.")
            # Potremmo ritornare None e gestirlo nella collate_fn, oppure sollevare eccezione
            # Per semplicità, solleviamo eccezione per ora, da gestire meglio in produzione
            raise RuntimeError(f"Errore tokenizzazione MIDI {midi_full_path}: {e}") from e

        # Aggiungi SOS/EOS e tronca se necessario
        tgt_seq = [self.sos_id] + midi_token_ids[:self.max_len_midi-2] + [self.eos_id]

        return torch.tensor(src_seq, dtype=torch.long), torch.tensor(tgt_seq, dtype=torch.long)


def pad_collate_fn(batch):
    """
    Collate function per DataLoader. Esegue il padding e crea le maschere.
    """
    # Separa sorgenti e target
    src_batch, tgt_batch = zip(*batch)

    # Trova l'ID del PAD token (assumendo sia lo stesso per entrambi i vocabolari)
    # Dovresti ottenerlo dal tokenizer o dal vocabolario in modo più robusto
    pad_id = 0 # Assumiamo che PAD_TOKEN sia mappato a 0

    # Padding delle sequenze sorgente (metadati)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    # Padding delle sequenze target (MIDI)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)

    # Creazione maschere di padding (True dove c'è padding)
    src_padding_mask = (src_padded == pad_id)
    tgt_padding_mask = (tgt_padded == pad_id)

    return src_padded, tgt_padded, src_padding_mask, tgt_padding_mask


# --- Modello Transformer ---

class PositionalEncoding(nn.Module):
    """Implementazione del Positional Encoding sinusoidale."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Registra come buffer, non come parametro

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead,
                 src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=emb_size, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True # batch_first=True è importante!
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size) # Proietta output decoder a dimensione vocabolario target
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        # Inizializzazione pesi (opzionale ma spesso utile)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, memory_key_padding_mask, tgt_mask=None):
        """
        Forward pass.
        Args:
            src: Sequenza metadati input (batch_size, src_seq_len)
            tgt: Sequenza MIDI target (batch_size, tgt_seq_len) - per teacher forcing
            src_padding_mask: Maschera padding per src (batch_size, src_seq_len) - True dove c'è PAD
            tgt_padding_mask: Maschera padding per tgt (batch_size, tgt_seq_len) - True dove c'è PAD
            memory_key_padding_mask: Uguale a src_padding_mask, passata all'attenzione incrociata del decoder.
            tgt_mask: Maschera causale per il decoder (tgt_seq_len, tgt_seq_len) - generata automaticamente se None
        """
        # Embedding e Positional Encoding
        src_emb = self.positional_encoding(self.src_tok_emb(src).transpose(0, 1)) # Transformer si aspetta (seq_len, batch, emb)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt).transpose(0, 1)) # Transformer si aspetta (seq_len, batch, emb)

        # Trasponi gli input perché nn.Transformer si aspetta (Seq_Len, Batch, Emb)
        # Le maschere devono corrispondere: src_padding_mask (Batch, Seq_Len), tgt_mask (Seq_Len, Seq_Len)

        # Genera maschera causale per il decoder se non fornita
        if tgt_mask is None:
             tgt_len = tgt_emb.size(0)
             tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(DEVICE)

        # Passaggio attraverso il Transformer
        # Nota: src_key_padding_mask e tgt_key_padding_mask corrispondono alle nostre maschere di padding
        outs = self.transformer(src_emb, tgt_emb,
                                tgt_mask=tgt_mask,
                                src_key_padding_mask=src_padding_mask,
                                tgt_key_padding_mask=tgt_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask) # memory_key_padding_mask è la maschera dell'encoder usata dal decoder

        # Proietta l'output sulla dimensione del vocabolario target
        # Riporta l'output a (Batch, Seq_Len, VocabSize)
        return self.generator(outs.transpose(0, 1))

    def encode(self, src, src_mask):
        """Funzione per usare solo l'encoder (utile per inferenza)."""
        src_emb = self.positional_encoding(self.src_tok_emb(src).transpose(0,1))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)

    def decode(self, tgt, memory, tgt_mask, tgt_padding_mask=None, memory_key_padding_mask=None):
        """Funzione per usare solo il decoder (utile per inferenza)."""
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt).transpose(0,1))
        return self.transformer.decoder(tgt_emb, memory,
                                         tgt_mask=tgt_mask,
                                         tgt_key_padding_mask=tgt_padding_mask,
                                         memory_key_padding_mask=memory_key_padding_mask)


# --- Ciclo di Addestramento e Valutazione ---

def train_epoch(model, optimizer, criterion, train_dataloader):
    model.train() # Imposta modalità training
    total_loss = 0
    processed_batches = 0

    progress_bar = tqdm(train_dataloader, desc="Training Epoch")
    for src, tgt, src_padding_mask, tgt_padding_mask in progress_bar:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src_padding_mask = src_padding_mask.to(DEVICE)
        tgt_padding_mask = tgt_padding_mask.to(DEVICE)

        # Prepara input e output per il modello
        # Il target input per il decoder è la sequenza tgt senza l'ultimo token (<EOS>)
        tgt_input = tgt[:, :-1]
        tgt_input_padding_mask = tgt_padding_mask[:, :-1]

        # Il target reale per la loss è la sequenza tgt senza il primo token (<SOS>)
        tgt_out = tgt[:, 1:]
        tgt_out_padding_mask = tgt_padding_mask[:, 1:] # Maschera per la loss

        optimizer.zero_grad()

        # Forward pass
        logits = model(src=src,
                       tgt=tgt_input,
                       src_padding_mask=src_padding_mask,
                       tgt_padding_mask=tgt_input_padding_mask, # Maschera per l'input del decoder
                       memory_key_padding_mask=src_padding_mask) # Maschera dell'encoder passata al decoder

        # Calcola loss ignorando il padding nel target reale
        # La loss CrossEntropy si aspetta (Batch * SeqLen, VocabSize) e (Batch * SeqLen)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        # Filtra la loss per ignorare il padding (necessario se criterion non ha ignore_index)
        # Se criterion ha ignore_index=pad_id, questo non è strettamente necessario, ma verifica!
        # mask = (tgt_out != pad_id).view(-1)
        # loss = (loss * mask).sum() / mask.sum() # Loss media solo sui token non-pad

        loss.backward()
        # Clip gradient (opzionale ma utile per prevenire esplosione gradienti)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item()
        processed_batches += 1
        progress_bar.set_postfix({'train_loss': total_loss / processed_batches})


    return total_loss / len(train_dataloader)


def evaluate(model, criterion, val_dataloader):
    model.eval() # Imposta modalità valutazione
    total_loss = 0
    processed_batches = 0

    progress_bar = tqdm(val_dataloader, desc="Validation Epoch")
    with torch.no_grad(): # Disabilita calcolo gradienti
        for src, tgt, src_padding_mask, tgt_padding_mask in progress_bar:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            src_padding_mask = src_padding_mask.to(DEVICE)
            tgt_padding_mask = tgt_padding_mask.to(DEVICE)

            tgt_input = tgt[:, :-1]
            tgt_input_padding_mask = tgt_padding_mask[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_out_padding_mask = tgt_padding_mask[:, 1:]

            logits = model(src=src,
                           tgt=tgt_input,
                           src_padding_mask=src_padding_mask,
                           tgt_padding_mask=tgt_input_padding_mask,
                           memory_key_padding_mask=src_padding_mask)

            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            # mask = (tgt_out != pad_id).view(-1)
            # loss = (loss * mask).sum() / mask.sum()

            total_loss += loss.item()
            processed_batches += 1
            progress_bar.set_postfix({'val_loss': total_loss / processed_batches})

    return total_loss / len(val_dataloader)


# --- Esecuzione Principale ---

if __name__ == "__main__":

    # 1. Prepara Tokenizer e Vocabolari
    logging.info("--- Preparazione Tokenizer e Vocabolari ---")
    # Trova tutti i file MIDI per costruire il vocabolario (potrebbe essere lento)
    # Idealmente, usa solo il training set per il vocabolario MIDI
    train_jsonl_path = SPLITS_DIR / "train.jsonl"
    all_train_metadata = []
    midi_files_for_vocab_build = []
    try:
         with open(train_jsonl_path, 'r', encoding='utf-8') as f:
              for line in f:
                   try:
                        entry = json.loads(line)
                        midi_path = MIDI_BASE_DIR / entry['midi_relative_path']
                        if midi_path.exists():
                            midi_files_for_vocab_build.append(str(midi_path))
                            all_train_metadata.append(entry['metadata'])
                   except Exception:
                        pass # Ignora righe/file problematici per la costruzione del vocab
         logging.info(f"Trovati {len(midi_files_for_vocab_build)} file MIDI nel training set per il vocabolario.")
    except FileNotFoundError:
         logging.error(f"File di training non trovato: {train_jsonl_path}. Impossibile costruire vocabolari.")
         exit()

    # Costruisci/Carica tokenizer e vocabolari
    midi_tokenizer = build_or_load_tokenizer(midi_files_for_vocab_build, force_build=False)
    metadata_vocab_map, _ = build_or_load_metadata_vocab(all_train_metadata, force_build=False)

    # Ottieni dimensioni vocabolario e ID padding
    MIDI_VOCAB_SIZE = len(midi_tokenizer)
    META_VOCAB_SIZE = len(metadata_vocab_map)
    PAD_ID = metadata_vocab_map[PAD_TOKEN] # Assumi pad_id sia lo stesso
    logging.info(f"Vocabolario MIDI: {MIDI_VOCAB_SIZE}, Vocabolario Meta: {META_VOCAB_SIZE}, PAD_ID: {PAD_ID}")


    # 2. Crea Dataset e DataLoader
    logging.info("--- Creazione Dataset e DataLoader ---")
    try:
        train_dataset = MutopiaDataset(SPLITS_DIR / "train.jsonl", MIDI_BASE_DIR, midi_tokenizer, metadata_vocab_map, MAX_SEQ_LEN_MIDI, MAX_SEQ_LEN_META)
        val_dataset = MutopiaDataset(SPLITS_DIR / "validation.jsonl", MIDI_BASE_DIR, midi_tokenizer, metadata_vocab_map, MAX_SEQ_LEN_MIDI, MAX_SEQ_LEN_META)
        # test_dataset = MutopiaDataset(SPLITS_DIR / "test.jsonl", ...) # Per la valutazione finale
    except FileNotFoundError:
        logging.error("Uno o più file .jsonl degli split non trovati. Interruzione.")
        exit()
    except Exception as e:
        logging.error(f"Errore durante la creazione dei Dataset: {e}", exc_info=True)
        exit()


    if len(train_dataset) == 0 or len(val_dataset) == 0:
         logging.error("Dataset di training o validazione vuoti. Controllare la fase precedente.")
         exit()

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn, num_workers=0) # num_workers > 0 può velocizzare ma causare problemi su Windows
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn, num_workers=0)
    # test_dataloader = DataLoader(test_dataset, ...)


    # 3. Inizializza Modello, Ottimizzatore, Loss
    logging.info("--- Inizializzazione Modello ---")
    model = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD,
        META_VOCAB_SIZE, MIDI_VOCAB_SIZE, FFN_HID_DIM, DROPOUT
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID) # Ignora padding nella loss

    # Conta parametri (opzionale)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Numero totale parametri addestrabili: {total_params:,}")

    # 4. Ciclo di Addestramento
    logging.info("--- Inizio Addestramento ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 3 # Numero di epoche senza miglioramenti prima di fermarsi (early stopping)

    for epoch in range(1, EPOCHS + 1):
        logging.info(f"--- Epoch {epoch}/{EPOCHS} ---")

        train_loss = train_epoch(model, optimizer, criterion, train_dataloader)
        val_loss = evaluate(model, criterion, val_dataloader)

        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Salva il modello migliore e implementa early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # TODO: Implementare salvataggio checkpoint modello
            model_save_path = DATA_DIR / f"transformer_mutopia_best_epoch_{epoch}.pt"
            logging.info(f"Miglioramento validation loss. Salvataggio modello in {model_save_path}")
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': best_val_loss,
            # }, model_save_path)
        else:
            epochs_no_improve += 1
            logging.info(f"Validation loss non migliorata. Epoche senza miglioramento: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            logging.info(f"Nessun miglioramento per {patience} epoche consecutive. Early stopping.")
            break

    logging.info("--- Addestramento Terminato ---")

    # 5. TODO: Valutazione finale sul Test Set
    # logging.info("--- Valutazione su Test Set ---")
    # test_loss = evaluate(model, criterion, test_dataloader)
    # logging.info(f"Test Loss Finale: {test_loss:.4f}")

    # 6. TODO: Implementare la logica di generazione/inferenza per creare nuovi MIDI
    # Esempio concettuale:
    # metadata_input = ["<sos_meta>", "Style=Classical", "Key=G_Major", "TimeSig=3/4", "<eos_meta>"]
    # generated_midi_ids = generate_sequence(model, metadata_input, start_token_id=SOS_ID, end_token_id=EOS_ID, ...)
    # midi_object = midi_tokenizer.tokens_to_midi([generated_midi_ids])
    # midi_object.dump("generated_output.mid")