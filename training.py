import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
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
from functools import partial # IMPORTANTE: Aggiunto per DataLoader collate_fn
from symusic import Score
import re
import numpy as np
import argparse

# --- USAGE: ---
# python training.py --data_dir PATH/TO/DATASET --model_save_dir PATH/TO/SAVE/MODELS 

# --- Configurazione del logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configurazione / Costanti ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurazioni Tokenizer MIDI (scegliere una strategia)
MIDI_TOKENIZER_STRATEGY = miditok.REMI # Esempio scelto
MIDI_VOCAB_TARGET_SIZE = 50000 # Esempio: Dimensione target per il vocabolario MIDI se addestrato

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
EPOCHS = 100
BATCH_SIZE = 512 # Riduci se hai poca memoria GPU
LEARNING_RATE = 0.0001
EMB_SIZE = 128 # Dimensione embedding
NHEAD = 4 # Numero di head nell'attention (deve dividere EMB_SIZE)
FFN_HID_DIM = 256 # Dimensione layer nascosto FeedForward
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DROPOUT = 0.1
MAX_SEQ_LEN_MIDI = 512 # Lunghezza massima sequenza MIDI (suddivide in più chunks se più lunga)
MAX_SEQ_LEN_META = 128 # Aumentata per includere potenziale titolo lungo

# Programmi MIDI considerati come "pianoforte"
PIANO_PROGRAMS = list(range(0, 8))

# --- NUOVI IPERPARAMETRI PER MODALITÀ DI PROCESSAMENTO ---
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
    tokens = [] #
    key_to_tokenize_str = None # Stringa che conterrà la tonalità da tokenizzare

    # Logica per determinare la stringa della tonalità
    # Se PROCESSING_MODE è "piano_only" e il campo 'transposed_to_key' è disponibile e valido
    if PROCESSING_MODE == "piano_only" and \
       'transposed_to_key' in metadata_dict and \
       metadata_dict['transposed_to_key']: #

        raw_transposed_info = metadata_dict['transposed_to_key'] #

        # dataset_creator.py imposta questo campo a "C major / a minor"
        # Creiamo un token specifico per rappresentare questo stato di trasposizione mirata.
        if raw_transposed_info == "C major / a minor":
            key_to_tokenize_str = "Target_Cmaj_Amin" 
        else:
            # Se il valore è diverso, sanitizzalo in modo generico
            # (questo caso è meno probabile se dataset_creator.py è consistente)
            temp_key = str(raw_transposed_info).replace(' ', '_').replace('/', '_').replace('#','sharp')
            key_to_tokenize_str = re.sub(r'[^a-zA-Z0-9_]', '', temp_key) #
            if not key_to_tokenize_str: # Se la sanitizzazione produce una stringa vuota
                key_to_tokenize_str = "Unknown_Transposed_Key"

    # Fallback se non siamo in "piano_only" mode o se 'transposed_to_key' non è stato usato/trovato
    if not key_to_tokenize_str:
        if 'key' in metadata_dict and metadata_dict['key']: # Logica originale se 'key' esiste
            key_to_tokenize_str = metadata_dict['key'] #
        elif 'music21_detected_key' in metadata_dict and metadata_dict['music21_detected_key']:
            # Utilizza 'music21_detected_key' se 'key' non c'è (popolato da dataset_creator)
            key_to_tokenize_str = metadata_dict['music21_detected_key']
        elif 'mido_declared_key_signature' in metadata_dict and metadata_dict['mido_declared_key_signature']:
            # Utilizza 'mido_declared_key_signature' come ulteriore fallback (popolato da dataset_creator)
            key_to_tokenize_str = metadata_dict['mido_declared_key_signature']

    if key_to_tokenize_str: #
        # Sanitizzazione generale per qualsiasi stringa di tonalità scelta
        # Rimuove spazi, sostituisce caratteri speciali per creare un token valido
        clean_key_token = str(key_to_tokenize_str).replace(' ', '_').replace('#', 'sharp') #
        # Mantieni solo caratteri alfanumerici e underscore
        clean_key_token = re.sub(r'[^a-zA-Z0-9_]', '', clean_key_token) #

        if clean_key_token: # Assicurati che non sia una stringa vuota dopo la pulizia
            tokens.append(f"Key={clean_key_token}") #

    # 2. Metro (Time Signature) - Invariato
    if 'time_signature' in metadata_dict and metadata_dict['time_signature']:
        tokens.append(f"TimeSig={metadata_dict['time_signature']}")

    # 3. BPM (Battiti Per Minuto) - Nuovo, categorizzato
    if 'bpm_rounded' in metadata_dict and metadata_dict['bpm_rounded'] is not None:
        bpm = metadata_dict['bpm_rounded']
        if bpm <= 0: # Improbabile ma gestiamo
            token_bpm = "Tempo_Unknown"
        elif bpm <= 60:
            token_bpm = "Tempo_VerySlow"  # (es. Largo, Grave)
        elif bpm <= 76: # Fino a Adagio
            token_bpm = "Tempo_Slow"
        elif bpm <= 108: # Fino a Andante/Moderato
            token_bpm = "Tempo_Moderate"
        elif bpm <= 132: # Fino a Allegro
            token_bpm = "Tempo_Fast"
        elif bpm <= 168: # Fino a Vivace/Presto
            token_bpm = "Tempo_VeryFast"
        else: # Prestissimo
            token_bpm = "Tempo_ExtremelyFast"
        tokens.append(token_bpm)

    # 4. Velocity Media - Nuovo, categorizzato
    if 'avg_velocity_rounded' in metadata_dict and metadata_dict['avg_velocity_rounded'] is not None:
        avg_vel = metadata_dict['avg_velocity_rounded']
        if avg_vel < 0 : token_avg_vel = "AvgVel_Unknown" # Improbabile
        elif avg_vel <= 35: # Pianissimo (pp) / Piano (p)
            token_avg_vel = "AvgVel_VeryLow"
        elif avg_vel <= 60: # MezzoPiano (mp)
            token_avg_vel = "AvgVel_Low"
        elif avg_vel <= 85: # MezzoForte (mf)
            token_avg_vel = "AvgVel_Medium"
        elif avg_vel <= 110: # Forte (f)
            token_avg_vel = "AvgVel_High"
        else: # Fortissimo (ff)
            token_avg_vel = "AvgVel_VeryHigh"
        tokens.append(token_avg_vel)

    # 5. Range di Velocity - Nuovo, categorizzato
    if 'velocity_range_rounded' in metadata_dict and metadata_dict['velocity_range_rounded'] is not None:
        vel_range = metadata_dict['velocity_range_rounded']
        if vel_range < 0: token_vel_range = "VelRange_Unknown" # Improbabile
        elif vel_range <= 15: # Poca variazione dinamica
            token_vel_range = "VelRange_Narrow"
        elif vel_range <= 40:
            token_vel_range = "VelRange_Medium"
        elif vel_range <= 70:
            token_vel_range = "VelRange_Wide"
        else: # Variazione dinamica molto ampia
            token_vel_range = "VelRange_VeryWide"
        tokens.append(token_vel_range)

    # 6. Numero di Strumenti (Opzionale, ma può essere utile per lo "stile")
    num_instruments = 0
    if 'midi_instruments' in metadata_dict and isinstance(metadata_dict['midi_instruments'], list):
        num_instruments = len(metadata_dict['midi_instruments'])
    
    if num_instruments == 0:
        token_num_inst = "NumInst_None" # O potresti ometterlo
    elif num_instruments == 1:
        token_num_inst = "NumInst_Solo"
    elif num_instruments == 2:
        token_num_inst = "NumInst_Duet"
    elif num_instruments <= 4: # Trio, Quartetto
        token_num_inst = "NumInst_SmallChamber"
    elif num_instruments <= 8: # Ensemble piccolo/medio
        token_num_inst = "NumInst_MediumEnsemble"
    else: # Ensemble grande
        token_num_inst = "NumInst_LargeEnsemble"
    tokens.append(token_num_inst)


    # 7. Nomi degli Strumenti (Logica precedente, adattata per chiarezza)
    instrument_tokens_added_from_midi_list = False
    if 'midi_instruments' in metadata_dict and isinstance(metadata_dict['midi_instruments'], list) and metadata_dict['midi_instruments']:
        # Priorità alla lista di strumenti direttamente dal MIDI se presente e valida e non vuota
        for instrument_name in metadata_dict['midi_instruments']:
            if instrument_name and isinstance(instrument_name, str): # Verifica aggiuntiva
                # Pulisci il nome dello strumento per creare un token valido
                # Rimuovi caratteri speciali, spazi, normalizza case se necessario.
                # Qui usiamo una pulizia semplice.
                clean_instrument_name = instrument_name.replace(' ', '_').replace('(', '').replace(')', '').replace('#','sharp')
                clean_instrument_name = re.sub(r'[^a-zA-Z0-9_]', '', clean_instrument_name) # Mantieni solo alfanumerici e underscore
                if clean_instrument_name: # Assicurati che non sia vuoto dopo la pulizia
                    tokens.append(f"Instrument={clean_instrument_name}")
                    instrument_tokens_added_from_midi_list = True
                    
    # 8. Uso del Sustain Pedal e del Pitch Bend
    if 'sustain_pedal_used' in metadata_dict and metadata_dict['sustain_pedal_used']:
        tokens.append("Sustain=Used")
    else:
        tokens.append("Sustain=NotUsed")

    if 'pitch_bend_used' in metadata_dict and metadata_dict['pitch_bend_used']:
        tokens.append("PitchBend=Used")
    else:
        tokens.append("PitchBend=NotUsed")
        
    
    # Fallback a mutopiainstrument se midi_instruments non ha prodotto token
    # (o se vuoi che 'mutopiainstrument' aggiunga/sovrascriva - modifica la logica di conseguenza)
    if not instrument_tokens_added_from_midi_list and 'mutopiainstrument' in metadata_dict and metadata_dict['mutopiainstrument']:
        instrument_string = metadata_dict['mutopiainstrument']
        # Sostituisci " and " e altri separatori comuni
        instrument_string_normalized = re.sub(r'\s+(?:and|,|&)\s+', ',', instrument_string, flags=re.IGNORECASE)
        instrument_string_normalized = re.sub(r'[()]', '', instrument_string_normalized) # Rimuovi parentesi

        instrument_names_from_ly = [name.strip() for name in instrument_string_normalized.split(',') if name.strip()]
        
        for instrument_name in instrument_names_from_ly:
            if instrument_name:
                clean_instrument_name = instrument_name.replace(' ', '_').replace('#','sharp')
                clean_instrument_name = re.sub(r'[^a-zA-Z0-9_]', '', clean_instrument_name)
                if clean_instrument_name:
                    tokens.append(f"Instrument={clean_instrument_name}")
    
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
    def __init__(self, jsonl_path, midi_tokenizer, metadata_vocab_map,
                 max_len_midi_padded, max_len_meta, splits_dir, filter_key=None,
                 # min_chunk_len_midi is no longer needed if chunks are pre-filtered
                 ):
        # midi_base_dir is no longer strictly needed if we don't load MIDI files here
        # self.midi_base_dir = Path(midi_base_dir) 
        self.splits_dir = Path(splits_dir)
        self.midi_tokenizer = midi_tokenizer # Still needed for special token IDs
        self.metadata_vocab_map = metadata_vocab_map
        self.max_len_midi_padded = max_len_midi_padded # Max length AFTER SOS/EOS and padding
        self.max_len_meta = max_len_meta
        self.filter_key = filter_key
        # self.min_chunk_len_midi = min_chunk_len_midi # Assuming pre-chunked data is already filtered

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

        logging.info(f"Inizializzazione Dataset da {jsonl_path} (assumendo dati pre-chunked e pre-tokenizzati).")
        self.samples = [] 

        skipped_key_filter = 0
        skipped_missing_token_ids = 0
        skipped_token_ids_too_short_pre_sos_eos = 0 # If any check is needed
        processed_json_lines = 0

        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        entry = json.loads(line) # Each line is one pre-processed chunk/sample
                        processed_json_lines += 1
                        actual_metadata = entry.get('metadata', {})

                        if self.filter_key and actual_metadata.get('key') != self.filter_key:
                            skipped_key_filter += 1
                            continue
                        
                        # 1. Prepara la sequenza dei metadati (src) - same as before
                        meta_tokens_str = tokenize_metadata(actual_metadata)
                        meta_token_ids = [self.metadata_vocab_map.get(token, self.unk_meta_id) for token in meta_tokens_str]
                        src_seq_list = [self.sos_meta_id] + meta_token_ids[:self.max_len_meta-2] + [self.eos_meta_id]
                        # Padding for src_seq_list will be handled by collate_fn, or do it here if fixed length needed before collate
                        src_tensor = torch.tensor(src_seq_list, dtype=torch.long)


                        # 2. Recupera il PERCORSO del file dei token, non la lista
                        token_ids_relative_path = entry.get('token_ids_path')
                        if not token_ids_relative_path:
                            skipped_missing_token_ids +=1
                            # logging.warning(f"Entry {entry.get('id', 'N/A')} in {jsonl_path} manca 'token_ids'. Saltato.")
                            if processed_json_lines % 1000 == 0: # Log occasionally
                                logging.warning(f"Entry {entry.get('id', 'N/A')} in {jsonl_path} (line ~{line_num+1}) manca 'token_ids'. Saltato.")
                            continue
                        
                        # 3. Salva il tensore dei metadati e il PERCORSO nella lista dei campioni
                        # Non carichiamo il file binario qui, lo faremo in __getitem__ per efficienza
                        self.samples.append((src_tensor, token_ids_relative_path))

                    except json.JSONDecodeError:
                        logging.warning(f"Errore decodifica JSON linea {line_num+1} in {jsonl_path}")
                    except Exception as e:
                        logging.error(f"Errore imprevisto processando entry da {jsonl_path} (linea {line_num+1}): ID {entry.get('id', 'N/A')}. Errore: {e}", exc_info=False) # Set exc_info=True for full trace

        except FileNotFoundError:
            logging.error(f"File dataset non trovato: {jsonl_path}")
            raise
        
        logging.info(f"Dataset inizializzato. Numero totale di campioni (pre-chunked): {len(self.samples)}")
        logging.info(f"Linee JSON totali processate: {processed_json_lines}")
        if skipped_key_filter > 0: logging.info(f"Campioni saltati per filtro tonalità: {skipped_key_filter}")
        if skipped_missing_token_ids > 0: logging.info(f"Campioni saltati per 'token_ids' mancanti: {skipped_missing_token_ids}")
        # if skipped_token_ids_too_short_pre_sos_eos > 0: logging.info(f"Campioni saltati per token_ids troppo corti: {skipped_token_ids_too_short_pre_sos_eos}")


        if not self.samples:
            # Potentially critical if no samples loaded, could be due to all files being filtered,
            # or data format issues (e.g., all missing 'token_ids')
            logging.error(f"Nessun campione caricato da {jsonl_path}. Controlla i log e il formato del file JSONL.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):        
        # 1. Recupera il tensore dei metadati e il percorso relativo del file dei token
        src_tensor, token_ids_relative_path = self.samples[idx]

        # 2. Costruisci il percorso assoluto e carica l'array di token dal file .npy
        try:
            # self.splits_dir è la cartella /dataset_splits/
            full_token_path = self.splits_dir / token_ids_relative_path
            raw_midi_ids_from_file = np.load(full_token_path)
        except Exception as e:
            logging.error(f"Errore nel caricare il file di token binario: {full_token_path}. Errore: {e}")
            return None # Il collate_fn salterà questo campione

        # 3. La logica per aggiungere SOS/EOS e creare il tensore finale rimane la stessa di prima
        max_data_tokens = self.max_len_midi_padded - 2
        truncated_midi_ids = raw_midi_ids_from_file[:max_data_tokens]
        
        tgt_seq_list = [self.sos_midi_id] + truncated_midi_ids.tolist() + [self.eos_midi_id]
        tgt_tensor = torch.tensor(tgt_seq_list, dtype=torch.long)

        # 4. Restituisci la coppia di tensori, pronta per il collate_fn
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

        # Prepara gli input e gli output per il modello
        tgt_input = tgt[:, :-1]
        tgt_input_padding_mask = tgt_padding_mask[:, :-1]
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad(set_to_none=True) # 'set_to_none=True' è leggermente più veloce

        # Forward pass
        logits = model(src=src, tgt=tgt_input,
                       src_padding_mask=src_padding_mask,
                       tgt_padding_mask=tgt_input_padding_mask,
                       memory_key_padding_mask=src_padding_mask)
        
        # Calcolo della loss
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        # Backward pass e step dell'ottimizzatore
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss

        # Elimina esplicitamente i tensori che non servono più in questo ciclo.
        # Questo segnala al gestore di memoria di PyTorch che può liberare lo spazio.
        del src, tgt, src_padding_mask, tgt_padding_mask, tgt_input, tgt_out, logits, loss
        
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
    # Gestione degli argomenti
    parser = argparse.ArgumentParser(description="Addestra un modello Transformer per la generazione musicale.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Percorso della cartella base del dataset, che contiene la sottocartella 'dataset_splits'."
    )
    parser.add_argument(
        "--model_save_dir",
        type=Path,
        required=True,
        help="Percorso della cartella dove salvare i checkpoint del modello."
    )
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    MODEL_SAVE_DIR = args.model_save_dir

    # I percorsi derivati ora usano le nuove variabili
    SPLITS_DIR = DATA_DIR / "dataset_splits"
    VOCAB_PATH = DATA_DIR / "midi_vocab.json"
    METADATA_VOCAB_PATH = DATA_DIR / "metadata_vocab.json"
    MIDI_BASE_DIR = DATA_DIR # Usato per trovare file MIDI di esempio per il vocabolario se necessario

    # Assicurati che la cartella di salvataggio esista
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Utilizzo del dataset da: {DATA_DIR}")
    logging.info(f"I modelli verranno salvati in: {MODEL_SAVE_DIR}")
    
    
    logging.info("--- Preparazione Tokenizer e Vocabolari ---")
    train_jsonl_path = SPLITS_DIR / "train.jsonl" # This JSONL is now expected to be pre-chunked
    all_train_metadata_for_vocab = [] # For building metadata vocab
    # midi_files_for_vocab_build is only needed if tokenizer vocab needs to be built/trained by training.py
    # If dataset_creator.py guarantees vocab creation, this can be optional.
    midi_files_for_tokenizer_vocab_build = []
    force_rebuild_vocabs = False # Set to True to rebuild both vocabs if needed

    logging.info(f"Inizio lettura {train_jsonl_path} per costruire vocabolario metadati (e opzionalmente MIDI tokenizer)...")
    try:
        if not train_jsonl_path.is_file(): raise FileNotFoundError(f"JSONL non trovato: {train_jsonl_path}")
        
        # For metadata vocab, we still need to parse the metadata from the training split
        # For MIDI tokenizer vocab, if it's not built by dataset_creator or needs to be verified/rebuilt
        # we might need to point to original MIDI files or assume it's done.
        # For this example, we assume metadata vocab is built here.
        # MIDI tokenizer is loaded/built using its own path, potentially with a sample of files if building.
        
        # Collect all metadata for its vocab
        # (This part is somewhat redundant if dataset_creator.py uses a similar process for its own needs,
        # but essential for this script to define its metadata vocab independently or verify it)
        temp_midi_paths_for_midi_vocab = set() # To get a diverse set for tokenizer BPE if needed

        with open(train_jsonl_path, 'r', encoding='utf-8') as f:
            for line_count, line in enumerate(f): # Iterate over a subset for speed if only for vocab
                if line_count > 20000 and not force_rebuild_vocabs: # Process a limited number of lines for vocab building speed
                    # if building vocab from many small chunks, this might not be representative of full file metadata
                    # Consider having a separate metadata dump or point to original files for vocab build
                    # logging.info("Limiting metadata vocab scan to first 20000 entries for speed.")
                    # break 
                    pass # Process all for metadata vocab

                try:
                    entry = json.loads(line)
                    metadata_dict = entry.get('metadata', {})
                    all_train_metadata_for_vocab.append(metadata_dict)
                    
                    # If MIDI tokenizer needs BPE training and vocab isn't final from dataset_creator
                    # we might need to collect original MIDI paths.
                    # However, with pre-chunking, this script relies on `VOCAB_PATH`.
                    # We can pass a sample of original MIDI files if `VOCAB_PATH` needs to be created here.
                    # This part becomes more complex if `dataset_creator.py` and `training.py` don't share `VOCAB_PATH`
                    # For now, assume `build_or_load_tokenizer` handles its own file needs if `VOCAB_PATH` is missing.

                except json.JSONDecodeError: pass
                except Exception: pass
        
        # Example: If VOCAB_PATH (for MIDI) might not exist, gather some source MIDI files
        # This step is ideally done by dataset_creator or a dedicated vocab script
        if not VOCAB_PATH.exists(): # Or if force_rebuild_vocabs
            logging.info("MIDI vocab might need building. Trying to find original MIDI files from dataset (example).")
            # This is a conceptual step: find original files referenced in train_jsonl_path
            # For simplicity, we'll assume dataset_creator.py created the vocab,
            # or we pass a generic list of MIDI files if building here from scratch.
            # Example: scan MIDI_BASE_DIR for a few files
            candidate_files = list(MIDI_BASE_DIR.rglob("*.mid")) + list(MIDI_BASE_DIR.rglob("*.midi"))
            if candidate_files:
                 midi_files_for_tokenizer_vocab_build = [str(p) for p in random.sample(candidate_files, min(len(candidate_files), 200))] # sample 200
                 logging.info(f"Using {len(midi_files_for_tokenizer_vocab_build)} sample MIDI files if vocab needs building.")


    except Exception as e:
        logging.error(f"ERRORE CRITICO lettura {train_jsonl_path} per vocabolari: {e}", exc_info=True)
        sys.exit(1)

    if not all_train_metadata_for_vocab:
        logging.error("ERRORE CRITICO: Nessun metadato per vocabolario metadati.")
        # This could happen if train.jsonl is empty or malformed.
        # sys.exit(1) # Allow to proceed if metadata vocab can be loaded
    
    # midi_tokenizer will load from VOCAB_PATH or build if files are provided and force_build=True
    midi_tokenizer = build_or_load_tokenizer(
        midi_file_paths=midi_files_for_tokenizer_vocab_build if not VOCAB_PATH.exists() or force_rebuild_vocabs else None, 
        force_build=force_rebuild_vocabs
    )
    metadata_vocab_map, _ = build_or_load_metadata_vocab(all_train_metadata_for_vocab, force_build=force_rebuild_vocabs)

    MIDI_VOCAB_SIZE = len(midi_tokenizer)
    META_VOCAB_SIZE = len(metadata_vocab_map)
    try:
        MIDI_PAD_ID = midi_tokenizer[MIDI_PAD_TOKEN_NAME]
        META_PAD_ID = metadata_vocab_map[META_PAD_TOKEN_NAME]
    except KeyError as e:
        logging.error(f"ERRORE CRITICO: Token PAD non trovato dopo inizializzazione vocabolari: {e}")
        sys.exit(1)

    logging.info(f"MIDI Vocab Size: {MIDI_VOCAB_SIZE}, MIDI PAD ID: {MIDI_PAD_ID}")
    logging.info(f"Meta Vocab Size: {META_VOCAB_SIZE}, Meta PAD ID: {META_PAD_ID}")

    logging.info("--- Creazione Dataset e DataLoader (con dati pre-chunked) ---")
    try:
        train_dataset = MutopiaDataset(SPLITS_DIR / "train.jsonl", midi_tokenizer, metadata_vocab_map, 
                                     MAX_SEQ_LEN_MIDI, MAX_SEQ_LEN_META,
                                     splits_dir=SPLITS_DIR) # <-- AGGIUNGI QUESTO
        val_dataset = MutopiaDataset(SPLITS_DIR / "validation.jsonl", midi_tokenizer, metadata_vocab_map,
                                   MAX_SEQ_LEN_MIDI, MAX_SEQ_LEN_META,
                                   splits_dir=SPLITS_DIR) # <-- AGGIUNGI QUESTO
    except Exception as e:
        logging.error(f"Errore creazione Dataset: {e}", exc_info=True)
        sys.exit(1)

    if len(train_dataset) == 0:
         logging.error("Dataset di training vuoto. Controllare file JSONL e filtri.")
         sys.exit(1)
    if len(val_dataset) == 0:
         logging.warning("Dataset di validazione vuoto. Controllare file JSONL e filtri.")
         # Potrebbe essere accettabile non avere un val set, ma di solito non è voluto.

    collate_fn_with_padding_ids = partial(pad_collate_fn, meta_pad_id=META_PAD_ID, midi_pad_id=MIDI_PAD_ID)
    
    # Imposta il numero di workers in base al sistema operativo.
    if os.name == 'nt':  # Se il sistema operativo è Windows
        # Su Windows, un numero più basso di worker è spesso più efficiente per limitare
        # l'overhead del metodo 'spawn'. 4 è un buon compromesso per un sistema con molti core.
        num_dataloader_workers = 4
        logging.info(f"Sistema operativo Windows rilevato. Imposto un numero conservativo di workers: {num_dataloader_workers}")
    else:
        # Su Linux e macOS, possiamo usare tutti i core disponibili in modo più efficiente.
        try:
            num_dataloader_workers = os.cpu_count()
            logging.info(f"Sistema operativo non-Windows rilevato. Utilizzo tutti i core disponibili: {num_dataloader_workers}")
        except NotImplementedError:
            num_dataloader_workers = 2
            logging.warning("os.cpu_count() non disponibile. Imposto un valore di default: 2")

    # Assicuriamoci che il valore non sia None o zero, se non esplicitamente voluto
    if num_dataloader_workers is None:
        num_dataloader_workers = 0

    logging.info(f"Utilizzo di {num_dataloader_workers} worker per i DataLoaders.")

    # La logica `bool(num_dataloader_workers > 0)` gestisce il caso in cui i workers siano 0.
    # persistent_workers=True richiede num_workers > 0.
    use_persistent_workers = bool(num_dataloader_workers > 0)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                collate_fn=collate_fn_with_padding_ids, 
                                num_workers=num_dataloader_workers,
                                persistent_workers=use_persistent_workers)
    if len(val_dataset) > 0:
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                collate_fn=collate_fn_with_padding_ids, 
                                num_workers=num_dataloader_workers,
                                persistent_workers=use_persistent_workers)
    else:
        val_dataloader = None # Handle case with no validation data

    logging.info("--- Inizializzazione Modello ---")
    max_pe_len_calculated = max(MAX_SEQ_LEN_MIDI, MAX_SEQ_LEN_META) + 100 

    model = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD,
        META_VOCAB_SIZE, MIDI_VOCAB_SIZE, 
        max_pe_len=max_pe_len_calculated, 
        dim_feedforward=FFN_HID_DIM, dropout=DROPOUT
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=MIDI_PAD_ID) # MIDI_PAD_ID from midi_tokenizer
    logging.info(f"Numero parametri: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logging.info(f"Device: {DEVICE}")
    
    model_params_to_save = {
        'num_encoder_layers': NUM_ENCODER_LAYERS, 'num_decoder_layers': NUM_DECODER_LAYERS,
        'emb_size': EMB_SIZE, 'nhead': NHEAD, 'src_vocab_size': META_VOCAB_SIZE,
        'tgt_vocab_size': MIDI_VOCAB_SIZE, 'dim_feedforward': FFN_HID_DIM, 'dropout': DROPOUT,
        'max_pe_len': max_pe_len_calculated
    }
    vocab_info_to_save = {
        'midi_vocab_path': str(VOCAB_PATH), 'metadata_vocab_path': str(METADATA_VOCAB_PATH),
        'midi_pad_id': MIDI_PAD_ID, 'meta_pad_id': META_PAD_ID,
        'MAX_SEQ_LEN_MIDI': MAX_SEQ_LEN_MIDI, 'MAX_SEQ_LEN_META': MAX_SEQ_LEN_META,
        'midi_tokenizer_strategy': MIDI_TOKENIZER_STRATEGY.__name__ # Store strategy name
    }

    logging.info("--- Inizio Addestramento ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10 # Increased patience
    last_saved_epoch = 0 

    for epoch in range(1, EPOCHS + 1):
        logging.info(f"--- Epoch {epoch}/{EPOCHS} ---")
        start_time_epoch = time.time()
        train_loss_epoch = train_epoch(model, optimizer, criterion, train_dataloader)
        
        current_val_loss = float('inf') # Default if no val_dataloader
        if val_dataloader:
            current_val_loss = evaluate(model, criterion, val_dataloader)
        
        epoch_duration_total = time.time() - start_time_epoch
        logging.info(f"Epoch {epoch}: Train Loss = {train_loss_epoch:.4f}, Val Loss = {current_val_loss:.4f} (Durata: {epoch_duration_total:.2f}s)")

        timestamp = time.strftime("%Y%m%d-%H%M%S") # For unique filenames
        
        # Checkpoint saving logic
        is_best_model = current_val_loss < best_val_loss
        if is_best_model and val_dataloader: # Only save best if val_dataloader exists and loss improved
            best_val_loss = current_val_loss
            epochs_no_improve = 0
            best_model_filename = f"transformer_best_epoch{epoch}_valloss{best_val_loss:.4f}_{timestamp}.pt"
            best_model_path = MODEL_SAVE_DIR / best_model_filename
            logging.info(f"Nuova best validation loss: {best_val_loss:.4f}. Salvataggio modello in {best_model_path}")
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'current_val_loss': current_val_loss, 'best_val_loss': best_val_loss,
                'model_params': model_params_to_save, 'vocab_info': vocab_info_to_save
            }, best_model_path)
            last_saved_epoch = epoch # Also counts as a save
        elif val_dataloader: # If val_dataloader exists but no improvement
            epochs_no_improve += 1
            logging.info(f"Validation loss non migliorata ({current_val_loss:.4f} vs best {best_val_loss:.4f}). Epoche senza miglioramento: {epochs_no_improve}/{patience}")
        
        # Periodic saving every 10 epochs, regardless of improvement (if not just saved as best)
        if epoch % 10 == 0 and epoch != last_saved_epoch:
            periodic_model_filename = f"transformer_periodic_epoch{epoch}_valloss{current_val_loss:.4f}_{timestamp}.pt"
            periodic_model_path = MODEL_SAVE_DIR / periodic_model_filename
            logging.info(f"Salvataggio checkpoint periodico (epoch {epoch}): {periodic_model_path}")
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'current_val_loss': current_val_loss, 'best_val_loss': best_val_loss, 
                'model_params': model_params_to_save, 'vocab_info': vocab_info_to_save
            }, periodic_model_path)
            last_saved_epoch = epoch
        
        # Early stopping if val_dataloader exists
        if val_dataloader and epochs_no_improve >= patience:
            logging.info(f"Nessun miglioramento per {patience} epoche consecutive. Early stopping.")
            break 
        
        # If no val_dataloader, save periodically or at the end
        if not val_dataloader and epoch == EPOCHS: # Save last model if no validation
             # This block might be redundant if periodic save already caught it.
             pass


    # --- Salvataggio del modello finale ---
    # Ensure the very last state is saved if it wasn't captured by periodic or best model save.
    if epoch != last_saved_epoch : 
        final_timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Use 'current_val_loss' which holds the last computed validation loss, or train_loss_epoch if no validation
        loss_metric_for_filename = current_val_loss if val_dataloader else train_loss_epoch
        final_model_filename = f"transformer_final_epoch{epoch}_loss{loss_metric_for_filename:.4f}_{final_timestamp}.pt"
        final_model_path = MODEL_SAVE_DIR / final_model_filename
        logging.info(f"Salvataggio checkpoint finale (dopo epoch {epoch}): {final_model_path}")
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'current_val_loss': current_val_loss, # Last val loss or inf
            'best_val_loss': best_val_loss,     # Best overall val loss or inf
            'final_train_loss': train_loss_epoch, # Include final train loss
            'model_params': model_params_to_save, 'vocab_info': vocab_info_to_save
        }, final_model_path)
    else:
        logging.info(f"Lo stato finale del modello (epoch {epoch}) è già stato salvato come checkpoint periodico o migliore.")

    logging.info("--- Addestramento Terminato ---")
    logging.info("--- Script Terminato ---")