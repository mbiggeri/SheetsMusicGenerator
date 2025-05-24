import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
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
MIDI_VOCAB_TARGET_SIZE = 30000 # Esempio: Dimensione target per il vocabolario MIDI se addestrato

VOCAB_PATH = DATA_DIR / "midi_vocab.json" # Dove salvare/caricare il vocabolario MIDI
METADATA_VOCAB_PATH = DATA_DIR / "metadata_vocab.json" # Vocabolario per i token metadati

# Token Speciali MIDI (allineati con le convenzioni di miditok)
MIDI_PAD_TOKEN_NAME = "PAD_None"  # Usato internamente da miditok (es. tokenizer.MIDI_MIDI_PAD_TOKEN_NAME_id)
MIDI_SOS_TOKEN_NAME = "SOS_None"  # miditok.constants.BOS_TOKEN_NAME
MIDI_EOS_TOKEN_NAME = "EOS_None"  # miditok.constants.MIDI_EOS_TOKEN_NAME_NAME
MIDI_UNK_TOKEN_NAME = "UNK_None"  # Convenzione comune, verifica se CPWord/miditok ha un default specifico

# Token Speciali per Metadati (possono rimanere custom)
META_PAD_TOKEN_NAME = "<pad_meta>" # Rendi esplicito che è per i metadati se diverso dal PAD MIDI
META_UNK_TOKEN_NAME = "<unk_meta>"
META_SOS_TOKEN_NAME = "<sos_meta>"
META_EOS_TOKEN_NAME = "<eos_meta>"

# Iperparametri del Modello e Addestramento (Esempi!)
EPOCHS = 2
BATCH_SIZE = 64 # Riduci se hai poca memoria GPU
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
# General MIDI: 0-7 (Acoustic Grand, Bright Acoustic, Electric Grand, Honky-tonk, Electric Piano 1 & 2, Harpsichord, Clavinet)
PIANO_PROGRAMS = list(range(0, 8))

# --- NUOVI IPERPARAMETRI PER MODALITÀ DI PROCESSSAMENTO ---
# Scegli la modalità: "multi_instrument_stream" o "piano_only"
PROCESSING_MODE = "piano_only"  # CAMBIA QUI PER SELEZIONARE LA MODALITÀ
# PROCESSING_MODE = "multi_instrument_stream"

# Setup Logging (opzionale ma utile)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#------------------------
# Tokenizer e Vocabolario
#------------------------

def build_or_load_tokenizer(midi_file_paths=None, force_build=False): # Aggiunto midi_file_paths
    """Costruisce o carica il tokenizer MIDI e la sua configurazione/vocabolario."""
    special_tokens = [MIDI_PAD_TOKEN_NAME, MIDI_SOS_TOKEN_NAME, MIDI_EOS_TOKEN_NAME, MIDI_UNK_TOKEN_NAME]

    if VOCAB_PATH.exists() and not force_build:
        logging.info(f"Caricamento configurazione tokenizer MIDI da {VOCAB_PATH}")
        try:
            tokenizer = MIDI_TOKENIZER_STRATEGY(params=str(VOCAB_PATH)) # Usa str() per Path
            logging.info(f"Tokenizer caricato con successo da {VOCAB_PATH}")
        except Exception as e:
             logging.error(f"Errore nel caricare parametri tokenizer da {VOCAB_PATH}. Errore: {e}", exc_info=True)
             logging.info("Tentativo di ricostruire il tokenizer da zero.")
             # Passa midi_file_paths anche nella chiamata ricorsiva
             return build_or_load_tokenizer(midi_file_paths=midi_file_paths, force_build=True)
    else:
        logging.info("Creazione nuova configurazione tokenizer MIDI...")
        tokenizer_config_special_tokens = [MIDI_PAD_TOKEN_NAME, MIDI_SOS_TOKEN_NAME, MIDI_EOS_TOKEN_NAME, MIDI_UNK_TOKEN_NAME]
        
        tokenizer_params = miditok.TokenizerConfig(
        special_tokens=tokenizer_config_special_tokens,
        use_programs=True,  # Sempre True: necessario per identificare i programmi (pianoforte o altri)
        one_token_stream_for_programs=True, # Sempre True: vogliamo che il tokenizer produca un singolo stream
                                            # dopo il nostro pre-processing per la modalità "piano_only".
        program_changes=True, # Consigliato: inserisce token Program_X. 
                              # Utile per multi-instrument, innocuo per piano-only (userà Program_0, ecc.)
        # Altri parametri che potresti voler impostare, es:
        # beat_res = {(0, 4): 8, (4, 12): 4} # Esempio di risoluzione del beat
        # num_velocities = 32
        # use_chords = True (se CPWord li supporta e li vuoi)
        # use_rests = True
        # use_tempos = True
        # use_time_signatures = True
        )
        
        tokenizer = MIDI_TOKENIZER_STRATEGY(tokenizer_config=tokenizer_params)
        logging.info(f"Tokenizer {MIDI_TOKENIZER_STRATEGY.__name__} inizializzato con use_programs=True, one_token_stream_for_programs=True.")

        # --- ADDESTRAMENTO DEL TOKENIZER (SE APPLICABILE A CPWord) ---
        # Controlla la documentazione di MidiTok per CPWord per vedere se .train() o .learn_bpe() sono necessari/disponibili.
        # Se CPWord costruisce il vocabolario diversamente, questa sezione va adattata/rimossa.
        if midi_file_paths:
            logging.info(f"Verifica se il tokenizer {MIDI_TOKENIZER_STRATEGY.__name__} necessita di addestramento...")
            if hasattr(tokenizer, 'train'):
                logging.info(f"Addestramento tokenizer ({MIDI_TOKENIZER_STRATEGY.__name__}) su {len(midi_file_paths)} file. Target vocab size: {MIDI_VOCAB_TARGET_SIZE}")
                tokenizer.train(vocab_size=MIDI_VOCAB_TARGET_SIZE, files_paths=midi_file_paths)
                logging.info("Addestramento tokenizer completato.")
            elif hasattr(tokenizer, 'learn_bpe'): # Alcuni tokenizer usano .learn_bpe()
                logging.info(f"Apprendimento BPE per tokenizer ({MIDI_TOKENIZER_STRATEGY.__name__}) su {len(midi_file_paths)} file. Target vocab size: {MIDI_VOCAB_TARGET_SIZE}")
                tokenizer.learn_bpe(vocab_size=MIDI_VOCAB_TARGET_SIZE, files_paths=midi_file_paths)
                logging.info("Apprendimento BPE completato.")
            else:
                logging.warning(f"Il tokenizer {MIDI_TOKENIZER_STRATEGY.__name__} non ha un metodo .train() o .learn_bpe(). "
                                "Il vocabolario sarà basato sulla configurazione iniziale e sui token speciali. "
                                "Verifica la documentazione di miditok per CPWord su come il vocabolario viene costruito/gestito.")
        else:
            logging.warning("Nessun file MIDI fornito per l'addestramento del tokenizer durante la costruzione. "
                            "Il vocabolario potrebbe essere subottimale se l'addestramento è necessario.")
        # --- FINE ADDESTRAMENTO ---

        logging.info(f"Salvataggio configurazione tokenizer MIDI in {VOCAB_PATH}")
        VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            tokenizer.save(str(VOCAB_PATH)) # Usa str() per Path
        except AttributeError:
             # Fallback a save_params se save non esiste (versioni più vecchie di miditok?)
             logging.warning("Metodo tokenizer.save() non trovato, provo con tokenizer.save_params()")
             try:
                 tokenizer.save_params(str(VOCAB_PATH)) # Usa str() per Path
             except AttributeError:
                 logging.error("Impossibile salvare la configurazione del tokenizer. Nessun metodo save() o save_params() trovato.")

    logging.info(f"Dimensione vocabolario MIDI (dopo caricamento/costruzione): {len(tokenizer)}")

    # Verifica ID token speciali (come prima, ma ora dopo potenziale addestramento)
    missing_ids_info = {}
    try:
        pad_id_check = tokenizer[MIDI_PAD_TOKEN_NAME]
        sos_id_check = tokenizer[MIDI_SOS_TOKEN_NAME]
        eos_id_check = tokenizer[MIDI_EOS_TOKEN_NAME]
        unk_id_check = tokenizer[MIDI_UNK_TOKEN_NAME]
        logging.info(f"ID Token Speciali MIDI recuperati - PAD: {pad_id_check}, SOS: {sos_id_check}, EOS: {eos_id_check}, UNK: {unk_id_check}")

        if pad_id_check is None: missing_ids_info[MIDI_PAD_TOKEN_NAME] = "Non trovato (None)"
        if sos_id_check is None: missing_ids_info[MIDI_SOS_TOKEN_NAME] = "Non trovato (None)"
        if eos_id_check is None: missing_ids_info[MIDI_EOS_TOKEN_NAME] = "Non trovato (None)"
        if unk_id_check is None: missing_ids_info[MIDI_UNK_TOKEN_NAME] = "Non trovato (None)"

    except Exception as e:
        logging.error(f"Errore durante l'accesso agli ID dei token speciali MIDI tramite token_to_id: {e}", exc_info=True)
        missing_ids_info["ERRORE GENERALE RECUPERO ID MIDI"] = str(e)

    if missing_ids_info:
         logging.error(f"ERRORE CRITICO: Impossibile recuperare gli ID per i seguenti token speciali MIDI: {missing_ids_info}")
         logging.error("Controlla la configurazione del tokenizer, il processo di addestramento (se applicabile) e il file di vocabolario salvato.")
         sys.exit(1) # Esce se gli ID essenziali mancano

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
        required_specials = [META_PAD_TOKEN_NAME, META_UNK_TOKEN_NAME, META_SOS_TOKEN_NAME, META_EOS_TOKEN_NAME]
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
        all_tokens = [META_PAD_TOKEN_NAME, META_UNK_TOKEN_NAME, META_SOS_TOKEN_NAME, META_EOS_TOKEN_NAME] + sorted(list(metadata_tokens))        
        token_to_id = {token: i for i, token in enumerate(all_tokens)}
        id_to_token = {i: token for token, i in token_to_id.items()}

        # Verifica ID token speciali (PAD dovrebbe essere 0 se CrossEntropyLoss usa ignore_index=0)
        if token_to_id[META_PAD_TOKEN_NAME] != 0:
            logging.warning(f"ID del MIDI_MIDI_PAD_TOKEN_NAME ({META_PAD_TOKEN_NAME}) non è 0 ({token_to_id[META_PAD_TOKEN_NAME]}). "
                            "Potrebbe causare problemi con CrossEntropyLoss(ignore_index=0). Riassegno gli ID.")
            # Riassegna mettendo PAD a 0
            pad_tok = META_PAD_TOKEN_NAME
            other_specials = [t for t in [META_UNK_TOKEN_NAME, META_SOS_TOKEN_NAME, META_EOS_TOKEN_NAME] if t in token_to_id]
            actual_tokens = sorted(list(metadata_tokens))
            # Ordine: PAD, altri speciali, token reali
            all_tokens_reordered = [pad_tok] + other_specials + actual_tokens
            token_to_id = {token: i for i, token in enumerate(all_tokens_reordered)}
            # Aggiungi UNK se non presente nei dati originali ma definito
            if MIDI_UNK_TOKEN_NAME not in token_to_id:
                 unk_id = len(token_to_id)
                 token_to_id[MIDI_UNK_TOKEN_NAME] = unk_id
                 logging.info(f"Aggiunto MIDI_UNK_TOKEN_NAME con ID {unk_id}")

            id_to_token = {i: token for token, i in token_to_id.items()}
            logging.info(f"Nuovo ID MIDI_MIDI_PAD_TOKEN_NAME: {token_to_id[MIDI_MIDI_PAD_TOKEN_NAME]}")


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
            # Token Metadati
            self.meta_pad_id = metadata_vocab_map[META_PAD_TOKEN_NAME]
            self.sos_meta_id = metadata_vocab_map[META_SOS_TOKEN_NAME]
            self.eos_meta_id = metadata_vocab_map[META_EOS_TOKEN_NAME]
            self.unk_meta_id = metadata_vocab_map[META_UNK_TOKEN_NAME]

            # Token MIDI
            self.sos_midi_id = midi_tokenizer[MIDI_SOS_TOKEN_NAME]
            self.eos_midi_id = midi_tokenizer[MIDI_EOS_TOKEN_NAME]
            self.midi_pad_id = midi_tokenizer[MIDI_PAD_TOKEN_NAME]

            # Verifica che tutti gli ID necessari siano stati recuperati
            if None in [self.sos_midi_id, self.eos_midi_id, self.midi_pad_id,
                        self.meta_pad_id, self.sos_meta_id, self.eos_meta_id, self.unk_meta_id]:
                raise ValueError("Uno o più ID di token speciali non sono stati trovati nei rispettivi vocabolari.")

        except KeyError as e:
            logging.error(f"ERRORE CRITICO in Dataset __init__: Token speciale '{e}' non trovato nel vocabolario.")
            raise
        except ValueError as e:
            logging.error(f"ERRORE CRITICO in Dataset __init__: Problema con gli ID dei token speciali: {e}")
            raise

        logging.info(f"Caricamento dati da {jsonl_path} per Dataset...")
        self.data = []
        skipped_missing_midi_file = 0
        skipped_key_filter = 0
        skipped_no_relative_path = 0
        skipped_json_decode = 0
        skipped_other_entry_error = 0

        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        entry = json.loads(line)
                        
                        current_metadata = entry.get('metadata', {}) # Ottieni il dizionario metadata
                        
                        if self.filter_key and current_metadata.get('key') != self.filter_key:
                            skipped_key_filter += 1
                            continue

                        # --- CORREZIONE ACCESSO MIDI RELATIVE PATH ---
                        midi_relative_path_str = current_metadata.get('midi_relative_path')
                        if not midi_relative_path_str:
                            skipped_no_relative_path += 1
                            # logging.debug(f"Riga {line_num+1}: 'midi_relative_path' non trovato in metadata. Salto.")
                            continue
                        
                        midi_path_check = self.midi_base_dir / midi_relative_path_str
                        if midi_path_check.exists() and midi_path_check.is_file():
                            self.data.append(entry) # Aggiungi l'intera entry JSON se il file MIDI esiste
                        else:
                            skipped_missing_midi_file += 1
                            # logging.debug(f"Riga {line_num+1}: File MIDI non trovato o non è un file: {midi_path_check}. Salto.")
                            
                    except json.JSONDecodeError:
                        # logging.warning(f"Riga JSON malformata in {jsonl_path} (linea ~{line_num+1}), salto.")
                        skipped_json_decode += 1
                    except Exception as e:
                         # logging.warning(f"Errore caricando riga {line_num+1} del dataset: {e}")
                         skipped_other_entry_error += 1
        except FileNotFoundError:
             logging.error(f"File dataset non trovato: {jsonl_path}")
             raise

        logging.info(f"Caricati {len(self.data)} campioni validi da {jsonl_path}.")
        if self.filter_key: logging.info(f"   Campioni saltati per filtro chiave '{self.filter_key}': {skipped_key_filter}")
        logging.info(f"   Campioni saltati per 'midi_relative_path' mancante in metadata: {skipped_no_relative_path}")
        logging.info(f"   Campioni saltati per file MIDI non esistente su disco: {skipped_missing_midi_file}")
        logging.info(f"   Campioni saltati per JSON malformato: {skipped_json_decode}")
        logging.info(f"   Campioni saltati per altri errori di entry: {skipped_other_entry_error}")
        if not self.data:
            logging.error(f"Nessun dato caricato da {jsonl_path}. Verifica il file, i percorsi e i filtri.")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        max_retries = 2
        entry = self.data[idx]
        actual_metadata = entry.get('metadata', {})
        midi_relative_path_str = actual_metadata.get('midi_relative_path')

        if not midi_relative_path_str:
            logging.error(f"__getitem__ idx {idx}: 'midi_relative_path' mancante in metadata. Entry: {entry}")
            return None

        midi_full_path = self.midi_base_dir / midi_relative_path_str

        for attempt in range(max_retries):
            try:
                # 1. Carica il file MIDI in un oggetto Score di miditok
                # miditok.Score può caricare direttamente da un percorso
                try:
                    score = Score(str(midi_full_path))
                except Exception as e:
                    logging.warning(f"Tentativo {attempt+1}: Errore caricando MIDI {midi_full_path} con miditok.Score: {e}")
                    if attempt + 1 == max_retries:
                        return None # Salta campione dopo tentativi falliti
                    time.sleep(0.1) # Breve attesa prima del ritentativo
                    continue 

                # --- INIZIO LOGICA SPECIFICA PER MODALITÀ ---
                if PROCESSING_MODE == "piano_only":
                    piano_tracks_presenti = [track for track in score.tracks if track.program in PIANO_PROGRAMS]
                    
                    if not piano_tracks_presenti:
                        # logging.debug(f"Nessuna traccia di pianoforte trovata in {midi_full_path} (idx {idx}). Salto campione.")
                        return None # Salta se non ci sono tracce di pianoforte

                    # Sovrascrivi le tracce dello score solo con quelle di pianoforte
                    score.tracks = piano_tracks_presenti
                    
                    # Se ci sono più tracce di pianoforte, miditok le unirà automaticamente 
                    # in un unico flusso se la configurazione del tokenizer ha 
                    # one_token_stream_for_programs=True e use_programs=True (come abbiamo impostato).
                    # Se vuoi un controllo più fine sulla fusione (es. rinominare la traccia fusa),
                    # potresti usare miditok.utils.merge_tracks qui. Per ora, lasciamo che sia il tokenizer.
                    if len(score.tracks) == 0: # Doppio controllo dopo il filtro
                         # logging.debug(f"Score senza tracce dopo filtro pianoforte per {midi_full_path} (idx {idx}). Salto campione.")
                         return None
                
                # Per PROCESSING_MODE == "multi_instrument_stream", non facciamo nulla qui,
                # il tokenizer elaborerà tutte le tracce presenti nello Score originale
                # (dopo il suo preprocess_score interno).
                # --- FINE LOGICA SPECIFICA PER MODALITÀ ---

                # 2. Tokenizza Metadati (Source Sequence) - invariato
                meta_tokens_str = tokenize_metadata(actual_metadata)
                meta_token_ids = [self.metadata_vocab_map.get(token, self.unk_meta_id) for token in meta_tokens_str]
                src_seq = [self.sos_meta_id] + meta_token_ids[:self.max_len_meta-2] + [self.eos_meta_id]

                # 3. Tokenizza MIDI (Target Sequence)
                # Passiamo l'oggetto Score (potenzialmente modificato) al tokenizer.
                # Poiché tokenizer.one_token_stream è True (grazie alla config), 
                # midi_tokens_output sarà un singolo oggetto TokSequence.
                midi_tokens_output = self.midi_tokenizer(score) # Passa l'oggetto Score

                if not hasattr(midi_tokens_output, 'ids'):
                    logging.warning(f"L'output del tokenizer MIDI non ha l'attributo .ids per {midi_full_path} (idx {idx}) dopo aver processato l'oggetto Score.")
                    raise RuntimeError("Output tokenizer MIDI malformato o TokSequence non valido")
                
                raw_midi_ids = midi_tokens_output.ids

                if raw_midi_ids is None or not isinstance(raw_midi_ids, list): # Controllo aggiuntivo
                    logging.warning(f"raw_midi_ids non è una lista o è None per {midi_full_path} (idx {idx})")
                    raise RuntimeError("raw_midi_ids non validi")

                # La logica per processare raw_midi_ids (inclusa la gestione di CPWord multi-vocabolario
                # e l'appiattimento se raw_midi_ids[0] è una lista) rimane la stessa di prima.
                processed_midi_ids = []
                if len(raw_midi_ids) > 0:
                    if isinstance(raw_midi_ids[0], list): 
                        temp_flattened_ids = []
                        for component_group in raw_midi_ids:
                            if isinstance(component_group, list):
                                temp_flattened_ids.extend(component_group)
                            else: 
                                temp_flattened_ids.append(component_group) 
                        processed_midi_ids = temp_flattened_ids
                        if not processed_midi_ids and len(raw_midi_ids) > 0: # Se raw_midi_ids non era vuoto ma l'appiattimento lo è
                            logging.warning(f"Appiattimento ID multi-voc ha prodotto sequenza vuota per {midi_full_path} (idx {idx}). raw_midi_ids originale: {raw_midi_ids}")
                            # Non sollevare eccezione qui, potrebbe essere un MIDI valido ma corto/strano per CPWord.
                            # Lascia che venga gestito dal controllo di lunghezza sotto.
                    elif isinstance(raw_midi_ids[0], int): 
                        processed_midi_ids = raw_midi_ids
                    else:
                        logging.warning(f"Formato ID MIDI inatteso (raw_midi_ids[0] non è né lista né int) per {midi_full_path} (idx {idx}). Tipo: {type(raw_midi_ids[0])}. Salto campione.")
                        raise RuntimeError("Formato ID MIDI inatteso all'interno della lista di ID.")
                # Se raw_midi_ids era inizialmente vuoto, processed_midi_ids rimarrà vuoto.
                
                if not processed_midi_ids: # Se, dopo tutto, non abbiamo ID MIDI (es. MIDI vuoto, filtro pianoforte non ha trovato nulla, o CPWord ha dato output vuoto)
                    # logging.debug(f"Nessun ID MIDI processabile per {midi_full_path} (idx {idx}). Salto campione.")
                    return None # Salta il campione

                tgt_seq = [self.sos_midi_id] + processed_midi_ids[:self.max_len_midi-2] + [self.eos_midi_id]
                return torch.tensor(src_seq, dtype=torch.long), torch.tensor(tgt_seq, dtype=torch.long)

            except FileNotFoundError:
                logging.error(f"Tentativo {attempt+1}/{max_retries} fallito per idx {idx} ({midi_full_path}): File MIDI non trovato durante __getitem__.")
                if attempt + 1 == max_retries: return None # Sarà filtrato
            except RuntimeError as e: # Per errori di tokenizzazione o tracce vuote gestiti sopra
                logging.warning(f"Tentativo {attempt+1}/{max_retries} (RuntimeError) fallito per idx {idx} ({midi_full_path}): {e}")
                if attempt + 1 == max_retries: return None # Sarà filtrato
            except Exception as e:
                logging.warning(f"Tentativo {attempt+1}/{max_retries} fallito per idx {idx} ({midi_full_path}): {type(e).__name__} - {e}", exc_info=False)
                if attempt + 1 == max_retries:
                    logging.error(f"Errore definitivo processando idx {idx} ({midi_full_path}): {e}. Ritorno None.", exc_info=True)
                    return None # Sarà filtrato
        return None # Se tutti i tentativi falliscono

def pad_collate_fn(batch, meta_pad_id, midi_pad_id):
    """
    Collate function per DataLoader. Esegue il padding, crea le maschere
    e filtra eventuali elementi None restituiti dal Dataset.
    Usa meta_pad_id per sorgenti (metadati) e midi_pad_id per target (MIDI).
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None, None

    src_batch, tgt_batch = zip(*batch)

    # Padding delle sequenze sorgente (metadati) con meta_pad_id
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=meta_pad_id)
    # Padding delle sequenze target (MIDI) con midi_pad_id
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=midi_pad_id)

    # Creazione maschere di padding (True dove c'è padding)
    src_padding_mask = (src_padded == meta_pad_id)
    tgt_padding_mask = (tgt_padded == midi_pad_id)

    return src_padded, tgt_padded, src_padding_mask, tgt_padding_mask

#------------------------
# Modello Transformer (invariato)
#------------------------

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
             tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool), diagonal=1)
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

#------------------------
# Ciclo di Addestramento e Valutazione
#------------------------

def train_epoch(model, optimizer, criterion, train_dataloader): # Rimosso pad_id
    model.train()
    total_loss = 0
    processed_batches = 0
    progress_bar = tqdm(train_dataloader, desc="Training Epoch", leave=False)

    for batch_data in progress_bar:
        if batch_data[0] is None: # Batch vuoto dopo il filtraggio in collate_fn
            logging.debug("Batch vuoto ricevuto dal dataloader (train), salto.")
            continue
        # ... (resto della logica di training è ok, la criterion usa già MIDI_PAD_ID corretto) ...
        src, tgt, src_padding_mask, tgt_padding_mask = batch_data
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src_padding_mask = src_padding_mask.to(DEVICE)
        tgt_padding_mask = tgt_padding_mask.to(DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_input_padding_mask = tgt_padding_mask[:, :-1] # Maschera per decoder self-attention
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src=src,
                       tgt=tgt_input,
                       src_padding_mask=src_padding_mask, # Maschera per encoder self-attention & memory
                       tgt_padding_mask=tgt_input_padding_mask, # Maschera per decoder self-attention
                       memory_key_padding_mask=src_padding_mask) # Maschera per encoder-decoder attention (dalla sorgente)
        
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        processed_batches += 1
        progress_bar.set_postfix({'train_loss': f'{total_loss / processed_batches:.4f}'})

    if processed_batches == 0:
        logging.warning("Nessun batch processato nell'epoca di training.")
        return 0.0
    return total_loss / processed_batches

def evaluate(model, criterion, dataloader): # Rimosso pad_id
    model.eval()
    total_loss = 0
    processed_batches = 0
    progress_bar = tqdm(dataloader, desc="Evaluation", leave=False)

    with torch.no_grad():
        for batch_data in progress_bar:
            if batch_data[0] is None: # Batch vuoto
                logging.debug("Batch vuoto ricevuto dal dataloader (eval), salto.")
                continue
            # ... (resto della logica di evaluazione è ok) ...
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
            progress_bar.set_postfix({'eval_loss': f'{total_loss / processed_batches:.4f}'})
            
    if processed_batches == 0:
        logging.warning("Nessun batch processato durante la valutazione.")
        return float('inf')
    return total_loss / processed_batches

#------------------------
# Funzione di Generazione
#------------------------

def generate_sequence(model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
                      max_len=500, temperature=0.7, top_k=None, device=DEVICE):
    model.eval()

    try:
        sos_meta_id = metadata_vocab_map[META_SOS_TOKEN_NAME]
        eos_meta_id = metadata_vocab_map[META_EOS_TOKEN_NAME]
        unk_meta_id = metadata_vocab_map[META_UNK_TOKEN_NAME]
        meta_pad_id = metadata_vocab_map[META_PAD_TOKEN_NAME] # Per src_padding_mask

        sos_midi_id = midi_tokenizer[MIDI_SOS_TOKEN_NAME]
        eos_midi_id = midi_tokenizer[MIDI_EOS_TOKEN_NAME]
        # Non hai bisogno di MIDI_PAD_ID qui a meno che tu non lo usi esplicitamente nella generazione

        if None in [sos_meta_id, eos_meta_id, unk_meta_id, meta_pad_id, sos_midi_id, eos_midi_id]:
            raise ValueError("Uno o più ID di token speciali non trovati nei vocabolari per la generazione.")
    except (KeyError, ValueError) as e:
        logging.error(f"Errore critico nel recuperare ID token speciali in generate_sequence: {e}")
        return [] # Ritorna lista vuota in caso di errore setup

    # 1. Prepara input metadati
    meta_token_ids = [metadata_vocab_map.get(token, unk_meta_id) for token in metadata_prompt]
    src_seq = torch.tensor([[sos_meta_id] + meta_token_ids[:MAX_SEQ_LEN_META-2] + [eos_meta_id]], dtype=torch.long, device=device)
    src_padding_mask = (src_seq == meta_pad_id) # Usa meta_pad_id per la maschera sorgente

    with torch.no_grad():
        # 2. Codifica i metadati
        memory = model.encode(src_seq, src_padding_mask)
        memory_key_padding_mask = src_padding_mask

        # 3. Inizia generazione decoder
        tgt_tokens = torch.tensor([[sos_midi_id]], dtype=torch.long, device=device)

        for i in range(max_len - 1): # Genera fino a max_len-1 nuovi token
            tgt_len = tgt_tokens.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)
            # Durante la generazione, tgt_padding_mask è solitamente tutti False
            tgt_padding_mask_step = torch.zeros_like(tgt_tokens, dtype=torch.bool, device=device)

            decoder_output = model.decode(tgt=tgt_tokens, memory=memory, tgt_mask=tgt_mask,
                                          tgt_padding_mask=tgt_padding_mask_step, # Maschera per input decoder (qui tutto non-paddato)
                                          memory_key_padding_mask=memory_key_padding_mask) # Maschera per l'attenzione alla memoria

            logits = model.generator(decoder_output)
            last_logits = logits[:, -1, :] # Prendi solo l'output per l'ultimo token

            if temperature > 0:
                last_logits = last_logits / temperature
            
            if top_k is not None and top_k > 0:
                # Sostituisci i valori non top-k con -inf per escluderli da softmax
                indices_to_remove = last_logits < torch.topk(last_logits, min(top_k, last_logits.size(-1)))[0][..., -1, None]
                last_logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(last_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            if next_token_id.item() == eos_midi_id:
                logging.info(f"Token EOS generato dopo {i+1} step.")
                break
            
            tgt_tokens = torch.cat((tgt_tokens, next_token_id), dim=1)
            if tgt_tokens.size(1) >= max_len:
                logging.info(f"Raggiunta lunghezza massima di generazione ({max_len}).")
                break
    
    # Ritorna la sequenza generata (escludendo SOS iniziale)
    return tgt_tokens[0, 1:].tolist()

#------------------------ 
# Esecuzione Principale
#------------------------

if __name__ == "__main__":
    # --- 1. Prepara Tokenizer e Vocabolari ---
    logging.info("--- Preparazione Tokenizer e Vocabolari ---")
    train_jsonl_path = SPLITS_DIR / "train.jsonl"
    all_train_metadata = []
    midi_files_for_vocab_build = [] # Lista di path stringa ai file MIDI

    # --- Blocco per popolare midi_files_for_vocab_build e all_train_metadata ---
    # (Assicurati che questa logica sia robusta e corretta per la tua struttura JSONL)
    logging.info(f"Inizio lettura {train_jsonl_path} per costruire vocabolari...")
    processed_lines_count_vocab = 0
    found_midi_for_vocab = 0
    try:
        if not train_jsonl_path.is_file():
            raise FileNotFoundError(f"File JSONL di training non trovato: {train_jsonl_path}")
        with open(train_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                processed_lines_count_vocab += 1
                try:
                    entry = json.loads(line)
                    # Accedi a 'midi_relative_path' DENTRO 'metadata'
                    metadata_dict = entry.get('metadata', {})
                    relative_path_str = metadata_dict.get('midi_relative_path')

                    if relative_path_str:
                        midi_full_path_str = str(MIDI_BASE_DIR / relative_path_str) # Converti a stringa
                        if Path(midi_full_path_str).exists() and Path(midi_full_path_str).is_file():
                            found_midi_for_vocab += 1
                            midi_files_for_vocab_build.append(midi_full_path_str)
                            all_train_metadata.append(metadata_dict) # Aggiungi solo se MIDI esiste
                        # else: # Opzionale: log per file non trovati
                        #     if line_num < 10: logging.debug(f"Vocab build: MIDI non trovato {midi_full_path_str}")
                    # else: # Opzionale: log per path mancante
                    #     if line_num < 10: logging.debug(f"Vocab build: 'midi_relative_path' mancante in {entry}")
                except json.JSONDecodeError:
                    logging.warning(f"Vocab build: Riga JSON malformata {line_num+1}, salto.")
                except Exception as e:
                    logging.warning(f"Vocab build: Errore riga {line_num+1}: {e}, salto.")
        logging.info(f"Vocab build: Processate {processed_lines_count_vocab} righe.")
        logging.info(f"Vocab build: Trovati {found_midi_for_vocab} file MIDI validi.")
        logging.info(f"Vocab build: Raccolti {len(all_train_metadata)} record metadati.")
    except FileNotFoundError:
        logging.error(f"ERRORE CRITICO: File di training non trovato: {train_jsonl_path}. Impossibile costruire vocabolari.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"ERRORE CRITICO nell'apertura/lettura di {train_jsonl_path} per vocabolari: {e}", exc_info=True)
        sys.exit(1)

    if not midi_files_for_vocab_build:
        logging.error("ERRORE CRITICO: Nessun file MIDI valido trovato per costruire/addestrare il tokenizer MIDI. Controlla i percorsi e il file JSONL.")
        sys.exit(1)
    if not all_train_metadata: # Può essere un warning se i metadati non sono cruciali per il task base
        logging.warning("ATTENZIONE: Nessun record metadati valido trovato. Il vocabolario dei metadati sarà limitato o vuoto (a parte i token speciali).")

    force_rebuild_vocabs = True # Imposta a True per forzare la ricostruzione       #--- TEMPORANEO, REIMPOSTA A FALSE QUANDO FUNZIONA ---#
    # Passa la lista di file MIDI a build_or_load_tokenizer
    midi_tokenizer = build_or_load_tokenizer(midi_file_paths=midi_files_for_vocab_build, force_build=force_rebuild_vocabs)
    metadata_vocab_map, metadata_id_to_token = build_or_load_metadata_vocab(all_train_metadata, force_build=force_rebuild_vocabs)

    MIDI_VOCAB_SIZE = len(midi_tokenizer)
    META_VOCAB_SIZE = len(metadata_vocab_map)

    # Recupero PAD ID (essenziale per DataLoader e Loss)
    try:
        MIDI_PAD_ID = midi_tokenizer[MIDI_PAD_TOKEN_NAME]
        if MIDI_PAD_ID is None: raise KeyError(f"'{MIDI_PAD_TOKEN_NAME}' restituisce None dal midi_tokenizer.")
    except KeyError as e:
        logging.error(f"ERRORE CRITICO: Token PAD '{MIDI_PAD_TOKEN_NAME}' non trovato nel vocabolario del midi_tokenizer ({e}).")
        sys.exit(1)
    try:
        META_PAD_ID = metadata_vocab_map[META_PAD_TOKEN_NAME]
    except KeyError:
        logging.error(f"ERRORE CRITICO: Token PAD '{META_PAD_TOKEN_NAME}' non trovato nel vocabolario dei metadati.")
        sys.exit(1)

    logging.info(f"Dimensione Vocabolario MIDI: {MIDI_VOCAB_SIZE}, PAD ID MIDI: {MIDI_PAD_ID}")
    logging.info(f"Dimensione Vocabolario Metadati: {META_VOCAB_SIZE}, PAD ID Metadati: {META_PAD_ID}")

    # --- 2. Crea Dataset e DataLoader ---
    logging.info("--- Creazione Dataset e DataLoader ---")
    KEY_FILTER = None # Imposta se vuoi filtrare per chiave
    
    try:
        train_dataset = MutopiaDataset(SPLITS_DIR / "train.jsonl", MIDI_BASE_DIR, midi_tokenizer, metadata_vocab_map,
                                     MAX_SEQ_LEN_MIDI, MAX_SEQ_LEN_META, filter_key=KEY_FILTER)
        val_dataset = MutopiaDataset(SPLITS_DIR / "validation.jsonl", MIDI_BASE_DIR, midi_tokenizer, metadata_vocab_map,
                                   MAX_SEQ_LEN_MIDI, MAX_SEQ_LEN_META, filter_key=KEY_FILTER)
        # test_dataset = MutopiaDataset(SPLITS_DIR / "test.jsonl", ...) # Se necessario
    except FileNotFoundError as e:
        logging.error(f"File .jsonl non trovato durante la creazione dei Dataset: {e}. Interruzione.")
        sys.exit(1)
    except Exception as e: # Cattura più generica per errori in init di Dataset
        logging.error(f"Errore imprevisto durante la creazione dei Dataset: {e}", exc_info=True)
        sys.exit(1)

    if len(train_dataset) == 0:
         logging.error("Dataset di training è vuoto dopo l'inizializzazione. Controllare i log precedenti per errori nel caricamento dei dati, percorsi o filtri.")
         sys.exit(1)
    if len(val_dataset) == 0: # Opzionale, ma buon controllo
         logging.warning("Dataset di validazione è vuoto.")


    # --- USA functools.partial PER PASSARE PAD_ID A pad_collate_fn ---
    collate_fn_with_padding_ids = partial(pad_collate_fn, meta_pad_id=META_PAD_ID, midi_pad_id=MIDI_PAD_ID)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn_with_padding_ids, num_workers=0, # num_workers=0 per debug
                                  pin_memory=(DEVICE.type == 'cuda'))
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn_with_padding_ids, num_workers=0,
                                pin_memory=(DEVICE.type == 'cuda'))
    # test_dataloader = DataLoader(test_dataset, ..., collate_fn=collate_fn_with_padding_ids, ...)

    # --- 3. Inizializza Modello, Ottimizzatore, Loss ---
    logging.info("--- Inizializzazione Modello ---")
    model = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD,
        META_VOCAB_SIZE, MIDI_VOCAB_SIZE, FFN_HID_DIM, DROPOUT
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # La CrossEntropyLoss USA MIDI_PAD_ID per ignorare i token di padding nel target (MIDI)
    logging.info(f"Impostazione CrossEntropyLoss con ignore_index={MIDI_PAD_ID} (per sequenze MIDI target)")
    criterion = nn.CrossEntropyLoss(ignore_index=MIDI_PAD_ID)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Numero totale parametri addestrabili: {total_params:,}")
    logging.info(f"Modello eseguito su: {DEVICE}")

    # --- 4. Ciclo di Addestramento ---
    logging.info("--- Inizio Addestramento ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5
    best_model_path = None

    for epoch in range(1, EPOCHS + 1):
        logging.info(f"--- Epoch {epoch}/{EPOCHS} ---")
        start_time_epoch = time.time()
        
        # Passa solo gli argomenti necessari a train_epoch e evaluate
        train_loss_epoch = train_epoch(model, optimizer, criterion, train_dataloader)
        val_loss_epoch = evaluate(model, criterion, val_dataloader)
        
        epoch_duration_total = time.time() - start_time_epoch
        logging.info(f"Epoch {epoch}: Train Loss = {train_loss_epoch:.4f}, Val Loss = {val_loss_epoch:.4f} (Durata: {epoch_duration_total:.2f}s)")

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            epochs_no_improve = 0
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # Includi informazioni più dettagliate nel nome del file se utile
            model_filename = f"transformer_mutopia_best_epoch{epoch}_valloss{best_val_loss:.4f}_{timestamp}.pt"
            current_best_model_path = MODEL_SAVE_DIR / model_filename
            logging.info(f"Miglioramento validation loss. Salvataggio modello in {current_best_model_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss, # Salva la best_val_loss corrente
                # Parametri del modello per ricrearlo facilmente
                'model_params': {
                    'num_encoder_layers': NUM_ENCODER_LAYERS,
                    'num_decoder_layers': NUM_DECODER_LAYERS,
                    'emb_size': EMB_SIZE,
                    'nhead': NHEAD,
                    'src_vocab_size': META_VOCAB_SIZE,
                    'tgt_vocab_size': MIDI_VOCAB_SIZE,
                    'dim_feedforward': FFN_HID_DIM,
                    'dropout': DROPOUT # Salva anche il dropout usato
                },
                # Info sui vocabolari e PAD ID usati
                'vocab_info': {
                    'midi_vocab_path': str(VOCAB_PATH),
                    'metadata_vocab_path': str(METADATA_VOCAB_PATH),
                    'midi_pad_id': MIDI_PAD_ID,
                    'meta_pad_id': META_PAD_ID,
                }
            }, current_best_model_path)
            best_model_path = current_best_model_path # Aggiorna il path del modello migliore
        else:
            epochs_no_improve += 1
            logging.info(f"Validation loss non migliorata ({val_loss_epoch:.4f} vs best {best_val_loss:.4f}). Epoche senza miglioramento: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            logging.info(f"Nessun miglioramento per {patience} epoche consecutive. Early stopping.")
            break
    
    logging.info("--- Addestramento Terminato ---")

    # --- 5. Valutazione finale sul Test Set (opzionale) ---
    if best_model_path and best_model_path.exists():
        logging.info(f"--- Valutazione su Test Set usando il modello migliore: {best_model_path} ---")
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        
        # Ricrea il modello con i parametri salvati
        model_params_loaded = checkpoint.get('model_params', {}) # Usa get per default vuoto
        if not model_params_loaded: # Fallback ai valori globali se non salvati (vecchio checkpoint)
            logging.warning("Parametri modello non trovati nel checkpoint, uso i valori globali correnti.")
            model_params_loaded = {
                'num_encoder_layers': NUM_ENCODER_LAYERS, 'num_decoder_layers': NUM_DECODER_LAYERS,
                'emb_size': EMB_SIZE, 'nhead': NHEAD,
                'src_vocab_size': META_VOCAB_SIZE, 'tgt_vocab_size': MIDI_VOCAB_SIZE, # Questi potrebbero essere nel checkpoint principale
                'dim_feedforward': FFN_HID_DIM, 'dropout': DROPOUT
            }
        # Sovrascrivi con valori specifici se presenti direttamente nel checkpoint (per compatibilità)
        model_params_loaded['src_vocab_size'] = checkpoint.get('meta_vocab_size', model_params_loaded['src_vocab_size'])
        model_params_loaded['tgt_vocab_size'] = checkpoint.get('midi_vocab_size', model_params_loaded['tgt_vocab_size'])


        loaded_model_for_test = Seq2SeqTransformer(**model_params_loaded).to(DEVICE)
        loaded_model_for_test.load_state_dict(checkpoint['model_state_dict'])
        
        # Crea il test_dataloader se non l'hai già fatto e vuoi valutare
        # test_jsonl_path = SPLITS_DIR / "test.jsonl"
        # if test_jsonl_path.exists():
        #    test_dataset = MutopiaDataset(test_jsonl_path, MIDI_BASE_DIR, midi_tokenizer, metadata_vocab_map,
        #                                MAX_SEQ_LEN_MIDI, MAX_SEQ_LEN_META, filter_key=KEY_FILTER)
        #    if len(test_dataset) > 0:
        #        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        #                                     collate_fn=collate_fn_with_padding_ids, num_workers=0,
        #                                     pin_memory=(DEVICE.type == 'cuda'))
        #        logging.info("Inizio valutazione su Test Set...")
        #        test_loss = evaluate(loaded_model_for_test, criterion, test_dataloader)
        #        logging.info(f"Test Loss Finale (dal modello {best_model_path.name}): {test_loss:.4f}")
        #    else:
        #        logging.warning("Test dataset è vuoto. Salto valutazione su Test Set.")
        # else:
        #    logging.info("File test.jsonl non trovato. Salto valutazione su Test Set.")
        final_model_to_use = loaded_model_for_test # Usa questo per la generazione
    else:
        logging.warning("Nessun modello migliore salvato o path non valido. Salto valutazione su Test Set.")
        logging.info("Usando l'ultimo stato del modello dall'addestramento per la generazione (se applicabile).")
        final_model_to_use = model # Usa l'ultimo modello addestrato

    # --- 6. Generazione di esempio ---
    if final_model_to_use: # Assicurati che ci sia un modello da usare
        logging.info("--- Esempio di Generazione ---")
        example_metadata_prompt = ["Style=Folk", "Key=A_minor", "TimeSig=4/4", "Title=kayo_kami"]
        logging.info(f"Prompt metadati per generazione: {example_metadata_prompt}")

        try:
            generated_token_ids = generate_sequence(
                final_model_to_use, midi_tokenizer, metadata_vocab_map, example_metadata_prompt,
                max_len=MAX_SEQ_LEN_MIDI, temperature=0.75, top_k=40, device=DEVICE
            )

            if generated_token_ids:
                logging.info(f"Generati {len(generated_token_ids)} token MIDI.")
                try:
                    # miditok >= 2.0: tokens_to_midi restituisce un oggetto symusic.Score (o miditoolkit.MidiFile)
                    # Assicurati che generated_token_ids sia una lista di int.
                    # tokens_to_midi si aspetta una lista di sequenze di token (quindi lista di liste di int)
                    # o una singola sequenza di token (lista di int) a seconda della versione/config.
                    # Per una singola sequenza:
                    generated_midi_object = midi_tokenizer.tokens_to_midi([generated_token_ids]) # Passa come lista contenente la sequenza
                    # Se hai una sola sequenza e il tokenizer si aspetta solo quella:
                    # generated_midi_object = midi_tokenizer(generated_token_ids) # Chiamare il tokenizer direttamente
                except Exception as e:
                    logging.error(f"Errore durante midi_tokenizer.tokens_to_midi: {e}", exc_info=True)
                    generated_midi_object = None

                if generated_midi_object:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    # Pulisci il prompt per il nome del file
                    clean_prompt_name = "_".join(token.replace("=", "_").replace("/", "_") for token in example_metadata_prompt)
                    output_filename = GENERATION_OUTPUT_DIR / f"generated_{clean_prompt_name}_{timestamp}.mid"
                    try:
                        # Il metodo .dump() è tipico per oggetti MIDI di miditoolkit o symusic
                        generated_midi_object.dump(str(output_filename)) # Assicura che il path sia una stringa
                        logging.info(f"File MIDI generato salvato in: {output_filename}")
                    except Exception as e:
                        logging.error(f"Errore durante il salvataggio del file MIDI generato ({output_filename}): {e}", exc_info=True)
            else:
                logging.warning("Generazione fallita o ha prodotto una sequenza vuota.")
        except Exception as e:
            logging.error(f"Errore generale durante la fase di generazione: {e}", exc_info=True)
    else:
        logging.warning("Nessun modello disponibile per la generazione.")

    logging.info("--- Script Terminato ---")