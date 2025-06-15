import os
import sys
from contextlib import contextmanager
import json
import random
import math
import logging
from pathlib import Path
import time
from tqdm import tqdm
import concurrent.futures
import mido
import argparse
import warnings
import music21
import numpy as np
from music21 import converter, key, pitch, stream
from music21.exceptions21 import Music21Exception
from music21 import midi as music21_midi
import tempfile
import re
import multiprocessing
import psutil   # for auto-calibrating best number of workers
import collections
from tokenize_metadata import tokenize_metadata
import config
import hashlib

@contextmanager
def suppress_cpp_warnings():
    """
    A context manager to temporarily suppress both stdout and stderr on Windows.
    This is a more aggressive version to silence verbose C++ libraries like symusic.
    """
    # File descriptors for stdout is 1, for stderr is 2
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    # Save a copy of the original file descriptors
    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)
    
    try:
        # Open a file handle to NUL (the equivalent of /dev/null on Windows)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        
        # Redirect stdout and stderr to NUL
        os.dup2(devnull_fd, original_stdout_fd)
        os.dup2(devnull_fd, original_stderr_fd)
        
        # The code inside the 'with' block will now run with both streams suppressed
        yield
    finally:
        # Restore the original stdout and stderr
        os.close(devnull_fd)
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)

# --- Sopprimi TUTTI gli avvisi di music21 ---
warnings.filterwarnings('ignore', module='music21')

'''
# --- USAGE ----------------------------------------------
python dataset_creator.py --base_data_dir /percorso/alla/tua/cartella/mutopia_data --output_mode chunked 

--force_tokenizer_build (force the reconstruction of the tokenizer, useful if you change the config.py)
--piano_only (filter only piano channels) 
--extract_genre (extract genres as a metadata)
--no-key (Disabilita l'inclusione dei metadati sulla tonalità)
--no-time-signature (Disabilita l'inclusione del metro)
--no-tempo (Disabilita l'inclusione del BPM)
--no-velocity (Disabilita l'inclusione della velocity (media e range))
--no-instruments (Disabilita l'inclusione dei nomi e del numero di strumenti)
--transpose_piano_only (transpose piano notes to C major / A minor, use only with --piano_only)
--fast (fast control of keys from metadata -without inferring from music21-)
--delete_skipped_files (delete all files not included in the final dataset)
--dry_run_delete (simulate the deletion without actually deleting files)
--min_token_freq [int] (only for chunked mode, minimum frequency of tokens to keep in the vocabulary)
----------------------------------------------------------
'''

# --- EXAMPLE --------------------------------------------
# python dataset_creator.py --base_data_dir "C:\Users\Michael\Desktop\MusicDatasets\Datasets\adl_piano_midi" --transpose-to-C --piano_only --fast --min_token_freq 1500
# --------------------------------------------------------

# --- Additions for Tokenization and Chunking ---
import miditok
from symusic import Score # For loading MIDI for miditok
from symusic import Note as SymusicNote # Import Note for type hinting and manipulation

ESSENTIAL_METADATA = [] # ['time_signature', 'bpm_rounded', 'avg_velocity_rounded', 'velocity_range_rounded', 'mido_declared_key_signature'] Commented in genres mode
MIN_MIDO_DURATION_SECONDS = 30
MIN_NOTE_ON_MESSAGES = 100
TRAIN_SPLIT = 0.85
VALIDATION_SPLIT = 0.15

global MIDI_TOKENIZER

GM_INSTRUMENT_MAP = {
    0: "Acoustic Grand Piano", 1: "Bright Acoustic Piano", 2: "Electric Grand Piano", 3: "Honky-tonk Piano",
    4: "Electric Piano 1", 5: "Electric Piano 2", 6: "Harpsichord", 7: "Clavinet",
    8: "Celesta", 9: "Glockenspiel", 10: "Music Box", 11: "Vibraphone", 12: "Marimba", 13: "Xylophone",
    14: "Tubular Bells", 15: "Dulcimer", 16: "Drawbar Organ", 17: "Percussive Organ", 18: "Rock Organ",
    19: "Church Organ", 20: "Reed Organ", 21: "Accordion", 22: "Harmonica", 23: "Tango Accordion",
    24: "Acoustic Guitar (nylon)", 25: "Acoustic Guitar (steel)", 26: "Electric Guitar (jazz)",
    27: "Electric Guitar (clean)", 28: "Electric Guitar (muted)", 29: "Overdriven Guitar",
    30: "Distortion Guitar", 31: "Guitar Harmonics",32: "Acoustic Bass", 33: "Electric Bass (finger)",
    34: "Electric Bass (pick)", 35: "Fretless Bass", 36: "Slap Bass 1", 37: "Slap Bass 2", 38: "Synth Bass 1",
    39: "Synth Bass 2", 40: "Violin", 41: "Viola", 42: "Cello", 43: "Contrabass", 44: "Tremolo Strings",
    45: "Pizzicato Strings", 46: "Orchestral Harp", 47: "Timpani", 48: "String Ensemble 1",
    49: "String Ensemble 2", 50: "Synth Strings 1", 51: "Synth Strings 2", 52: "Choir Aahs",
    53: "Voice Oohs", 54: "Synth Voice", 55: "Orchestra Hit", 56: "Trumpet", 57: "Trombone", 58: "Tuba",
    59: "Muted Trumpet", 60: "French Horn", 61: "Brass Section", 62: "Synth Brass 1", 63: "Synth Brass 2",
    64: "Soprano Sax", 65: "Alto Sax", 66: "Tenor Sax", 67: "Baritone Sax", 68: "Oboe", 69: "English Horn",
    70: "Bassoon", 71: "Clarinet", 72: "Piccolo", 73: "Flute", 74: "Recorder", 75: "Pan Flute",
    76: "Blown Bottle", 77: "Shakuhachi", 78: "Whistle", 79: "Ocarina", 80: "Lead 1 (square)",
    81: "Lead 2 (sawtooth)", 82: "Lead 3 (calliope)", 83: "Lead 4 (chiff)", 84: "Lead 5 (charang)",
    85: "Lead 6 (voice)", 86: "Lead 7 (fifths)", 87: "Lead 8 (bass + lead)", 88: "Pad 1 (new age)",
    89: "Pad 2 (warm)", 90: "Pad 3 (polysynth)", 91: "Pad 4 (choir)", 92: "Pad 5 (bowed)",
    93: "Pad 6 (metallic)", 94: "Pad 7 (halo)", 95: "Pad 8 (sweep)", 96: "FX 1 (rain)",
    97: "FX 2 (soundtrack)", 98: "FX 3 (crystal)", 99: "FX 4 (atmosphere)", 100: "FX 5 (brightness)",
    101: "FX 6 (goblins)", 102: "FX 7 (echoes)", 103: "FX 8 (sci-fi)", 104: "Sitar",
    105: "Banjo", 106: "Shamisen", 107: "Koto", 108: "Kalimba", 109: "Bagpipe",
    110: "Fiddle", 111: "Shanai", 112: "Tinkle Bell", 113: "Agogo", 114: "Steel Drums",
    115: "Woodblock", 116: "Taiko Drum", 117: "Melodic Tom", 118: "Synth Drum",
    119: "Reverse Cymbal", 120: "Guitar Fret Noise", 121: "Breath Noise", 122: "Seashore",
    123: "Bird Tweet", 124: "Telephone Ring", 125: "Helicopter", 126: "Applause",
    127: "Gunshot"
}

# --- Configurazione del Tokenizer MIDI ---
REFERENCE_KEY_MAJOR_M21 = music21.key.Key(config.REFERENCE_KEY_MAJOR, 'major')
REFERENCE_KEY_MINOR_M21 = music21.key.Key(config.REFERENCE_KEY_MINOR, 'minor')


# Mappa per convertire le firme di chiave di mido in scostamenti di semitoni dalla chiave di riferimento
# Basato sulla Circle of Fifths (da C)
KEY_SEMITONE_MAP = {
    "C": 0, "Am": 0,
    "G": 7, "Em": 7,
    "D": 2, "Bm": 2, # D is 2 sharps, so +2 semitones from C
    "A": 9, "F#m": 9,
    "E": 4, "C#m": 4,
    "B": 11, "G#m": 11,
    "F#": 6, "D#m": 6,
    "C#": 1, "A#m": 1,

    "F": 5, "Dm": 5, # F is 1 flat, so -5 semitones or +7 semitones (same as G)
    "Bb": 10, "Gm": 10,
    "Eb": 3, "Cm": 3,
    "Ab": 8, "Fm": 8,
    "Db": 1, "Bbm": 1, # Db is 5 flats, so -1 semitone (same as C#)
    "Gb": 6, "Ebm": 6, # Gb is 6 flats, so -6 semitones (same as F#)
    "Cb": 11, "Abm": 11 # Cb is 7 flats, so -1 semitone (same as B)
}
# Aggiustamenti per le chiavi bemolli per renderle coerenti (es. Db = C#)
KEY_SEMITONE_MAP["Db"] = KEY_SEMITONE_MAP["C#"]
KEY_SEMITONE_MAP["Gb"] = KEY_SEMITONE_MAP["F#"]
KEY_SEMITONE_MAP["Cb"] = KEY_SEMITONE_MAP["B"]


# --- FUNZIONE PER IL PRE-FILTRAGGIO ---
def check_file_validity(args_tuple):
    """
    Worker leggero per il pre-filtraggio. Controlla se un file MIDI è valido
    per l'elaborazione principale senza eseguire la tokenizzazione completa.
    Ritorna il percorso del file se è valido, altrimenti None.
    """
    midi_file_path, transposition_enabled = args_tuple
    
    # Controllo di base con mido
    mido_check_result = quick_midi_check_mido(str(midi_file_path))
    if not mido_check_result['passed']:
        return None  # Scarta se non passa i controlli base

    # Controllo specifico per la trasposizione (se abilitata)
    if transposition_enabled:
        declared_key = mido_check_result.get('key_signature_declared')
        if not declared_key:
            return None # Scarta se la trasposizione è richiesta ma la chiave non c'è
            
        # Potremmo anche controllare se la chiave è riconosciuta, ma per ora basta questo
        if get_transposition_semitones(declared_key, config.REFERENCE_KEY_MAJOR, config.REFERENCE_KEY_MINOR) is None:
            return None # Scarta se la chiave non è nel nostro map

    # Se tutti i controlli passano, il file è valido
    return midi_file_path

# --- Funzione per costruire o caricare il tokenizer MIDI ---
def build_or_load_tokenizer_for_creator(midi_file_paths_for_vocab_build=None, force_build=False):
    """
    Costruisce o carica il tokenizer MIDI utilizzando la configurazione centralizzata
    da config.py e gestisce correttamente l'addestramento e il salvataggio dei
    file specifici per la strategia (es. dimensioni vocabolario per Octuple).
    """
    paths = config.get_project_paths(args.base_data_dir)
    VOCAB_PATH = paths["midi_vocab"]
    tokenizer = None

    if VOCAB_PATH.exists() and not force_build:
        logging.info(f"Caricamento del tokenizer MIDI da {VOCAB_PATH}")
        try:
            # La warning sui "controls" qui è normale, la libreria ci avvisa che li disabilita
            tokenizer = config.MIDI_TOKENIZER_STRATEGY(params=str(VOCAB_PATH))
            logging.info(f"Tokenizer caricato con successo da {VOCAB_PATH}")
        except Exception as e:
            logging.error(f"Errore durante il caricamento del tokenizer da {VOCAB_PATH}: {e}", exc_info=True)
            force_build = True

    if not VOCAB_PATH.exists() or force_build:
        logging.info("Creazione di una nuova configurazione per il tokenizer MIDI...")
        tokenizer = config.MIDI_TOKENIZER_STRATEGY(tokenizer_config=config.TOKENIZER_PARAMS)

        # --- MODIFICA: Chiama .train() solo se il tokenizer lo supporta ---
        if midi_file_paths_for_vocab_build and hasattr(tokenizer, 'train'):
            logging.info(f"Addestramento del tokenizer con {len(midi_file_paths_for_vocab_build)} file.")
            tokenizer.train(
                vocab_size=50000,
                model="BPE",
                files_paths=midi_file_paths_for_vocab_build
            )
            logging.info("Addestramento del tokenizer completato.")
        elif midi_file_paths_for_vocab_build:
            logging.info(f"Il tokenizer {type(tokenizer).__name__} non richiede addestramento, salto il passaggio.")
        # --- FINE MODIFICA ---
        
        VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(VOCAB_PATH))
        logging.info(f"Configurazione del tokenizer salvata in {VOCAB_PATH}")

    if isinstance(tokenizer, miditok.Octuple):
        VOCAB_SIZES_PATH = paths.get("midi_vocab_sizes")
        if VOCAB_SIZES_PATH:
            logging.info("Rilevata strategia Octuple, salvataggio delle dimensioni del vocabolario.")
            vocab_sizes = tokenizer.vocab_size # Corretto in vocab_size (singolare)
            try:
                VOCAB_SIZES_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(VOCAB_SIZES_PATH, 'w', encoding='utf-8') as f:
                    json.dump(vocab_sizes, f, ensure_ascii=False, indent=2)
                logging.info(f"Dimensioni del vocabolario Octuple salvate in {VOCAB_SIZES_PATH}")
            except Exception as e:
                logging.error(f"Errore durante il salvataggio del file delle dimensioni del vocabolario Octuple in {VOCAB_SIZES_PATH}: {e}")
        else:
            logging.warning("Percorso 'midi_vocab_sizes' non definito in config.py. Impossibile salvare le dimensioni del vocabolario.")

    try:
        assert tokenizer[config.MIDI_PAD_TOKEN_NAME] is not None
        assert tokenizer[config.MIDI_SOS_TOKEN_NAME] is not None
        assert tokenizer[config.MIDI_EOS_TOKEN_NAME] is not None
    except AssertionError:
        logging.error("CRITICO: Token speciali MIDI mancanti nel tokenizer.")
        raise ValueError("Token speciali MIDI mancanti nel tokenizer.")

    logging.info(f"Tokenizer MIDI pronto. Dimensione vocabolario: {len(tokenizer)}")
    return tokenizer

# --- Funzioni di Analisi (mido) ---
def quick_midi_check_mido(file_path_str):
    # (Your existing quick_midi_check_mido function - unchanged)
    try:
        mid = mido.MidiFile(file_path_str)
        duration = mid.length
        if duration < MIN_MIDO_DURATION_SECONDS:
            return {'passed': False, 'reason': 'mido_too_short', 'duration': duration}

        note_on_count = 0
        velocities = [] 
        time_signature = None
        key_signature = None
        tempo_microseconds = None 
        
        found_instruments = set()
        channel_9_active_for_drums = False

        for i, track in enumerate(mid.tracks):
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_on_count += 1
                    velocities.append(msg.velocity)
                    if hasattr(msg, 'channel') and msg.channel == 9:
                        channel_9_active_for_drums = True 

                if not time_signature and msg.is_meta and msg.type == 'time_signature':
                    time_signature = f"{msg.numerator}/{msg.denominator}"
                if not key_signature and msg.is_meta and msg.type == 'key_signature':
                    key_signature = msg.key
                
                if tempo_microseconds is None and msg.is_meta and msg.type == 'set_tempo':
                    tempo_microseconds = msg.tempo

                if msg.type == 'program_change':
                    if hasattr(msg, 'channel'):
                        if msg.channel == 9: 
                            channel_9_active_for_drums = True
                        else:
                            instrument_gm_name = GM_INSTRUMENT_MAP.get(msg.program)
                            if instrument_gm_name:
                                found_instruments.add(instrument_gm_name)
        if note_on_count < MIN_NOTE_ON_MESSAGES:
            return {'passed': False, 'reason': 'mido_too_few_notes', 'note_count': note_on_count, 'duration': duration}

        if channel_9_active_for_drums:
            found_instruments.add("Drums") 
        
        # --- BLOCCO DI CONTROLLO PER STRUMENTO DI DEFAULT ---
        # Se, dopo aver analizzato tutto, non abbiamo trovato strumenti espliciti
        # MA il file contiene delle note, assumiamo lo strumento di default
        # dello standard General MIDI, ovvero l'Acoustic Grand Piano (Program 0).
        if not found_instruments and note_on_count > 0:
            logging.debug(f"File {Path(file_path_str).name}: Nessun Program Change trovato, ma sono presenti delle note. Assumo strumento di default (Acoustic Grand Piano).")
            found_instruments.add("Acoustic Grand Piano")
        # --- FINE NUOVO BLOCCO ---

        bpm_rounded = None
        if tempo_microseconds:
            bpm = 60_000_000 / tempo_microseconds
            bpm_rounded = round(bpm / 5) * 5

        avg_velocity_rounded = None
        velocity_range_rounded = None
        if velocities:
            avg_vel = sum(velocities) / len(velocities)
            vel_range = max(velocities) - min(velocities)
            avg_velocity_rounded = round(avg_vel / 5) * 5 
            velocity_range_rounded = round(vel_range / 5) * 5
            
        found_instruments.discard("")

        return {
            'passed': True,
            'reason': 'mido_passed',
            'duration_seconds': duration,
            'note_count': note_on_count,
            'time_signature': time_signature,
            'key_signature_declared': key_signature, # Include this for transposition logic
            'midi_instruments': sorted(list(found_instruments)), 
            'bpm_rounded': bpm_rounded,
            'avg_velocity_rounded': avg_velocity_rounded,
            'velocity_range_rounded': velocity_range_rounded
        }
    except IndexError: 
        return {'passed': False, 'reason': 'mido_index_error', 'detail': 'Error processing MIDI tracks/messages'}
    except Exception as e: # Catch more general mido errors
        # logging.warning(f"Mido parsing error for {file_path_str}: {e}")
        return {'passed': False, 'reason': 'mido_parse_error', 'detail': str(e)}


# --- Funzione di trasposizione ---
def transpose_score(score, semitones: int):
    """
    Transposes all notes in a symusic.Score object by the given number of semitones.
    """
    if semitones == 0:
        return score # No transposition needed

    for track in score.tracks:
        for note in track.notes:
            note.pitch = max(0, min(127, note.pitch + semitones)) # Ensure pitch stays within MIDI range
    return score

def get_transposition_semitones(declared_key: str, target_major: str, target_minor: str) -> int | None:
    """
    Calculates the semitone offset to transpose from declared_key to target_major/minor.
    Assumes declared_key is a string like 'C', 'Am', 'Db', 'F#m'.
    Returns None if declared_key is not recognized, otherwise returns the semitone offset (which can be 0).
    """
    if not declared_key:
        return None

    declared_key_normalized = declared_key.replace(' ', '').replace('m', 'm').strip()
    
    semitones_from_C = KEY_SEMITONE_MAP.get(declared_key_normalized)

    if semitones_from_C is None:
        if declared_key_normalized.endswith('major'):
            semitones_from_C = KEY_SEMITONE_MAP.get(declared_key_normalized[:-5])
        elif declared_key_normalized.endswith('minor'):
            semitones_from_C = KEY_SEMITONE_MAP.get(declared_key_normalized[:-5] + 'm')
        else:
            semitones_from_C = KEY_SEMITONE_MAP.get(declared_key_normalized)

    if semitones_from_C is None:
        logging.warning(f"Declared key '{declared_key}' not recognized for transposition. Skipping transposition for this file.")
        return None

    transposition_amount = -semitones_from_C
    
    return transposition_amount


# --- Funzione Worker per il Multiprocessing ---
def process_single_file(args_tuple):
    # --- MODIFICATO: Aggiunti i nuovi flag per il controllo dei metadati ---
    (midi_file_path, base_dir, midi_input_dir, output_dir, 
     binary_chunks_dir, output_mode, tokenizer_present, 
     transposition_enabled, use_fast_mode,
     is_piano_only, 
     # --- NUOVI ARGOMENTI ---
     should_extract_genre,
     include_key,
     include_time_signature,
     include_tempo,
     include_velocity,
     include_instruments
     ) = args_tuple
    
    global MIDI_TOKENIZER
    if output_mode == "chunked" and not tokenizer_present:
        logging.error(f"Tokenizer requested for chunking {midi_file_path.name} but not available in worker.")
        return [{'status': 'skipped_tokenizer_unavailable', 'filename': midi_file_path.name}]

    mido_check_result = quick_midi_check_mido(str(midi_file_path))

    if not mido_check_result['passed']:
        return [{'status': f"skipped_{mido_check_result['reason']}",
                'filename': midi_file_path.name,
                'skipped_path': str(midi_file_path),
                'detail': mido_check_result.get('detail', '')}]

    if len(mido_check_result.get('midi_instruments', [])) > 16:
        return [{'status': 'skipped_too_many_instruments',
                 'filename': midi_file_path.name,
                 'skipped_path': str(midi_file_path),
                 'instruments_found': len(mido_check_result.get('midi_instruments', []))}]

    final_metadata = {}
    
        # --- MODIFICATO: Creazione del dizionario dei metadati in base ai flag ---
    final_metadata = {}
    
    # Titolo e path sono sempre inclusi come riferimento
    final_metadata['title'] = midi_file_path.stem
    try:
        relative_path = str(midi_file_path.relative_to(base_dir))
    except ValueError:
        relative_path = str(midi_file_path.relative_to(midi_input_dir))
    final_metadata['midi_relative_path'] = relative_path
    
    # 1. Genere
    if should_extract_genre:
        try:
            relative_path_for_genre = midi_file_path.relative_to(midi_input_dir)
            if relative_path_for_genre.parts:
                genre = relative_path_for_genre.parts[0]
                final_metadata['genre'] = genre
        except (ValueError, IndexError):
            logging.warning(f"Could not extract genre from path for {midi_file_path.name}.")
            final_metadata['genre'] = None
            
    # 2. Tonalità (Key)
    if include_key:
        # La logica di trasposizione aggiungerà da sola i metadati relativi alla chiave
        # Qui ci assicuriamo che la chiave dichiarata da mido sia presente se la trasposizione non è attiva
        final_metadata['mido_declared_key_signature'] = mido_check_result.get('key_signature_declared')

    # 3. Metro (Time Signature)
    if include_time_signature:
        final_metadata['time_signature'] = mido_check_result.get('time_signature')

    # 4. Tempo (BPM)
    if include_tempo:
        final_metadata['bpm_rounded'] = mido_check_result.get('bpm_rounded')

    # 5. Velocity
    if include_velocity:
        final_metadata['avg_velocity_rounded'] = mido_check_result.get('avg_velocity_rounded')
        final_metadata['velocity_range_rounded'] = mido_check_result.get('velocity_range_rounded')
    
    # 6. Strumenti
    if include_instruments:
        final_metadata['midi_instruments'] = mido_check_result.get('midi_instruments', [])

    missing_essentials = [k for k in ESSENTIAL_METADATA if final_metadata.get(k) is None]
    if missing_essentials:
        return [{'status': 'skipped_mido_missing_metadata',
                'filename': midi_file_path.name,
                'skipped_path': str(midi_file_path),
                'missing': missing_essentials}]

    try:
        relative_path = str(midi_file_path.relative_to(base_dir))
    except ValueError:
        relative_path = str(midi_file_path.relative_to(midi_input_dir))
    final_metadata['midi_relative_path'] = relative_path
    
    if output_mode == "classic":
        return [{'status': 'success', 
                 'data': {'id': midi_file_path.stem, 'metadata': final_metadata}}]

    elif output_mode == "chunked":
        if MIDI_TOKENIZER is None:
             logging.error(f"MIDI_TOKENIZER is None in worker for {midi_file_path}, cannot proceed with chunking.")
             return [{'status': 'skipped_tokenizer_not_init',
                      'skipped_path': str(midi_file_path),
                      'filename': midi_file_path.name}]
        
        # >>> INIZIO BLOCCO TRY/EXCEPT PRINCIPALE MODIFICATO <<<
        try:
            score = None # Inizializziamo la variabile 'score'

            # -------------------------------------------------
            #  MODALITÀ LENTA (DEFAULT): Usa music21 per l'analisi
            # -------------------------------------------------
            if not use_fast_mode:
                logging.debug(f"Processing {midi_file_path.name} with standard/hybrid mode")
                score_m21 = converter.parse(str(midi_file_path))

                # La trasposizione si applica solo in modalità piano_only
                if transposition_enabled:
                    key_analysis = None # Oggetto che conterrà la tonalità di music21
                    
                    # 1. Recupera la tonalità dichiarata da Mido
                    declared_key = mido_check_result.get('key_signature_declared')

                    # 2. Tenta di usare la tonalità dichiarata se presente (percorso veloce)
                    if declared_key:
                        try:
                            key_analysis = music21.key.Key(declared_key)
                            logging.debug(f"Used declared key '{declared_key}' for {midi_file_path.name}")
                            final_metadata['original_key_source'] = 'mido_declared'
                        except music21.key.KeyException:
                            logging.warning(f"Mido key '{declared_key}' not recognized by music21. Falling back to full analysis.")
                            key_analysis = None

                    # 3. Se la tonalità dichiarata è assente o non valida, esegui l'analisi completa (percorso lento)
                    if not key_analysis:
                        try:
                            logging.debug(f"Declared key not found. Performing full music21 analysis for {midi_file_path.name}")
                            key_analysis = score_m21.analyze('key')
                            final_metadata['original_key_source'] = 'music21_analyzed'
                        except Exception as e:
                            logging.warning(f"music21.analyze('key') failed for {midi_file_path.name}: {e}")
                            key_analysis = None

                    # 4. Procedi con la trasposizione usando la tonalità trovata (se trovata)
                    if key_analysis:
                        try:
                            original_key_m21_str = str(key_analysis)
                            final_metadata['music21_detected_key'] = original_key_m21_str

                            target_key = REFERENCE_KEY_MAJOR_M21 if key_analysis.mode == 'major' else REFERENCE_KEY_MINOR_M21
                            interval = music21.interval.Interval(key_analysis.tonic, target_key.tonic)
                            
                            # La trasposizione modifica l'oggetto score_m21
                            score_m21.transpose(interval, inPlace=True)

                            final_metadata['transposed_to_key'] = f"{str(REFERENCE_KEY_MAJOR_M21)} / {str(REFERENCE_KEY_MINOR_M21)}"
                            final_metadata['transposition_semitones'] = interval.semitones
                        except Exception as e:
                            logging.error(f"Transposition failed for {midi_file_path.name} despite having a key. Error: {e}")
                
                temp_midi_file = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tf:
                        temp_midi_file = tf.name
                        # Scrive l'oggetto music21 (trasposto o meno) nel file temporaneo
                        score_m21.write('midi', fp=temp_midi_file)
                    # Carica il file temporaneo in symusic
                    with suppress_cpp_warnings():
                        score = Score(temp_midi_file)
                finally:
                    # Assicura la pulizia del file temporaneo
                    if temp_midi_file and Path(temp_midi_file).exists():
                        os.remove(temp_midi_file)


            # -------------------------------------------------
            #  MODALITÀ VELOCE (--fast): Usa mido e symusic
            # -------------------------------------------------
            else:
                logging.debug(f"Processing {midi_file_path.name} with --fast mode")
                # Carica direttamente con symusic (molto più veloce)
                with suppress_cpp_warnings():
                    score = Score(midi_file_path)
                
                if transposition_enabled:
                    declared_key = mido_check_result.get('key_signature_declared')
                    transposition_semitones = None # Inizializza
                    
                    if declared_key:
                        transposition_semitones = get_transposition_semitones(declared_key, config.REFERENCE_KEY_MAJOR, config.REFERENCE_KEY_MINOR)
                    
                    # Se la trasposizione è abilitata, ma non siamo riusciti a determinare
                    # di quanto trasporre (perché la chiave non era dichiarata o non era valida),
                    # scartiamo il file per garantire la coerenza del dataset.
                    if transposition_semitones is None:
                        return [{'status': 'skipped_transposition_key_not_found',
                                'filename': midi_file_path.name,
                                'skipped_path': str(midi_file_path)}]
                    
                    if transposition_semitones != 0:
                        score = transpose_score(score, transposition_semitones)
                            
                        # IMPOSTA SEMPRE I METADATI se la tonalità è stata riconosciuta, anche se l'offset è 0
                    final_metadata['transposed_to_key'] = f"{config.REFERENCE_KEY_MAJOR} major / {config.REFERENCE_KEY_MINOR} minor"
                    final_metadata['original_key_mido'] = declared_key
                    final_metadata['transposition_semitones'] = transposition_semitones
            
            # --- DA QUI, LA LOGICA È COMUNE A ENTRAMBE LE MODALITÀ ---
            if score is None:
                return [{'status': 'skipped_score_not_generated', 'skipped_path': str(midi_file_path), 'filename': midi_file_path.name}]
            
            if is_piano_only:
                # Prova prima un filtro stretto basato sui program number
                filtered_tracks = [track for track in score.tracks if track.program in config.PIANO_PROGRAMS]

                # Se il filtro stretto non trova nulla, applica una logica più permissiva:
                # in un dataset di pianoforte, assumi che ogni traccia non di percussioni sia pianoforte.
                if not filtered_tracks:
                    logging.info(f"File {midi_file_path.name}: Nessuna traccia con program 0-7. Mantengo tutte le tracce non-drum.")
                    filtered_tracks = [track for track in score.tracks if not track.is_drum]

                # Se anche dopo il filtro permissivo non ci sono tracce, scarta il file.
                if not filtered_tracks:
                    return [{'status': 'skipped_no_melodic_tracks_found', 'skipped_path': str(midi_file_path), 'filename': midi_file_path.name}]
                
                # Applica il filtro allo score
                score.tracks = filtered_tracks

            # --- NUOVA LOGICA DI ESTRAZIONE DEI TOKEN (SEMPLIFICATA) ---
            # Grazie a one_token_stream_for_programs=True, il tokenizer ora restituisce
            # sempre un singolo oggetto TokSequence.
            tokens_sequence = MIDI_TOKENIZER(score)

            # La proprietà .ids ci fornisce direttamente i token nel formato corretto:
            # - Per Octuple: una lista di liste di int (es. [[t1_p, t1_v], [t2_p, t2_v]])
            # - Per altri tokenizer (es. REMI): una lista di int (es. [t1, t2, t3])
            raw_midi_ids = tokens_sequence.ids
            # --- FINE NUOVA LOGICA ---
            
            if not raw_midi_ids or len(raw_midi_ids) < config.MIN_CHUNK_LEN_MIDI:
                return [{'status': 'skipped_too_short_after_tokenization', 'skipped_path': str(midi_file_path), 'filename': midi_file_path.name, 'token_count': len(raw_midi_ids)}]

            chunked_samples = []
            effective_chunk_len_for_data = config.MAX_SEQ_LEN_MIDI - 2 
            
            if effective_chunk_len_for_data < config.MIN_CHUNK_LEN_MIDI:
                 logging.error(f"Configuration error: effective_chunk_len_for_data is less than MIN_CHUNK_LEN_MIDI.")
                 return [{'status': 'skipped_chunk_config_error', 'skipped_path': str(midi_file_path), 'filename': midi_file_path.name}]

            num_file_chunks = 0
            for i in range(0, len(raw_midi_ids), effective_chunk_len_for_data):
                chunk_token_ids = raw_midi_ids[i : i + effective_chunk_len_for_data]
                if len(chunk_token_ids) < config.MIN_CHUNK_LEN_MIDI:
                    if num_file_chunks == 0:
                         return [{'status': 'skipped_first_chunk_too_short', 'skipped_path': str(midi_file_path), 'filename': midi_file_path.name, 'token_count': len(chunk_token_ids)}]
                    break

                relative_path_str = midi_file_path.relative_to(midi_input_dir).as_posix()
                path_hash = hashlib.sha1(relative_path_str.encode('utf-8')).hexdigest()
                safe_path_id = f"{midi_file_path.stem}_{path_hash}"
                chunk_id = f"{safe_path_id}_chunk{num_file_chunks}"
                binary_file_path = binary_chunks_dir / f"{chunk_id}.npy"
                
                # np.array gestirà correttamente sia le liste 1D (REMI) che 2D (Octuple)
                token_array = np.array(chunk_token_ids, dtype=np.uint16)

                try:
                    binary_chunks_dir.mkdir(parents=True, exist_ok=True)
                    np.save(binary_file_path, token_array)
                except Exception as e:
                    logging.error(f"Errore nel salvare il file binario {binary_file_path}: {e}")
                    continue

                chunk_data = {
                    'id': chunk_id,
                    'metadata': final_metadata.copy(),
                    'token_ids_path': binary_file_path.relative_to(output_dir).as_posix()
                }
                
                chunk_data['metadata']['original_file_id'] = midi_file_path.stem
                chunked_samples.append({'status': 'success_chunked', 'data': chunk_data})
                num_file_chunks +=1
            
            if not chunked_samples:
                return [{'status': 'skipped_no_valid_chunks_created', 'skipped_path': str(midi_file_path), 'filename': midi_file_path.name, 'total_tokens': len(raw_midi_ids)}]

            return chunked_samples

        # >>> BLOCCO EXCEPT MODIFICATO PER CATTURARE GLI ERRORI FATALI <<<
        except IndexError as e:
            # Cattura specificamente l'errore che ha causato il crash
            logging.warning(f"Caught IndexError (likely >15 instruments) in music21 for {midi_file_path.name}: {e}. Skipping file.")
            return [{'status': 'skipped_music21_index_error', 'skipped_path': str(midi_file_path), 'filename': midi_file_path.name, 'detail': str(e)}]
        except Music21Exception as e:
            logging.warning(f"Music21 related error for {midi_file_path.name}: {e}")
            return [{'status': 'skipped_music21_error', 'skipped_path': str(midi_file_path), 'filename': midi_file_path.name, 'detail': str(e)}]
        except FileNotFoundError:
             logging.warning(f"File not found by music21/symusic for tokenization: {midi_file_path.name}")
             return [{'status': 'skipped_file_not_found_for_tokenization', 'skipped_path': str(midi_file_path), 'filename': midi_file_path.name}]
        except Exception as e: # Cattura qualsiasi altro errore imprevisto
            logging.error(f"Unexpected error during chunking for {midi_file_path.name}: {e}", exc_info=False) # exc_info=False per non riempire il log
            return [{'status': 'skipped_chunking_unexpected_error', 'skipped_path': str(midi_file_path), 'filename': midi_file_path.name, 'detail': str(e)}]
    else:
        logging.error(f"Unknown output mode: {output_mode}")
        return [{'status': 'skipped_unknown_output_mode', 'skipped_path': str(midi_file_path), 'filename': midi_file_path.name}]

def save_dataset_split(data, filename):
    output_path = OUTPUT_DIR / f"{filename}.jsonl"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry_wrapper in data:
                if isinstance(entry_wrapper, dict) and entry_wrapper.get('status', '').startswith('success'):
                    f.write(json.dumps(entry_wrapper['data']) + '\n')
                elif isinstance(entry_wrapper, list):
                     for item in entry_wrapper:
                         if isinstance(item, dict) and item.get('status', '').startswith('success'):
                            f.write(json.dumps(item['data']) + '\n')

    except TypeError as e:
         logging.error(f"TypeError during JSON writing in {output_path}: {e}. Check data structure.")
    except Exception as e:
        logging.error(f"Error saving {output_path}: {e}")

def init_worker(vocab_p, midi_tok_strat_class):
    global MIDI_TOKENIZER
    logging.info(f"Worker {os.getpid()} initializing tokenizer.")
    try:
        MIDI_TOKENIZER = midi_tok_strat_class(params=str(vocab_p)) 
        logging.info(f"Worker {os.getpid()} tokenizer initialized with vocab size: {len(MIDI_TOKENIZER)}")
    except Exception as e:
        logging.error(f"Worker {os.getpid()} failed to initialize tokenizer: {e}", exc_info=True)
        MIDI_TOKENIZER = None

def _worker_wrapper_for_calibration(init_args_tuple, process_args_tuple):
    """
    Wrapper per i processi di calibrazione: prima inizializza il worker,
    poi esegue la funzione di elaborazione del file.
    """
    # Chiama l'inizializzatore del worker per creare MIDI_TOKENIZER
    init_worker(init_args_tuple[0], init_args_tuple[1])
    
    # Esegui la funzione di elaborazione del file
    # Nota: non ci interessa il risultato, serve solo per misurare la memoria
    process_single_file(process_args_tuple)

def determine_optimal_workers(all_midi_paths, processing_args_template, safety_factor=0.85, sample_size=10):
    """
    Determina il numero ottimale di processi worker basandosi sulla RAM disponibile
    e sulla memoria consumata da un campione di elaborazioni.
    """
    logging.info("--- Fase di Auto-Calibrazione per il Numero di Worker ---")

    # 1. Stima della memoria per singolo worker
    logging.info(f"Elaborazione di un campione di {sample_size} file per stimare l'uso di memoria...")
    if not all_midi_paths:
        logging.warning("Nessun file MIDI per la calibrazione, ritorno un valore di default di 2 workers.")
        return 2

    sample_files = random.sample(all_midi_paths, min(sample_size, len(all_midi_paths)))
    peak_memory_usage = 0

    # --- MODIFICA: Prepara gli argomenti per l'inizializzatore del worker ---
    init_args_for_worker = (VOCAB_PATH, config.MIDI_TOKENIZER_STRATEGY)
    
    for i, midi_file in enumerate(sample_files):
        # Prepara gli argomenti per questa specifica elaborazione
        args_tuple_for_processing = (midi_file,) + processing_args_template[1:]
        
        # --- MODIFICA: Usa la funzione wrapper che inizializza il tokenizer ---
        process = multiprocessing.Process(
            target=_worker_wrapper_for_calibration,
            args=(init_args_for_worker, args_tuple_for_processing)
        )
        # --- FINE MODIFICA ---
        
        process.start()
        
        p = psutil.Process(process.pid)
        local_peak = 0
        while process.is_alive():
            try:
                mem_info = p.memory_info().rss
                if mem_info > local_peak:
                    local_peak = mem_info
            except psutil.NoSuchProcess:
                break
            time.sleep(0.1)
        
        process.join()
        logging.info(f"Campione {i+1}/{sample_size}: Picco di memoria rilevato: {local_peak / 1024**2:.2f} MB")
        if local_peak > peak_memory_usage:
            peak_memory_usage = local_peak

    if peak_memory_usage == 0:
        logging.warning("Impossibile misurare l'uso di memoria. Ritorno un valore di default di 2 workers.")
        return 2

    memory_per_worker_estimate = peak_memory_usage * 1.25
    logging.info(f"Stima memoria per worker (con buffer del 25%): {memory_per_worker_estimate / 1024**2:.2f} MB")

    total_wsl_memory = psutil.virtual_memory().total
    available_for_workers = total_wsl_memory * safety_factor
    logging.info(f"RAM totale in WSL: {total_wsl_memory / 1024**2:.2f} MB. RAM utilizzabile per i workers ({safety_factor*100}%): {available_for_workers / 1024**2:.2f} MB")

    num_cpu_cores = os.cpu_count()
    
    if memory_per_worker_estimate == 0:
        calculated_workers = num_cpu_cores
    else:
        calculated_workers = int(available_for_workers / memory_per_worker_estimate)
    
    optimal_workers = max(1, min(num_cpu_cores, calculated_workers))

    logging.info(f"Numero di core CPU disponibili: {num_cpu_cores}")
    logging.info(f"Numero di worker calcolato in base alla RAM: {calculated_workers}")
    logging.info(f"--- Numero ottimale di worker impostato a: {optimal_workers} ---")
    
    return optimal_workers

# --- Funzioni per la creazione dei vocabolari dei metadati ---

def build_metadata_vocabs_and_frequencies(all_metadata_examples, min_freq=1):
    """
    Crea i vocabolari dei metadati e calcola la frequenza di ogni token.
    Non scrive file, ma ritorna le strutture dati.
    """
    logging.info("Costruzione vocabolario metadati e calcolo frequenze...")
    metadata_tokens = set()
    frequency_counter = collections.Counter()

    # Itera su tutti i metadati per tokenizzare e contare
    for meta_dict in tqdm(all_metadata_examples, desc="Tokenizing metadata for vocab"):
        tokens = tokenize_metadata(meta_dict)
        if tokens:
            metadata_tokens.update(tokens)
            frequency_counter.update(tokens)
            
    # --- FASE DI FILTRAGGIO ---
    logging.info(f"Conteggio iniziale: {len(frequency_counter)} token unici.")
    
    # Crea un set di token che soddisfano la soglia di frequenza
    frequent_metadata_tokens = {
        token for token, count in frequency_counter.items() if count >= min_freq
    }
    logging.info(f"Dopo il filtraggio (soglia >= {min_freq}): {len(frequent_metadata_tokens)} token unici rimasti.")
    # --- FINE FILTRAGGIO ---

    # Crea il vocabolario token -> id
    special_tokens = [config.META_PAD_TOKEN_NAME, config.META_UNK_TOKEN_NAME, config.META_SOS_TOKEN_NAME, config.META_EOS_TOKEN_NAME]
    all_tokens_list = special_tokens + sorted(list(metadata_tokens))
    
    # Rimuovi duplicati mantenendo l'ordine (se un token metadato avesse lo stesso nome di uno speciale)
    seen = set()
    unique_tokens_ordered = [x for x in all_tokens_list if not (x in seen or seen.add(x))]

    token_to_id = {token: i for i, token in enumerate(unique_tokens_ordered)}
    
    # Assicura che PAD sia a 0
    if token_to_id.get(config.META_PAD_TOKEN_NAME) != 0:
        logging.warning(f"ID del META_PAD_TOKEN_NAME non era 0. Riassegno gli ID per coerenza.")
        # Crea una nuova lista ordinata con PAD all'inizio
        pad_token = config.META_PAD_TOKEN_NAME
        other_tokens = [tok for tok in unique_tokens_ordered if tok != pad_token]
        final_tokens_list = [pad_token] + other_tokens
        token_to_id = {token: i for i, token in enumerate(final_tokens_list)}

    logging.info(f"Costruzione vocabolario metadati completata. Trovati {len(token_to_id)} token unici.")
    return token_to_id, frequency_counter

def save_metadata_vocab(token_to_id, path):
    """Salva il vocabolario standard dei metadati (token -> id) in un file JSON."""
    logging.info(f"Salvataggio vocabolario Metadati in {path}")
    try:
        id_to_token = {i: t for t, i in token_to_id.items()}
        vocab_data = {'token_to_id': token_to_id, 'id_to_token': id_to_token}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Errore durante il salvataggio del file vocabolario {path}: {e}")

def save_metadata_frequency(frequency_counter, path):
    """Salva il conteggio delle frequenze dei metadati in un file JSON, ordinato per occorrenze."""
    logging.info(f"Salvataggio del conteggio delle frequenze dei metadati in {path}")
    try:
        # Ordina il contatore per frequenza (dal più alto al più basso)
        sorted_freq_dict = dict(sorted(frequency_counter.items(), key=lambda item: item[1], reverse=True))
        
        output_data = {
            "metadata_token_counts": sorted_freq_dict,
            "total_unique_tokens": len(sorted_freq_dict)
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logging.info("Salvataggio del conteggio delle frequenze completato.")
    except Exception as e:
        logging.error(f"Errore durante il salvataggio del file delle frequenze {path}: {e}")

def build_or_load_metadata_vocab(all_metadata_examples, METADATA_VOCAB_PATH, force_build=False):
    if METADATA_VOCAB_PATH.exists() and not force_build:
        logging.info(f"Caricamento vocabolario Metadati da {METADATA_VOCAB_PATH}")
        with open(METADATA_VOCAB_PATH, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        token_to_id = vocab_data['token_to_id']
        required_specials = [config.META_PAD_TOKEN_NAME, config.META_UNK_TOKEN_NAME, config.META_SOS_TOKEN_NAME, config.META_EOS_TOKEN_NAME]
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

        all_tokens_list = [config.META_PAD_TOKEN_NAME, config.META_UNK_TOKEN_NAME, config.META_SOS_TOKEN_NAME, config.META_EOS_TOKEN_NAME] + sorted(list(metadata_tokens))
        token_to_id = {token: i for i, token in enumerate(all_tokens_list)}
        id_to_token = {i: token for token, i in token_to_id.items()}

        if token_to_id[config.META_PAD_TOKEN_NAME] != 0:
            logging.warning(f"ID del META_PAD_TOKEN_NAME ({config.META_PAD_TOKEN_NAME}) non è 0. Riassegno gli ID per coerenza con ignore_index.")
            # Riassegna per avere PAD = 0
            pad_tok = config.META_PAD_TOKEN_NAME
            other_specials = [t for t in [config.META_UNK_TOKEN_NAME, config.META_SOS_TOKEN_NAME, config.META_EOS_TOKEN_NAME] if t in token_to_id] # quelli già presenti
            unique_metadata_tokens_sorted = sorted(list(metadata_tokens))
            
            all_tokens_reordered = [pad_tok] + \
                                   [s for s in other_specials if s != pad_tok] + \
                                   [mt for mt in unique_metadata_tokens_sorted if mt not in [pad_tok] + other_specials]
            # Assicura che tutti i token originali siano presenti, specialmente se un token metadato avesse lo stesso nome di uno speciale
            final_token_set = set(all_tokens_reordered)
            for special_tok_defined in [config.META_UNK_TOKEN_NAME, config.META_SOS_TOKEN_NAME, config.META_EOS_TOKEN_NAME]:
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
            logging.info(f"Nuovo ID META_PAD_TOKEN_NAME: {token_to_id.get(config.META_PAD_TOKEN_NAME, 'NON TROVATO DOPO RIORDINO')}")


        vocab_data = {'token_to_id': token_to_id, 'id_to_token': id_to_token}
        logging.info(f"Salvataggio vocabolario Metadati in {METADATA_VOCAB_PATH}")
        METADATA_VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(METADATA_VOCAB_PATH, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        logging.info(f"Dimensione vocabolario Metadati (incl. speciali): {len(token_to_id)}")
        return token_to_id, id_to_token

# --- Main Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create MIDI dataset splits.")
    # ... (Argomenti --base_data_dir, --output_mode, --force_tokenizer_build rimangono invariati) ...
    parser.add_argument(
        "--base_data_dir", type=Path, required=True,
        help="Path to the base data directory, which should contain the 'MIDI' subdirectory."
    )
    parser.add_argument(
        "--output_mode", type=str, choices=["classic", "chunked"], default="chunked",
        help="Output mode."
    )
    parser.add_argument(
        "--force_tokenizer_build", action="store_true",
        help="Force rebuild of MIDI tokenizer vocabulary."
    )
    
    # --- GRUPPO DI ARGOMENTI PER CONTROLLO DEI METADATI ---
    metadata_group = parser.add_argument_group('Metadata Inclusion Controls')
    metadata_group.add_argument(
        "--extract_genre", action="store_true",
        help="Estrae il genere dalla struttura delle cartelle."
    )
    metadata_group.add_argument(
        "--no-key", dest='include_key', action='store_false',
        help="NON includere i metadati relativi alla tonalità."
    )
    metadata_group.add_argument(
        "--no-time-signature", dest='include_time_signature', action='store_false',
        help="NON includere i metadati relativi al metro (time signature)."
    )
    metadata_group.add_argument(
        "--no-tempo", dest='include_tempo', action='store_false',
        help="NON includere i metadati relativi al tempo (BPM)."
    )
    metadata_group.add_argument(
        "--no-velocity", dest='include_velocity', action='store_false',
        help="NON includere i metadati relativi alla velocity."
    )
    metadata_group.add_argument(
        "--no-instruments", dest='include_instruments', action='store_false',
        help="NON includere i metadati relativi agli strumenti (nomi e numero)."
    )
    parser.set_defaults(
        include_key=True, include_time_signature=True, include_tempo=True, 
        include_velocity=True, include_instruments=True
    )
    
    # --- ALTRI ARGOMENTI (piano, transpose, ecc.) ---
    parser.add_argument(
        "--piano_only", action="store_true",
        help="Filtra ogni file MIDI per mantenere solo le tracce di pianoforte (programmi 0-7)."
    )
    parser.add_argument(
    "--transpose-to-C",
    dest='transpose_enabled', # L'argomento imposterà la variabile 'transpose_enabled'
    action="store_true",
    help="Abilita la trasposizione di tutti i brani a Do Maggiore / La minore per normalizzare la tonalità."
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use fast processing mode."
    )
    destructive_group = parser.add_argument_group(
        'Operazioni Distruttive (Usare con Cautela!)'
    )
    destructive_group.add_argument(
        "--delete_skipped_files",
        action="store_true",
        help="ATTIVA MODALITÀ DISTRUTTIVA: Elimina permanentemente i file MIDI sorgente che vengono scartati "
             "durante il processo di creazione del dataset. Usare con estrema cautela."
    )
    destructive_group.add_argument(
        "--dry_run_delete",
        action="store_true",
        help="Esegue una simulazione (dry run) dell'eliminazione. Mostra quali file verrebbero eliminati "
             "da --delete_skipped_files senza cancellarli effettivamente. Consigliato prima di usare l'opzione di eliminazione."
    )
    metadata_group.add_argument(
        "--min_token_freq",
        type=int,
        default=10,
        help="Soglia minima di occorrenze per includere un token di metadati nel vocabolario finale. Default: 10"
    )
    args = parser.parse_args()

    # --- DEFINIZIONE DEI PERCORSI GLOBALI BASATA SUGLI ARGOMENTI ---
    paths = config.get_project_paths(args.base_data_dir)

    BASE_DATA_DIR = paths["base"]
    MIDI_INPUT_DIR = paths["midi_input"]
    OUTPUT_DIR = paths["output_splits"]
    BINARY_CHUNKS_DIR = paths["binary_chunks"]
    LOG_FILE = paths["log_file"]
    VOCAB_PATH = paths["midi_vocab"]
    
    # Riconfigura il logging per usare il nuovo percorso del file di log
    # Rimuovi eventuali handler esistenti
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Aggiungi i nuovi handler con il percorso corretto
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        handlers=[
            # Scrivi solo sul file di log e non più sul terminale
            logging.FileHandler(LOG_FILE, encoding='utf-8')
        ]
    )
    # Aggiungiamo un messaggio per essere sicuri che il log su file funzioni
    logging.info("Logging configurato. L'output da ora sarà visibile solo nel file di log.")
    print("Logging configurato. L'output da ora sarà visibile solo nel file di log. La barra di progresso avrà il terminale per sé.")

    logging.info(f"--- Dataset Creation Script --- Base Directory: {BASE_DATA_DIR} ---")
    logging.info(f"Output Mode: {args.output_mode}")
    logging.info(f"Transposition Enabled: {args.transpose_enabled}")

    temp_midi_files_for_vocab = []
    
    # --- TROVA TUTTI I FILE MIDI POTENZIALI ---
    logging.info("Ricerca di tutti i file MIDI potenziali...")
    all_potential_midi_files = list(MIDI_INPUT_DIR.rglob("*.mid")) + list(MIDI_INPUT_DIR.rglob("*.midi"))
    logging.info(f"Trovati {len(all_potential_midi_files)} file MIDI potenziali in {MIDI_INPUT_DIR}.")

    if not all_potential_midi_files:
        logging.error("Nessun file MIDI trovato. Uscita.")
        exit(1)

    # --- FASE 0: PRE-FILTRAGGIO DEI FILE MIDI (Eseguito solo in modalità --fast) ---
    valid_midi_files_for_processing = []
    if args.fast:
        logging.info("--- Fase 0: Pre-filtraggio dei file per garantire la validità (modalità --fast attiva) ---")
        
        pre_filter_tasks = [(path, args.transpose_enabled) for path in all_potential_midi_files]
        num_workers_prefilter = max(1, os.cpu_count() - 1)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers_prefilter) as executor:
            results_iterator = executor.map(check_file_validity, pre_filter_tasks)
            
            progress_bar = tqdm(results_iterator, total=len(all_potential_midi_files), desc="Pre-filtraggio file", unit="file")
            
            for result_path in progress_bar:
                if result_path is not None:
                    valid_midi_files_for_processing.append(result_path)

        logging.info(f"Pre-filtraggio completato. {len(valid_midi_files_for_processing)} su {len(all_potential_midi_files)} file sono risultati validi per l'elaborazione.")

        if not valid_midi_files_for_processing:
            logging.error("Nessun file MIDI valido trovato dopo il pre-filtraggio. Uscita.")
            exit(1)
    else:
        # Se non siamo in modalità --fast, saltiamo il pre-filtraggio. 
        # I file verranno controllati singolarmente durante l'elaborazione principale.
        logging.info("Modalità --fast non attiva. Salto del pre-filtraggio.")
        valid_midi_files_for_processing = all_potential_midi_files

    # --- FASE 0.5: COSTRUZIONE VOCABOLARIO TOKENIZER MIDI (ORA SUI FILE VALIDI) ---
    temp_midi_files_for_vocab = []
    if not VOCAB_PATH.exists() or args.force_tokenizer_build:
        logging.info("Preparazione campione per vocabolario tokenizer MIDI dai file validi...")
        
        # Usa la logica di campionamento che abbiamo già definito, ma sulla lista filtrata
        SAMPLE_PERCENTAGE = 0.2
        MAX_SAMPLE_SIZE = 50000
        total_valid_files = len(valid_midi_files_for_processing)
        sample_size = min(int(total_valid_files * SAMPLE_PERCENTAGE), MAX_SAMPLE_SIZE, total_valid_files)
        sample_size = max(50, sample_size)

        logging.info(f"Verrà usato un campione casuale di {sample_size} file validi per il vocabolario.")
        
        sampled_paths = random.sample(valid_midi_files_for_processing, sample_size)
        temp_midi_files_for_vocab = [str(p) for p in sampled_paths]

    try:
        MIDI_TOKENIZER = build_or_load_tokenizer_for_creator(
            midi_file_paths_for_vocab_build=temp_midi_files_for_vocab,
            force_build=args.force_tokenizer_build
        )
        tokenizer_successfully_initialized = MIDI_TOKENIZER is not None
    except Exception as e:
        logging.error(f"Impossibile inizializzare MIDI_TOKENIZER: {e}", exc_info=True)
        tokenizer_successfully_initialized = False

    if args.output_mode == "chunked" and not tokenizer_successfully_initialized:
        logging.error("Modalità Chunked selezionata, ma il tokenizer MIDI non è stato inizializzato. Uscita.")
        exit(1)

    # --- FASE 1: ELABORAZIONE PRINCIPALE SUI FILE VALIDI ---
    logging.info(f"--- Fase 1: Elaborazione principale di {len(valid_midi_files_for_processing)} file MIDI validi ---")
    all_processed_entries = []
    
    # La lista midi_file_paths ora contiene solo i file validi
    midi_file_paths = valid_midi_files_for_processing
    total_files_to_process = len(midi_file_paths)

    num_success = 0
    failure_summary = {}

    task_args_template = (
        Path, # Placeholder per il percorso
        BASE_DATA_DIR, 
        MIDI_INPUT_DIR, 
        OUTPUT_DIR, 
        BINARY_CHUNKS_DIR, 
        args.output_mode, 
        tokenizer_successfully_initialized, 
        args.transpose_enabled,  # Usa il nome corretto dell'argomento
        args.fast,
        args.piano_only,
        args.extract_genre,
        args.include_key,
        args.include_time_signature,
        args.include_tempo,
        args.include_velocity,
        args.include_instruments
    )
    
    # Usa la funzione di auto-calibrazione sui file validi per la massima stabilità
    num_workers = determine_optimal_workers(midi_file_paths, task_args_template)
    
    # Crea i task completi per l'elaborazione finale
    tasks = [(path,) + task_args_template[1:] for path in midi_file_paths]
    
    logging.info(f"Utilizzo di {num_workers} processi worker per l'elaborazione principale.")
    
    # (Il blocco per --delete_skipped_files rimane qui)
    if args.delete_skipped_files:
        # ... (stampa avviso)
        time.sleep(10)
    
    skipped_file_paths_to_delete = set()
    successful_file_paths = set()

    # Esegui l'elaborazione principale
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, 
        initializer=init_worker if args.output_mode == "chunked" else None, 
        initargs=(VOCAB_PATH, config.MIDI_TOKENIZER_STRATEGY) if args.output_mode == "chunked" else ()
    ) as executor:
        
        results_iterator = executor.map(process_single_file, tasks)
        
        progress_bar = tqdm(results_iterator, total=total_files_to_process, desc=f"Analisi MIDI ({args.output_mode})", unit="file")
        
        current_file_path_iterator = (task[0] for task in tasks)
        
        for file_path, file_results_list in zip(current_file_path_iterator, progress_bar):
            # ... (il resto del ciclo for per aggregare i risultati rimane identico a prima)
            is_successful = False
            if not file_results_list:
                failure_summary['unknown_empty_result'] = failure_summary.get('unknown_empty_result', 0) + 1
                is_successful = False
            else:
                if any(res.get('status', '').startswith('success') for res in file_results_list):
                    is_successful = True
                
                for result_item in file_results_list:
                    status = result_item.get('status', 'unknown_status')
                    if status.startswith('success'):
                        all_processed_entries.append(result_item)
                        num_success += 1
                    else:
                        failure_summary[status] = failure_summary.get(status, 0) + 1
            
            if is_successful:
                successful_file_paths.add(str(file_path))
            else:
                skipped_file_paths_to_delete.add(str(file_path))

            progress_bar.set_postfix_str(f"Valid items: {num_success}, Failures: {sum(failure_summary.values())}")
    
    logging.info(f"Conteggio finale voci valide (chunks/files): {num_success}")
    if failure_summary:
        logging.info("Dettaglio fallimenti/skip (durante l'elaborazione principale):")
        for reason, count in sorted(failure_summary.items()):
            logging.info(f"  '{reason}': {count}")

    if total_files_to_process > 0:
        logging.info(f"Numero totale di entry JSONL prodotte: {len(all_processed_entries)}")
    else:
        logging.info("Nessun file MIDI trovato o processato.")
    
    # --- FASE 1.5: Costruzione e Salvataggio Vocabolari Metadati ---
    if not all_processed_entries:
        logging.error("Nessuna entry valida prodotta, impossibile creare i vocabolari dei metadati.")
    else:
        logging.info("--- Fase 1.5: Costruzione Vocabolari Metadati ---")
        
        # Definisci i percorsi per i file di vocabolario
        METADATA_VOCAB_PATH = BASE_DATA_DIR / "metadata_vocab.json"
        METADATA_FREQUENCY_PATH = BASE_DATA_DIR / "metadata_frequency.json"

        # Raccogli tutti i dizionari di metadati dalle entry valide
        all_metadata_dicts = [
            entry['data']['metadata'] 
            for entry in all_processed_entries 
            if 'data' in entry and 'metadata' in entry['data']
        ]

        # Costruisci i vocabolari e calcola le frequenze in memoria
        token_to_id, frequency_counter = build_metadata_vocabs_and_frequencies(
            all_metadata_dicts, 
            min_freq=args.min_token_freq
        )

        # Salva il vocabolario standard
        save_metadata_vocab(token_to_id, METADATA_VOCAB_PATH)

        # Salva il nuovo vocabolario con le frequenze
        save_metadata_frequency(frequency_counter, METADATA_FREQUENCY_PATH)

        logging.info(f"Vocabolario dei metadati salvato in {METADATA_VOCAB_PATH}")
        logging.info(f"Vocabolario delle frequenze dei metadati salvato in {METADATA_FREQUENCY_PATH}")

    logging.info("--- Fase 2: Divisione del Dataset ---")
    if not all_processed_entries:
        logging.error("Dataset (lista di entries) è vuoto. Impossibile creare split.")
    else:
        random.shuffle(all_processed_entries)
        n = len(all_processed_entries)
        n_train = math.floor(TRAIN_SPLIT * n)
        n_val = math.floor(VALIDATION_SPLIT * n)

        train_data = all_processed_entries[:n_train]
        val_data = all_processed_entries[n_train : n_train + n_val]
        test_data = all_processed_entries[n_train + n_val :]

        logging.info(f"Dimensioni dataset (numero di entry JSONL): Totale={n}, Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")

        # Crea le directory di output se non esistono
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        BINARY_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        
        save_dataset_split(train_data, "train")
        logging.info(f"Salvato train.jsonl con {len(train_data)} voci.")
        save_dataset_split(val_data, "validation")
        logging.info(f"Salvato validation.jsonl con {len(val_data)} voci.")
        save_dataset_split(test_data, "test")
        logging.info(f"Salvato test.jsonl con {len(test_data)} voci.")
    
    # --- Fase di Eliminazione a Fine Script ---
    if args.delete_skipped_files or args.dry_run_delete:
        print("\n--- Fase di Pulizia dei File Sorgente ---")
        logging.info("--- Fase di Pulizia dei File Sorgente ---")
        
        # Sicurezza aggiuntiva: non eliminare se non sono stati prodotti file di split
        if not (OUTPUT_DIR / "train.jsonl").exists():
            print("ERRORE: File di training non creato. Annullamento dell'operazione di eliminazione per sicurezza.")
            logging.error("File di training non creato. Annullamento dell'operazione di eliminazione per sicurezza.")
        else:
            files_to_process = sorted(list(skipped_file_paths_to_delete))
            if not files_to_process:
                print("Nessun file scartato da eliminare.")
                logging.info("Nessun file scartato da eliminare.")
            else:
                if args.dry_run_delete:
                    print(f"[DRY RUN] Sono stati identificati {len(files_to_process)} file da eliminare:")
                    logging.info(f"[DRY RUN] Sono stati identificati {len(files_to_process)} file da eliminare:")
                    for f_path in files_to_process:
                        print(f"  - [DRY RUN] Verrebbe eliminato: {f_path}")
                        logging.info(f"  - [DRY RUN] Verrebbe eliminato: {f_path}")
                else: # --delete_skipped_files è attivo
                    print(f"PROCEDO CON L'ELIMINAZIONE di {len(files_to_process)} file scartati...")
                    logging.info(f"PROCEDO CON L'ELIMINAZIONE di {len(files_to_process)} file scartati...")
                    deleted_count = 0
                    for f_path in tqdm(files_to_process, desc="Eliminazione file", unit="file"):
                        try:
                            os.remove(f_path)
                            logging.info(f"Eliminato: {f_path}")
                            deleted_count += 1
                        except OSError as e:
                            print(f"Errore durante l'eliminazione di {f_path}: {e}")
                            logging.error(f"Errore durante l'eliminazione di {f_path}: {e}")
                    print(f"Operazione completata. Eliminati {deleted_count} file.")
                    logging.info(f"Operazione completata. Eliminati {deleted_count} file.")

    logging.info("--- Script Terminato ---")