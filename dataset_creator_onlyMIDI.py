import json
import random
import math
import logging
from pathlib import Path
import os
import time
from tqdm import tqdm
import concurrent.futures
import mido
import argparse # For command-line arguments

# --- Additions for Tokenization and Chunking ---
import miditok
from symusic import Score # For loading MIDI for miditok

# --- Configurazione Globale ---
BASE_DATA_DIR = Path("./mutopia_data")
MIDI_INPUT_DIR = BASE_DATA_DIR / "midi_files"
OUTPUT_DIR = BASE_DATA_DIR / "dataset_splits"
LOG_FILE = BASE_DATA_DIR / "dataset_processing_new.log"

ESSENTIAL_METADATA = ['time_signature', 'title', 'bpm_rounded', 'avg_velocity_rounded', 'velocity_range_rounded']
MIN_MIDO_DURATION_SECONDS = 10
MIN_NOTE_ON_MESSAGES = 50
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

global MIDI_TOKENIZER # Declare global for main thread and for workers to potentially inherit (depending on OS and start method)

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

# --- MIDI Tokenizer Specific Configurations (mirror from training.py or centralize) ---
# These should ideally be identical to what training.py will use
MIDI_TOKENIZER_STRATEGY = miditok.REMI
VOCAB_PATH = BASE_DATA_DIR / "midi_vocab.json" # Or a shared path
MIDI_PAD_TOKEN_NAME = "PAD_None"
MIDI_SOS_TOKEN_NAME = "SOS_None"
MIDI_EOS_TOKEN_NAME = "EOS_None"
MIDI_UNK_TOKEN_NAME = "UNK_None"

# Chunking parameters (mirror from training.py or centralize)
MAX_SEQ_LEN_MIDI_TOKENS = 512 # This is the max number of tokens *in a chunk*
MIN_CHUNK_LEN_MIDI_TOKENS = 50 # Min number of actual MIDI tokens in a chunk (excluding SOS/EOS)

# Piano programs if you implement piano_only filtering here
PIANO_PROGRAMS = list(range(0, 8))
# PROCESSING_MODE = "piano_only" # or "multi_instrument_stream" - for pre-filtering before chunking
PROCESSING_MODE = "multi_instrument_stream" # Default, as in training script

# Global tokenizer (initialized in main)
MIDI_TOKENIZER = None

def build_or_load_tokenizer_for_creator(midi_file_paths_for_vocab_build=None, force_build=False):
    """
    Builds or loads the MIDI tokenizer.
    Simplified version for dataset creator focusing on loading an existing vocab
    or building it if a representative set of MIDI files is provided.
    """
    special_tokens = [MIDI_PAD_TOKEN_NAME, MIDI_SOS_TOKEN_NAME, MIDI_EOS_TOKEN_NAME, MIDI_UNK_TOKEN_NAME]
    tokenizer_params = miditok.TokenizerConfig(
        special_tokens=special_tokens,
        use_programs=True, # Ensure these match training.py
        one_token_stream_for_programs=True,
        program_changes=True,
        use_chords=True,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_velocities=True,
    )
    
    if VOCAB_PATH.exists() and not force_build:
        logging.info(f"Loading MIDI tokenizer configuration from {VOCAB_PATH}")
        try:
            tokenizer = MIDI_TOKENIZER_STRATEGY(params=str(VOCAB_PATH))
            logging.info(f"Tokenizer loaded successfully from {VOCAB_PATH}")
        except Exception as e:
            logging.error(f"Error loading tokenizer params from {VOCAB_PATH}. Error: {e}", exc_info=True)
            logging.info("Attempting to build tokenizer from scratch (if MIDI files provided).")
            if not midi_file_paths_for_vocab_build:
                logging.error("Cannot build tokenizer: no MIDI files for vocab build provided.")
                raise
            tokenizer = MIDI_TOKENIZER_STRATEGY(tokenizer_config=tokenizer_params)
            # Training part is optional here if vocab is pre-built by training script first
            # but good to have if this script is run standalone to create vocab
            if hasattr(tokenizer, 'train') and MIDI_TOKENIZER_STRATEGY != miditok.TSD and MIDI_TOKENIZER_STRATEGY != miditok.REMI: # REMI/TSD don't train BPE
                logging.info(f"Training tokenizer with {len(midi_file_paths_for_vocab_build)} files for vocab.")
                tokenizer.train(vocab_size=50000, files_paths=midi_file_paths_for_vocab_build) # Example target size
                VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
                tokenizer.save(str(VOCAB_PATH))
                logging.info(f"Tokenizer trained and saved to {VOCAB_PATH}")
            else:
                 # For REMI, TSD, or if no .train method, the vocab is implicitly defined by the config.
                 # We still save the config so it can be loaded with `params=path`.
                VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
                tokenizer.save(str(VOCAB_PATH)) # Saves the configuration including special tokens
                logging.info(f"Tokenizer config (non-BPE or pre-defined) saved to {VOCAB_PATH}")


    else: # Build from scratch
        if not midi_file_paths_for_vocab_build:
            logging.error("Cannot build tokenizer: vocab path doesn't exist and no MIDI files for vocab build provided.")
            raise FileNotFoundError("VOCAB_PATH not found and no files to build it.")
            
        logging.info("Creating new MIDI tokenizer configuration...")
        tokenizer = MIDI_TOKENIZER_STRATEGY(tokenizer_config=tokenizer_params)
        if hasattr(tokenizer, 'train') and MIDI_TOKENIZER_STRATEGY != miditok.TSD and MIDI_TOKENIZER_STRATEGY != miditok.REMI:
            logging.info(f"Training tokenizer with {len(midi_file_paths_for_vocab_build)} files for vocab.")
            tokenizer.train(vocab_size=50000, files_paths=midi_file_paths_for_vocab_build) # Example
        
        VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(VOCAB_PATH))
        logging.info(f"New tokenizer created, trained (if applicable), and saved to {VOCAB_PATH}")

    # Verify special tokens
    try:
        assert tokenizer[MIDI_PAD_TOKEN_NAME] is not None
        assert tokenizer[MIDI_SOS_TOKEN_NAME] is not None
        assert tokenizer[MIDI_EOS_TOKEN_NAME] is not None
    except AssertionError:
        logging.error("CRITICAL: One or more special MIDI tokens not found in tokenizer after load/build.")
        raise ValueError("Special MIDI tokens missing in tokenizer.")
    logging.info(f"MIDI Tokenizer ready. Vocab size: {len(tokenizer)}")
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
            'key_signature_declared': key_signature,
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

# --- Funzione Worker per il Multiprocessing ---
def process_single_file(args_tuple):
    midi_file_path, output_mode, tokenizer_present = args_tuple # Unpack
    
    # Ensure MIDI_TOKENIZER is available in the worker process if needed
    global MIDI_TOKENIZER
    if output_mode == "chunked" and not tokenizer_present:
        # This is a fallback if the global tokenizer wasn't properly shared or initialized.
        # For ProcessPoolExecutor, globals aren't directly shared.
        # It's better to pass the tokenizer or ensure it's initialized per process
        # or use a ThreadPoolExecutor if tokenizer is thread-safe and GIL is not an issue.
        # For simplicity here, we rely on it being initialized in main if chunked.
        # A more robust way is to pass the vocab path and re-init tokenizer here.
        logging.error(f"Tokenizer requested for chunking {midi_file_path.name} but not available in worker.")
        return [{'status': 'skipped_tokenizer_unavailable', 'filename': midi_file_path.name}]


    mido_check_result = quick_midi_check_mido(str(midi_file_path))

    if not mido_check_result['passed']:
        return [{'status': f"skipped_{mido_check_result['reason']}",
                'filename': midi_file_path.name,
                'detail': mido_check_result.get('detail', '')}]

    final_metadata = {}
    final_metadata['key'] = mido_check_result.get('key_signature_declared')
    final_metadata['time_signature'] = mido_check_result.get('time_signature')
    final_metadata['title'] = midi_file_path.stem
    final_metadata['mido_duration_seconds'] = mido_check_result.get('duration_seconds')
    final_metadata['mido_note_count'] = mido_check_result.get('note_count')
    final_metadata['midi_instruments'] = mido_check_result.get('midi_instruments', [])
    final_metadata['bpm_rounded'] = mido_check_result.get('bpm_rounded')
    final_metadata['avg_velocity_rounded'] = mido_check_result.get('avg_velocity_rounded')
    final_metadata['velocity_range_rounded'] = mido_check_result.get('velocity_range_rounded')

    missing_essentials = [k for k in ESSENTIAL_METADATA if final_metadata.get(k) is None]
    if missing_essentials:
        return [{'status': 'skipped_mido_missing_metadata',
                'filename': midi_file_path.name,
                'missing': missing_essentials}]

    try:
        relative_path = str(midi_file_path.relative_to(BASE_DATA_DIR))
    except ValueError:
        relative_path = str(midi_file_path.relative_to(MIDI_INPUT_DIR))
    final_metadata['midi_relative_path'] = relative_path
    
    # --- Classic Mode: Return as is ---
    if output_mode == "classic":
        return [{'status': 'success', 
                 'data': {'id': midi_file_path.stem, 
                          'metadata': final_metadata,
                          # No token_ids field for classic mode, training script handles tokenization
                         }}]

    # --- Chunked Mode: Tokenize and Chunk ---
    elif output_mode == "chunked":
        if MIDI_TOKENIZER is None: # Should have been initialized in main
             logging.error(f"MIDI_TOKENIZER is None in worker for {midi_file_path}, cannot proceed with chunking.")
             return [{'status': 'skipped_tokenizer_not_init', 'filename': midi_file_path.name}]
        try:
            score = Score(str(midi_file_path)) # symusic.Score

            if PROCESSING_MODE == "piano_only":
                piano_tracks_present = [track for track in score.tracks if track.program in PIANO_PROGRAMS]
                if not piano_tracks_present:
                    return [{'status': 'skipped_no_piano_tracks', 'filename': midi_file_path.name}]
                score.tracks = piano_tracks_present
                if len(score.tracks) == 0: # Should be caught by above, but defensive
                    return [{'status': 'skipped_no_piano_tracks_after_filter', 'filename': midi_file_path.name}]

            midi_tokens_output = MIDI_TOKENIZER(score) # miditok

            raw_midi_ids = []
            if hasattr(midi_tokens_output, 'ids') and isinstance(midi_tokens_output.ids, list):
                if all(isinstance(sublist, list) for sublist in midi_tokens_output.ids):
                    raw_midi_ids = [item for sublist in midi_tokens_output.ids for item in sublist]
                elif all(isinstance(item, int) for item in midi_tokens_output.ids):
                    raw_midi_ids = midi_tokens_output.ids
                else:
                    return [{'status': 'skipped_midi_tokenization_unexpected_format', 'filename': midi_file_path.name}]
            else:
                 return [{'status': 'skipped_midi_tokenization_invalid_output', 'filename': midi_file_path.name}]

            if not raw_midi_ids or len(raw_midi_ids) < MIN_CHUNK_LEN_MIDI_TOKENS:
                return [{'status': 'skipped_too_short_after_tokenization', 'filename': midi_file_path.name, 'token_count': len(raw_midi_ids)}]

            chunked_samples = []
            # -2 for SOS and EOS to be added by training script's Dataset
            effective_chunk_len_for_data = MAX_SEQ_LEN_MIDI_TOKENS - 2 
            
            if effective_chunk_len_for_data < MIN_CHUNK_LEN_MIDI_TOKENS:
                 logging.error(f"Configuration error: effective_chunk_len_for_data ({effective_chunk_len_for_data}) "
                               f"is less than MIN_CHUNK_LEN_MIDI_TOKENS ({MIN_CHUNK_LEN_MIDI_TOKENS}). "
                               f"Increase MAX_SEQ_LEN_MIDI_TOKENS or decrease MIN_CHUNK_LEN_MIDI_TOKENS.")
                 # Skip this file due to config error for chunking
                 return [{'status': 'skipped_chunk_config_error', 'filename': midi_file_path.name}]


            num_file_chunks = 0
            for i in range(0, len(raw_midi_ids), effective_chunk_len_for_data):
                chunk_token_ids = raw_midi_ids[i : i + effective_chunk_len_for_data]
                if len(chunk_token_ids) < MIN_CHUNK_LEN_MIDI_TOKENS:
                    if num_file_chunks == 0: # First chunk itself is too short
                         # This specific sub-status helps to distinguish from files that were short overall
                         return [{'status': 'skipped_first_chunk_too_short', 'filename': midi_file_path.name, 'token_count': len(chunk_token_ids)}]
                    break # Stop if remaining part is too short for a valid chunk

                chunk_id = f"{midi_file_path.stem}_chunk{num_file_chunks}"
                chunk_data = {
                    'id': chunk_id,
                    'metadata': final_metadata.copy(), # Copy metadata for each chunk
                    'token_ids': chunk_token_ids # Store the raw token IDs for the chunk
                }
                # Add reference to original file if needed
                chunk_data['metadata']['original_file_id'] = midi_file_path.stem
                chunked_samples.append({'status': 'success_chunked', 'data': chunk_data})
                num_file_chunks +=1
            
            if not chunked_samples: # No valid chunks produced
                return [{'status': 'skipped_no_valid_chunks_created', 'filename': midi_file_path.name, 'total_tokens': len(raw_midi_ids)}]

            return chunked_samples

        except miditok.MIDITokenizerException as e:
            logging.warning(f"miditok error for {midi_file_path.name}: {e}")
            return [{'status': 'skipped_miditok_error', 'filename': midi_file_path.name, 'detail': str(e)}]
        except FileNotFoundError: # If symusic can't find/read
             logging.warning(f"File not found by symusic for tokenization: {midi_file_path.name}")
             return [{'status': 'skipped_symusic_file_not_found', 'filename': midi_file_path.name}]
        except Exception as e:
            logging.error(f"Unexpected error during chunking for {midi_file_path.name}: {e}", exc_info=True)
            return [{'status': 'skipped_chunking_unexpected_error', 'filename': midi_file_path.name, 'detail': str(e)}]
    else:
        logging.error(f"Unknown output mode: {output_mode}")
        return [{'status': 'skipped_unknown_output_mode', 'filename': midi_file_path.name}]

def save_dataset_split(data, filename):
    output_path = OUTPUT_DIR / f"{filename}.jsonl"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry_wrapper in data: # data is now a list of items from process_single_file
                 # process_single_file now returns a list of dicts, each dict has 'status' and 'data'
                 # We only want to save successful ones, and data can be a list of chunks
                if isinstance(entry_wrapper, dict) and entry_wrapper.get('status', '').startswith('success'):
                    f.write(json.dumps(entry_wrapper['data']) + '\n')
                # If entry_wrapper is a list (from chunked success), iterate
                elif isinstance(entry_wrapper, list):
                     for item in entry_wrapper:
                         if isinstance(item, dict) and item.get('status', '').startswith('success'):
                            f.write(json.dumps(item['data']) + '\n')

    except TypeError as e:
         logging.error(f"TypeError during JSON writing in {output_path}: {e}. Check data structure.")
    except Exception as e:
        logging.error(f"Error saving {output_path}: {e}")

# For ProcessPoolExecutor, the tokenizer needs to be picklable or initialized in each worker.
# A simple way for non-picklable objects is to initialize them globally if the worker inherits them,
# or pass necessary config to re-initialize. For miditok, loading from path is often fine.
def init_worker(vocab_p, midi_tok_strat_class):
    global MIDI_TOKENIZER
    logging.info(f"Worker {os.getpid()} initializing tokenizer.")
    try:
        # Ensure this path and strategy are correct and accessible
        MIDI_TOKENIZER = midi_tok_strat_class(params=str(vocab_p)) 
        logging.info(f"Worker {os.getpid()} tokenizer initialized with vocab size: {len(MIDI_TOKENIZER)}")
    except Exception as e:
        logging.error(f"Worker {os.getpid()} failed to initialize tokenizer: {e}", exc_info=True)
        MIDI_TOKENIZER = None # Explicitly set to None on failure

# --- Esecuzione principale ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create MIDI dataset splits.")
    parser.add_argument(
        "--output_mode",
        type=str,
        choices=["classic", "chunked"],
        default="classic",
        help="Output mode: 'classic' (one entry per file) or 'chunked' (one entry per MIDI chunk)."
    )
    parser.add_argument(
        "--force_tokenizer_build",
        action="store_true",
        help="Force rebuild of MIDI tokenizer vocabulary even if it exists."
    )
    args = parser.parse_args()

    logging.info(f"--- Dataset Creation Script --- Output Mode: {args.output_mode} ---")
    
    # Placeholder for MIDI files to build vocab if needed.
    # Collect these before starting the parallel processing.
    temp_midi_files_for_vocab = []
    if not VOCAB_PATH.exists() or args.force_tokenizer_build:
        logging.info("Collecting MIDI files to build tokenizer vocabulary...")
        # Limiting the number of files for vocab building for speed, adjust as needed
        all_potential_midi_files = list(MIDI_INPUT_DIR.rglob("*.mid")) + list(MIDI_INPUT_DIR.rglob("*.midi"))
        temp_midi_files_for_vocab = [str(p) for p in random.sample(all_potential_midi_files, min(len(all_potential_midi_files), 500))] # Sample 500 files
        if not temp_midi_files_for_vocab:
            logging.error("No MIDI files found to build tokenizer vocabulary. Exiting.")
            exit(1)

    try:
        # Initialize tokenizer in the main process. If using 'fork' (default on Linux),
        # workers might inherit. For 'spawn' or 'forkserver', use initializer.
        MIDI_TOKENIZER = build_or_load_tokenizer_for_creator(
            midi_file_paths_for_vocab_build=temp_midi_files_for_vocab if temp_midi_files_for_vocab else None,
            force_build=args.force_tokenizer_build
        )
        tokenizer_successfully_initialized = MIDI_TOKENIZER is not None
    except Exception as e:
        logging.error(f"Failed to initialize MIDI_TOKENIZER in main process: {e}", exc_info=True)
        MIDI_TOKENIZER = None # Ensure it's None if failed
        tokenizer_successfully_initialized = False


    if args.output_mode == "chunked" and not tokenizer_successfully_initialized:
        logging.error("Chunked mode selected, but MIDI tokenizer could not be initialized. Exiting.")
        exit(1)

    logging.info(f"--- Fase 1: Elaborazione File MIDI da {MIDI_INPUT_DIR} ---")
    all_processed_entries = [] # This will store all individual entries (could be multiple per file in chunked mode)
    total_files_found = 0

    if not MIDI_INPUT_DIR.exists() or not MIDI_INPUT_DIR.is_dir():
        logging.error(f"Directory MIDI di input non trovata o non valida: {MIDI_INPUT_DIR}")
    else:
        midi_file_paths = [p for p in MIDI_INPUT_DIR.rglob("*.mid")] + \
                          [p for p in MIDI_INPUT_DIR.rglob("*.midi")]
        
        total_files_found = len(midi_file_paths)
        logging.info(f"Trovati {total_files_found} file MIDI in {MIDI_INPUT_DIR} da elaborare.")

        if not midi_file_paths:
             logging.warning("Nessun file MIDI trovato.")
        else:
            # Simplified skip counts as process_single_file returns a list of dicts with status
            # We'll count unique failure reasons later
            num_success = 0
            failure_summary = {} # To count different failure reasons

            # Prepare arguments for map function
            # Pass tokenizer_successfully_initialized to avoid re-checking MIDI_TOKENIZER in every call from main thread perspective
            tasks = [(path, args.output_mode, tokenizer_successfully_initialized) for path in midi_file_paths]

            num_workers = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
            logging.info(f"Utilizzo di {num_workers} processi worker.")

            # initializer and initargs are used to set up each worker process
            # This is important if MIDI_TOKENIZER is not easily picklable or needs specific setup per process
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers, 
                initializer=init_worker if args.output_mode == "chunked" else None, 
                initargs=(VOCAB_PATH, MIDI_TOKENIZER_STRATEGY) if args.output_mode == "chunked" else ()
            ) as executor:
                
                # executor.map will pass each item from 'tasks' to 'process_single_file'
                results_iterator = executor.map(process_single_file, tasks)
                
                progress_bar = tqdm(results_iterator, total=total_files_found, desc=f"Analisi MIDI ({args.output_mode})", unit="file")
                
                for file_results_list in progress_bar: # file_results_list is a list of dicts from process_single_file
                    if not file_results_list: # Should not happen if process_single_file always returns a list
                        failure_summary['unknown_empty_result'] = failure_summary.get('unknown_empty_result', 0) + 1
                        continue
                    
                    for result_item in file_results_list: # Iterate through chunks or single result
                        status = result_item.get('status', 'unknown_status')
                        if status.startswith('success'):
                            all_processed_entries.append(result_item) # Append the dict containing 'status' and 'data'
                            num_success +=1 # Counts successful chunks or files
                        else:
                            failure_summary[status] = failure_summary.get(status, 0) + 1
                    
                    # Update progress bar postfix (optional, can be verbose for chunked)
                    # This shows number of successful *items* (chunks or files)
                    if args.output_mode == "classic" or num_success % 100 == 0 :
                        progress_bar.set_postfix_str(f"Valid items: {num_success}, Failures: {sum(failure_summary.values())}")
            
            logging.info(f"Conteggio finale voci valide (chunks/files): {num_success}")
            if failure_summary:
                logging.info("Dettaglio fallimenti/skip:")
                for reason, count in sorted(failure_summary.items()):
                    logging.info(f"  '{reason}': {count}")

    if total_files_found > 0:
        # This percentage is tricky for chunked mode as one file can produce many entries or none
        logging.info(f"Numero totale di entry JSONL prodotte: {len(all_processed_entries)}")
    else:
        logging.info("Nessun file MIDI trovato o processato.")

    logging.info("--- Fase 2: Divisione del Dataset ---")
    if not all_processed_entries:
        logging.error("Dataset (lista di entries) è vuoto. Impossibile creare split.")
    else:
        random.shuffle(all_processed_entries) # Shuffle all entries (chunks or files)
        n = len(all_processed_entries)
        n_train = math.floor(TRAIN_SPLIT * n)
        n_val = math.floor(VALIDATION_SPLIT * n)

        train_data = all_processed_entries[:n_train]
        val_data = all_processed_entries[n_train : n_train + n_val]
        test_data = all_processed_entries[n_train + n_val :]

        logging.info(f"Dimensioni dataset (numero di entry JSONL): Totale={n}, Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_dataset_split(train_data, "train")
        logging.info(f"Salvato train.jsonl con {len(train_data)} voci.")
        save_dataset_split(val_data, "validation")
        logging.info(f"Salvato validation.jsonl con {len(val_data)} voci.")
        save_dataset_split(test_data, "test")
        logging.info(f"Salvato test.jsonl con {len(test_data)} voci.")

    logging.info("--- Script Terminato ---")