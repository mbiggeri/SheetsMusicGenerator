import json
import random
import math
import logging
from pathlib import Path
import os
import time # Lo manteniamo
from tqdm import tqdm
import concurrent.futures
import mido

# --- Configurazione Globale ---
BASE_DATA_DIR = Path("./mutopia_data") #
MIDI_INPUT_DIR = BASE_DATA_DIR / "midi_files" #
OUTPUT_DIR = BASE_DATA_DIR / "dataset_splits" #
LOG_FILE = BASE_DATA_DIR / "dataset_processing.log" #

# Metadati essenziali che devono essere presenti dopo l'analisi con mido
# Rimuoviamo 'style' se non lo usiamo più o se viene derivato diversamente
ESSENTIAL_METADATA = ['time_signature', 'title', 'bpm_rounded', 'avg_velocity_rounded', 'velocity_range_rounded'] # Aggiunti nuovi campi derivati
# DEFAULT_STYLE_PLACEHOLDER = "unknown_style" # Rimosso

# Soglie per il filtraggio con mido
MIN_MIDO_DURATION_SECONDS = 10 #
MIN_NOTE_ON_MESSAGES = 50 #

TRAIN_SPLIT = 0.8 #
VALIDATION_SPLIT = 0.1 #

# Configurazione Logging (invariata)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
) #

GM_INSTRUMENT_MAP = {
    # ... (GM_INSTRUMENT_MAP completa come fornita precedentemente) ...
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
} #

# --- Funzioni di Analisi (solo mido) ---

def quick_midi_check_mido(file_path_str):
    """
    Esegue un controllo preliminare veloce usando mido.
    Estrae durata, conteggio note, time signature, key signature, strumenti, BPM e dinamiche.
    """
    try:
        mid = mido.MidiFile(file_path_str) #
        duration = mid.length #
        if duration < MIN_MIDO_DURATION_SECONDS: #
            return {'passed': False, 'reason': 'mido_too_short', 'duration': duration} #

        note_on_count = 0 #
        velocities = []
        time_signature = None #
        key_signature = None #
        tempo_microseconds = None # Per il primo tempo trovato
        found_instruments = set() #

        for i, track in enumerate(mid.tracks): #
            for msg in track: #
                if msg.type == 'note_on' and msg.velocity > 0: #
                    note_on_count += 1 #
                    velocities.append(msg.velocity)
                
                if not time_signature and msg.is_meta and msg.type == 'time_signature': #
                    time_signature = f"{msg.numerator}/{msg.denominator}" #
                if not key_signature and msg.is_meta and msg.type == 'key_signature': #
                    key_signature = msg.key #
                
                if tempo_microseconds is None and msg.is_meta and msg.type == 'set_tempo':
                    tempo_microseconds = msg.tempo
                
                if msg.is_meta and msg.type == 'instrument_name': #
                    try:
                        instrument_name_meta = msg.name.strip() #
                        found_instruments.add(instrument_name_meta) #
                    except: #
                        pass #
                
                if msg.type == 'program_change': #
                    instrument_gm_name = GM_INSTRUMENT_MAP.get(msg.program) #
                    if instrument_gm_name: #
                        found_instruments.add(instrument_gm_name) #

        if note_on_count < MIN_NOTE_ON_MESSAGES: #
            return {'passed': False, 'reason': 'mido_too_few_notes', 'note_count': note_on_count, 'duration': duration} #

        bpm_rounded = None
        if tempo_microseconds:
            bpm = 60_000_000 / tempo_microseconds
            bpm_rounded = round(bpm / 5) * 5 # Arrotonda al multiplo di 5 più vicino

        avg_velocity_rounded = None
        velocity_range_rounded = None
        if velocities:
            avg_vel = sum(velocities) / len(velocities)
            vel_range = max(velocities) - min(velocities)
            avg_velocity_rounded = round(avg_vel / 5) * 5 # Arrotonda al multiplo di 5
            velocity_range_rounded = round(vel_range / 5) * 5 # Arrotonda al multiplo di 5
            
        return {
            'passed': True, #
            'reason': 'mido_passed', #
            'duration_seconds': duration, #
            'note_count': note_on_count, #
            'time_signature': time_signature, #
            'key_signature_declared': key_signature, #
            'midi_instruments': sorted(list(found_instruments)), #
            'bpm_rounded': bpm_rounded, # BPM arrotondato
            'avg_velocity_rounded': avg_velocity_rounded, # Velocity media arrotondata
            'velocity_range_rounded': velocity_range_rounded # Range di velocity arrotondato
        }
    except mido.MIDIOpenError: #
        return {'passed': False, 'reason': 'mido_open_error', 'detail': 'Cannot open MIDI file'} #
    except IndexError: #
        return {'passed': False, 'reason': 'mido_index_error', 'detail': 'Error processing MIDI tracks/messages'} #
    except Exception as e: #
        return {'passed': False, 'reason': 'mido_parse_error', 'detail': str(e)} #

# --- Funzione Worker per il Multiprocessing ---
def process_single_file(midi_file_path: Path): #
    mido_check_result = quick_midi_check_mido(str(midi_file_path)) #

    if not mido_check_result['passed']: #
        return {'status': f"skipped_{mido_check_result['reason']}", #
                'filename': midi_file_path.name, #
                'detail': mido_check_result.get('detail', '')} #

    final_metadata = {}
    final_metadata['key'] = mido_check_result.get('key_signature_declared') #
    final_metadata['time_signature'] = mido_check_result.get('time_signature') #
    final_metadata['title'] = midi_file_path.stem #
    # final_metadata['style'] = DEFAULT_STYLE_PLACEHOLDER # Rimosso
    final_metadata['mido_duration_seconds'] = mido_check_result.get('duration_seconds') #
    final_metadata['mido_note_count'] = mido_check_result.get('note_count') #
    final_metadata['midi_instruments'] = mido_check_result.get('midi_instruments', []) #
    
    # Aggiungi i nuovi campi derivati
    final_metadata['bpm_rounded'] = mido_check_result.get('bpm_rounded')
    final_metadata['avg_velocity_rounded'] = mido_check_result.get('avg_velocity_rounded')
    final_metadata['velocity_range_rounded'] = mido_check_result.get('velocity_range_rounded')


    missing_essentials = [k for k in ESSENTIAL_METADATA if final_metadata.get(k) is None] # Controllo più robusto per None
    if missing_essentials: #
        return {'status': 'skipped_mido_missing_metadata', #
                'filename': midi_file_path.name, #
                'missing': missing_essentials} #

    try:
        relative_path = str(midi_file_path.relative_to(BASE_DATA_DIR)) #
    except ValueError: #
        logging.warning(f"File {midi_file_path} non trovato sotto BASE_DATA_DIR {BASE_DATA_DIR} come atteso. " #
                        f"Il percorso relativo sarà calcolato rispetto a MIDI_INPUT_DIR {MIDI_INPUT_DIR}.") #
        relative_path = str(midi_file_path.relative_to(MIDI_INPUT_DIR)) #

    final_metadata['midi_relative_path'] = relative_path #
    return {'status': 'success', 'data': {'id': midi_file_path.stem, 'metadata': final_metadata}} #

# save_dataset_split e __main__ rimangono sostanzialmente invariati come nello script originale
# ma assicurati che il logging e la gestione degli errori siano coerenti.

def save_dataset_split(data, filename): #
    output_path = OUTPUT_DIR / f"{filename}.jsonl" #
    try:
        with open(output_path, 'w', encoding='utf-8') as f: #
            for entry in data: #
                f.write(json.dumps(entry) + '\n') #
    except TypeError as e: #
         logging.error(f"Errore di tipo durante la scrittura JSON in {output_path}: {e}. Controlla i dati.") #
    except Exception as e: #
        logging.error(f"Errore nel salvataggio di {output_path}: {e}") #

if __name__ == "__main__": #
    logging.info(f"--- Fase 1: Elaborazione File MIDI da {MIDI_INPUT_DIR} con solo Mido (senza Music21) ---") #
    raw_dataset = [] #
    total_files_found = 0 #

    if not MIDI_INPUT_DIR.exists() or not MIDI_INPUT_DIR.is_dir(): #
        logging.error(f"Directory MIDI di input non trovata o non valida: {MIDI_INPUT_DIR}") #
    else:
        midi_file_paths = [p for p in MIDI_INPUT_DIR.rglob("*.mid")] + \
                          [p for p in MIDI_INPUT_DIR.rglob("*.midi")] #
        
        total_files_found = len(midi_file_paths) #
        logging.info(f"Trovati {total_files_found} file MIDI in {MIDI_INPUT_DIR} da elaborare.") #

        if not midi_file_paths: #
             logging.warning("Nessun file MIDI trovato.") #
        else:
            skip_counts = {
                'skipped_mido_too_short': 0, #
                'skipped_mido_too_few_notes': 0, #
                'skipped_mido_parse_error': 0, #
                'skipped_mido_missing_metadata': 0, #
                'skipped_mido_open_error': 0, # Aggiunto per tracciare errori di apertura
                'skipped_mido_index_error': 0, # Aggiunto per tracciare errori di indice
                'unknown_skip_or_error': 0 #
            }
            
            num_workers = max(1, os.cpu_count() - 1 if os.cpu_count() else 1) #
            logging.info(f"Utilizzo di {num_workers} processi worker.") #

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor: #
                results_iterator = executor.map(process_single_file, midi_file_paths) #
                
                progress_bar = tqdm(results_iterator, total=total_files_found, desc="Analisi MIDI (Mido)", unit="file", mininterval=0.5, miniters=max(1, total_files_found // 1000)) #
                
                for result in progress_bar: #
                    if result is None: #
                        skip_counts['unknown_skip_or_error'] += 1 #
                        continue #

                    status = result.get('status') #
                    if status == 'success': #
                        raw_dataset.append(result['data']) #
                    elif status in skip_counts: #
                        skip_counts[status] += 1 #
                        # Aggiorna la progress bar meno frequentemente o con un formato più conciso se necessario
                        if sum(skip_counts.values()) % 100 == 0 : #
                             progress_bar.set_postfix_str( #
                                 f"Validi: {len(raw_dataset)}, " #
                                 f"ErrOpen: {skip_counts['skipped_mido_open_error']}, "
                                 f"Corti: {skip_counts['skipped_mido_too_short']}, " #
                                 f"PocheNote: {skip_counts['skipped_mido_too_few_notes']}, " #
                                 f"NoMeta: {skip_counts['skipped_mido_missing_metadata']}" #
                                 # f"ErrParse: {skip_counts['skipped_mido_parse_error']}" # Potrebbe essere troppo lungo
                             ) #
                    else: #
                        skip_counts['unknown_skip_or_error'] += 1 #
            
            logging.info(f"Conteggio finale voci valide: {len(raw_dataset)}") #
            for reason, count in skip_counts.items(): #
                if count > 0: #
                    logging.info(f"File saltati per '{reason}': {count}") #

    if total_files_found > 0: #
        logging.info(f"Percentuale file validi: { (len(raw_dataset) / total_files_found) * 100 :.2f}%") #
    else: #
        logging.info("Nessun file MIDI trovato o processato.") #

    logging.info("--- Fase 2: Divisione del Dataset ---") #
    if not raw_dataset: #
        logging.error("Dataset grezzo è vuoto. Impossibile creare split.") #
    else:
        random.shuffle(raw_dataset) #
        n = len(raw_dataset) #
        n_train = math.floor(TRAIN_SPLIT * n) #
        n_val = math.floor(VALIDATION_SPLIT * n) #

        train_data = raw_dataset[:n_train] #
        val_data = raw_dataset[n_train : n_train + n_val] #
        test_data = raw_dataset[n_train + n_val :] #

        logging.info(f"Dimensioni dataset: Totale={n}, Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}") #

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True) #
        save_dataset_split(train_data, "train") #
        tqdm.write(f"Salvato train.jsonl con {len(train_data)} voci.") #
        save_dataset_split(val_data, "validation") #
        tqdm.write(f"Salvato validation.jsonl con {len(val_data)} voci.") #
        save_dataset_split(test_data, "test") #
        tqdm.write(f"Salvato test.jsonl con {len(test_data)} voci.") #

    logging.info("--- Script Terminato ---") #