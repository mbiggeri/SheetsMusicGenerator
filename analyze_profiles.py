# analyze_profiles.py (versione corretta e robusta)

import argparse
import json
import collections
from pathlib import Path
import re
from tqdm import tqdm
import logging

# --- USAGE: python analyze_profiles.py --input_dir /path/to/jsonl_files --output_file /path/to/output/metadata_profiles.json --min_support 50 --max_profiles 100
# --- EXAMPLE: python analyze_profiles.py --input_dir C:\Users\Michael\Desktop\MusicDatasets\Datasets\PianoDataset\dataset_splits --output_file C:\Users\Michael\Desktop\MusicDatasets\Datasets\PianoDataset\dataset_splits --min_support 150 --max_profiles 100

# --- CONFIGURAZIONE ---
PROCESSING_MODE = "piano_only" # or "multi_instrument_stream"

# --- FUNZIONE DI TOKENIZZAZIONE (adattata da dataset_creator.py) ---
def tokenize_metadata(metadata_dict):
    """
    Converte un dizionario di metadati in una lista di token stringa.
    Questa funzione deve essere coerente con quella in dataset_creator.py.
    """
    tokens = []
    key_to_tokenize_str = None

    if PROCESSING_MODE == "piano_only" and 'transposed_to_key' in metadata_dict and metadata_dict['transposed_to_key']:
        raw_transposed_info = metadata_dict['transposed_to_key']
        if "C major / a minor" in raw_transposed_info:
            key_to_tokenize_str = "Target_Cmaj_Amin"
        else:
            temp_key = str(raw_transposed_info).replace(' ', '_').replace('/', '_').replace('#','sharp')
            key_to_tokenize_str = re.sub(r'[^a-zA-Z0-9_]', '', temp_key)
            if not key_to_tokenize_str:
                key_to_tokenize_str = "Unknown_Transposed_Key"
    
    if not key_to_tokenize_str:
        key_source = metadata_dict.get('key') or \
                     metadata_dict.get('music21_detected_key') or \
                     metadata_dict.get('mido_declared_key_signature')
        if key_source:
             key_to_tokenize_str = str(key_source)

    if key_to_tokenize_str:
        clean_key_token = str(key_to_tokenize_str).replace(' ', '_').replace('#', 'sharp')
        clean_key_token = re.sub(r'[^a-zA-Z0-9_]', '', clean_key_token)
        if clean_key_token:
            tokens.append(f"Key={clean_key_token}")

    if 'time_signature' in metadata_dict and metadata_dict['time_signature']:
        tokens.append(f"TimeSig={metadata_dict['time_signature']}")

    if 'bpm_rounded' in metadata_dict and metadata_dict['bpm_rounded'] is not None:
        bpm = metadata_dict['bpm_rounded']
        if bpm <= 60: token_bpm = "Tempo_VerySlow"
        elif bpm <= 76: token_bpm = "Tempo_Slow"
        elif bpm <= 108: token_bpm = "Tempo_Moderate"
        elif bpm <= 132: token_bpm = "Tempo_Fast"
        elif bpm <= 168: token_bpm = "Tempo_VeryFast"
        else: token_bpm = "Tempo_ExtremelyFast"
        tokens.append(token_bpm)

    if 'avg_velocity_rounded' in metadata_dict and metadata_dict['avg_velocity_rounded'] is not None:
        avg_vel = metadata_dict['avg_velocity_rounded']
        if avg_vel <= 35: token_avg_vel = "AvgVel_VeryLow"
        elif avg_vel <= 60: token_avg_vel = "AvgVel_Low"
        elif avg_vel <= 85: token_avg_vel = "AvgVel_Medium"
        elif avg_vel <= 110: token_avg_vel = "AvgVel_High"
        else: token_avg_vel = "AvgVel_VeryHigh"
        tokens.append(token_avg_vel)

    if 'velocity_range_rounded' in metadata_dict and metadata_dict['velocity_range_rounded'] is not None:
        vel_range = metadata_dict['velocity_range_rounded']
        if vel_range <= 15: token_vel_range = "VelRange_Narrow"
        elif vel_range <= 40: token_vel_range = "VelRange_Medium"
        elif vel_range <= 70: token_vel_range = "VelRange_Wide"
        else: token_vel_range = "VelRange_VeryWide"
        tokens.append(token_vel_range)
    
    num_instruments = len(metadata_dict.get('midi_instruments', []))
    if num_instruments == 1: token_num_inst = "NumInst_Solo"
    elif num_instruments == 2: token_num_inst = "NumInst_Duet"
    elif num_instruments <= 4: token_num_inst = "NumInst_SmallChamber"
    elif num_instruments <= 8: token_num_inst = "NumInst_MediumEnsemble"
    else: token_num_inst = "NumInst_LargeEnsemble"
    if num_instruments > 0:
        tokens.append(token_num_inst)

    if 'midi_instruments' in metadata_dict and isinstance(metadata_dict['midi_instruments'], list):
        for instrument_name in metadata_dict['midi_instruments']:
            if instrument_name and isinstance(instrument_name, str):
                clean_instrument_name = instrument_name.replace(' ', '_').replace('(', '').replace(')', '').replace('#','sharp')
                clean_instrument_name = re.sub(r'[^a-zA-Z0-9_]', '', clean_instrument_name)
                if clean_instrument_name:
                    tokens.append(f"Instrument={clean_instrument_name}")
    
    return tokens


# --- FUNZIONE PRINCIPALE DI ANALISI ---
def build_and_save_profiles(list_of_token_sets, output_path, min_support=50, max_profiles=100):
    """
    Analizza i set di token di metadati per trovare profili comuni e li salva.
    """
    logging.info(f"Inizio creazione profili. Supporto minimo: {min_support} occorrenze.")
    
    instrument_combinations = collections.Counter()
    for tokens in tqdm(list_of_token_sets, desc="Contando combinazioni di strumenti"):
        instruments = tuple(sorted([t for t in tokens if t.startswith("Instrument=")]))
        if instruments:
            instrument_combinations[instruments] += 1
            
    profiles = []
    
    popular_combinations = {k: v for k, v in instrument_combinations.items() if v >= min_support}
    logging.info(f"Trovate {len(popular_combinations)} combinazioni di strumenti con supporto >= {min_support}.")
    
    sorted_combinations = sorted(popular_combinations.items(), key=lambda item: item[1], reverse=True)

    for instruments_tuple, count in tqdm(sorted_combinations, desc="Creando profili"):
        associated_metadata = collections.Counter()
        for tokens in list_of_token_sets:
            current_instruments = tuple(sorted([t for t in tokens if t.startswith("Instrument=")]))
            if current_instruments == instruments_tuple:
                other_tokens = [t for t in tokens if not t.startswith("Instrument=")]
                associated_metadata.update(other_tokens)

        if not associated_metadata:
            continue
            
        def get_most_common_token(prefix):
            category_tokens = {token: freq for token, freq in associated_metadata.items() if token.startswith(prefix)}
            if not category_tokens: return None
            return max(category_tokens, key=category_tokens.get)

        instrument_names = [i.replace("Instrument=", "").replace("_", " ") for i in instruments_tuple]
        
        # Aggiunge il numero di occorrenze (count) al nome del profilo.
        if len(instrument_names) > 5:
            # Crea un nome generico per ensemble grandi, includendo il conteggio
            profile_name = f"{len(instrument_names)} Instruments Ensemble ({count} files)"
        else:
            # Crea un nome basato sulla lista di strumenti, includendo il conteggio
            joined_names = ", ".join(instrument_names)
            profile_name = f"{joined_names} ({count} files)"
        
        profile = {
            "profile_name": profile_name,
            "support_count": count,
            "instruments": list(instruments_tuple),
            "recommended_key": get_most_common_token("Key="),
            "recommended_timesig": get_most_common_token("TimeSig="),
            "recommended_tempo": get_most_common_token("Tempo_"),
            "recommended_avg_vel": get_most_common_token("AvgVel_"),
            "recommended_vel_range": get_most_common_token("VelRange_"),
            "recommended_num_inst": get_most_common_token("NumInst_")
        }
        profiles.append(profile)

        if len(profiles) >= max_profiles:
            logging.info(f"Raggiunto il limite massimo di {max_profiles} profili. Interruzione.")
            break

    logging.info(f"Salvataggio di {len(profiles)} profili in {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"profiles": profiles}, f, ensure_ascii=False, indent=2)


# --- BLOCCO DI ESECUZIONE ---
def main():
    parser = argparse.ArgumentParser(
        description="Analizza le co-occorrenze dei metadati da un dataset .jsonl e crea profili."
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory contenente i file train.jsonl, validation.jsonl, etc.")
    parser.add_argument("--output_file", type=Path, required=True, help="Percorso del file JSON di output per i profili (es. metadata_profiles.json).")
    parser.add_argument("--min_support", type=int, default=50, help="Numero minimo di occorrenze per una combinazione di strumenti per essere considerata un profilo.")
    parser.add_argument("--max_profiles", type=int, default=100, help="Numero massimo di profili da generare.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Blocco di controllo del percorso di output
    output_path = args.output_file
    if output_path.is_dir():
        default_filename = "metadata_profiles.json"
        logging.warning(f"Il percorso di output è una directory. Il file verrà salvato come: {output_path / default_filename}")
        output_path = output_path / default_filename

    # Caricamento dati
    all_entries = []
    jsonl_files = list(args.input_dir.glob("*.jsonl"))
    if not jsonl_files:
        logging.error(f"Nessun file .jsonl trovato in {args.input_dir}")
        return

    logging.info(f"Trovati i seguenti file: {[f.name for f in jsonl_files]}")
    for file_path in jsonl_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Caricamento {file_path.name}"):
                all_entries.append(json.loads(line))
    
    logging.info(f"Caricate {len(all_entries)} entry in totale.")

    # Raggruppamento metadati per file originale
    original_files_metadata = collections.defaultdict(set)
    for entry in tqdm(all_entries, desc="Raggruppamento metadati per file"):
        metadata = entry.get('metadata', {})
        file_id = metadata.get('original_file_id', entry.get('id', 'unknown_id'))
        tokens = tokenize_metadata(metadata)
        original_files_metadata[file_id].update(tokens)
        
    logging.info(f"Raggruppati metadati per {len(original_files_metadata)} file unici.")

    # Analisi e salvataggio profili
    list_of_token_sets = list(original_files_metadata.values())
    
    ### MODIFICA CHIAVE ###
    # Passa la variabile 'output_path' corretta alla funzione, non 'args.output_file'.
    build_and_save_profiles(list_of_token_sets, output_path, args.min_support, args.max_profiles)

    logging.info("--- Analisi completata con successo! ---")
    logging.info(f"File profili salvato in: {output_path}")


if __name__ == "__main__":
    main()
