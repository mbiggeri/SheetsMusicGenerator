import json
import re
from pathlib import Path
import argparse
import logging
from collections import Counter, defaultdict
import mido
from tqdm import tqdm
import concurrent.futures
import os

# --- IMPORT MODULARE ---
# Assicurati che il file 'tokenize_metadata.py' sia nella stessa cartella.
try:
    from tokenize_metadata import tokenize_metadata
except ImportError:
    logging.error("Errore: Il file 'tokenize_metadata.py' non è stato trovato.")
    logging.error("Assicurati che sia presente nella stessa directory di questo script.")
    exit(1)


# --- COSTANTI E FUNZIONI DI SUPPORTO (COPIATE DA dataset_creator.py) ---
# (Queste sezioni rimangono invariate)
MIN_MIDO_DURATION_SECONDS = 30
MIN_NOTE_ON_MESSAGES = 100
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


def quick_midi_check_mido(file_path_str):
    try:
        mid = mido.MidiFile(file_path_str)
        duration = mid.length
        if duration < MIN_MIDO_DURATION_SECONDS:
            return {'passed': False, 'reason': 'mido_too_short'}

        note_on_count = 0
        velocities = []
        time_signature = None
        key_signature = None
        tempo_microseconds = None
        found_instruments = set()
        channel_9_active_for_drums = False

        for track in mid.tracks:
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
                if msg.type == 'program_change' and hasattr(msg, 'channel') and msg.channel != 9:
                    instrument_gm_name = GM_INSTRUMENT_MAP.get(msg.program)
                    if instrument_gm_name:
                        found_instruments.add(instrument_gm_name)

        if note_on_count < MIN_NOTE_ON_MESSAGES:
            return {'passed': False, 'reason': 'mido_too_few_notes'}
        if channel_9_active_for_drums:
            found_instruments.add("Drums")

        bpm_rounded = round(60_000_000 / tempo_microseconds / 5) * 5 if tempo_microseconds else None
        avg_velocity_rounded = round((sum(velocities) / len(velocities)) / 5) * 5 if velocities else None
        velocity_range_rounded = round((max(velocities) - min(velocities)) / 5) * 5 if velocities else None

        return {
            'passed': True,
            'time_signature': time_signature,
            'mido_declared_key_signature': key_signature,
            'midi_instruments': sorted(list(found_instruments)),
            'bpm_rounded': bpm_rounded,
            'avg_velocity_rounded': avg_velocity_rounded,
            'velocity_range_rounded': velocity_range_rounded
        }
    except Exception:
        return {'passed': False, 'reason': 'mido_parse_error'}


# --- NUOVA FUNZIONE WORKER PER IL MULTIPROCESSING ---
def process_file_for_analysis(file_path_tuple):
    """
    Funzione eseguita da ogni processo worker.
    Analizza un singolo file MIDI e ritorna una lista di token.
    """
    midi_file, midi_dir = file_path_tuple  # Estrai gli argomenti

    # 1. Estrai metadati grezzi dal file MIDI
    metadata_dict = quick_midi_check_mido(str(midi_file))
    
    if metadata_dict['passed']:
        # 2. Aggiungi il genere se la struttura delle cartelle lo permette
        try:
            relative_path = midi_file.relative_to(midi_dir)
            if len(relative_path.parts) > 1:
                metadata_dict['genre'] = relative_path.parts[0]
        except ValueError:
            pass  # Il file non è in una sottocartella, nessun genere da estrarre

        # 3. Tokenizza i metadati usando la funzione importata
        return tokenize_metadata(metadata_dict)
    
    return [] # Ritorna una lista vuota se il file viene scartato


# --- FUNZIONE PRINCIPALE (MODIFICATA PER IL MULTIPROCESSING) ---
def analyze_metadata_from_midi_directory(midi_dir: Path):
    """
    Analizza tutti i file MIDI in una directory usando processi paralleli,
    estrae tutti i metadati possibili, li tokenizza e restituisce un conteggio categorizzato.
    """
    if not midi_dir.is_dir():
        logging.error(f"Directory non trovata: {midi_dir}")
        return None, 0

    all_midi_files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
    logging.info(f"Trovati {len(all_midi_files)} file MIDI da analizzare.")

    all_tokens = []
    tasks = [(p, midi_dir) for p in all_midi_files] # Prepara gli argomenti per ogni worker
    
    num_workers = os.cpu_count() or 1
    logging.info(f"Avvio dell'analisi in parallelo con {num_workers} worker...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Esegui la funzione 'process_file_for_analysis' su tutti i file in parallelo
        # `executor.map` distribuisce i task e raccoglie i risultati
        results_iterator = executor.map(process_file_for_analysis, tasks)
        
        # Usa tqdm per mostrare una barra di progresso
        progress_bar = tqdm(results_iterator, total=len(all_midi_files), desc="Analisi Metadati MIDI")

        # Aggrega i risultati da tutti i worker
        for tokens_for_file in progress_bar:
            if tokens_for_file:  # Assicurati che la lista non sia vuota
                all_tokens.extend(tokens_for_file)

    logging.info(f"Analisi completata. Trovati {len(all_tokens)} token totali.")

    # Conta le occorrenze di ogni token
    token_counts = Counter(all_tokens)
    
    # Categorizza i token per un report più leggibile (logica invariata)
    categorized_counts = defaultdict(lambda: defaultdict(int))
    for token, count in token_counts.items():
        if token.startswith("Genre="): category, value = "Genre", token.split("=", 1)[1]
        elif token.startswith("Key="): category, value = "Key", token.split("=", 1)[1]
        elif token.startswith("TimeSig="): category, value = "TimeSignature", token.split("=", 1)[1]
        elif token.startswith("Tempo_"): category, value = "Tempo", token
        elif token.startswith("AvgVel_"): category, value = "AverageVelocity", token
        elif token.startswith("VelRange_"): category, value = "VelocityRange", token
        elif token.startswith("NumInst_"): category, value = "NumberOfInstruments", token
        elif token.startswith("Instrument="): category, value = "Instruments", token.split("=", 1)[1]
        else: category, value = "Other", token
        categorized_counts[category][value] = count

    # Ordina i risultati per leggibilità
    final_report = {
        category: dict(sorted(values.items(), key=lambda item: item[1], reverse=True))
        for category, values in sorted(categorized_counts.items())
    }

    return final_report, len(token_counts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Analizza i metadati da una directory di file MIDI in parallelo e genera un report.")
    parser.add_argument("midi_directory", type=str, help="Percorso della directory contenente i file .mid/.midi.")
    
    args = parser.parse_args()
    midi_path = Path(args.midi_directory)
    
    report, total_unique_tokens = analyze_metadata_from_midi_directory(midi_path)
    
    if report:
        # --- 1. GESTIONE FILE DI REPORT ---
        # Definisci i percorsi per entrambi i file di output
        output_json_report_file = midi_path.parent / f"{midi_path.name}_metadata_analysis_report.json"
        output_txt_report_file = midi_path.parent / f"{midi_path.name}_metadata_report.txt"

        # Salva il report dettagliato in formato JSON (per uso programmatico)
        with open(output_json_report_file, 'w', encoding='utf-8') as f_out:
            json.dump(report, f_out, indent=2, ensure_ascii=False)
        
        logging.info(f"\n--- REPORT COMPLETO DEI METADATI ---")
        logging.info(f"Numero totale di token UNICI trovati: {total_unique_tokens}")
        logging.info(f"Il report dettagliato JSON è stato salvato in: {output_json_report_file}")

        # --- 2. STAMPA A SCHERMO E CREAZIONE DEL FILE .TXT ---
        # Salva il report leggibile in un file di testo e contemporaneamente lo stampa a schermo
        logging.info(f"Salvataggio del report testuale leggibile in: {output_txt_report_file}")
        try:
            with open(output_txt_report_file, 'w', encoding='utf-8') as f_txt:
                summary_header = "--- Riepilogo Completo dell'Analisi dei Metadati ---\n"
                print("\n" + summary_header.strip())
                f_txt.write(summary_header)

                # Itera su ogni categoria nel report
                for category, values in report.items():
                    category_header = f"\n[{category}] - {len(values)} valori unici\n"
                    print(category_header.strip())
                    f_txt.write(category_header)

                    # Stampa e scrive TUTTI i valori per la categoria, non solo i top 5
                    for value, count in values.items():
                        line = f"  - {value}: {count} occorrenze\n"
                        # Stampa la linea sul terminale senza il newline finale
                        print(line.strip()) 
                        # Scrivi la linea completa nel file di testo
                        f_txt.write(line)

            print(f"\nReport JSON completo salvato in: {output_json_report_file}")
            print(f"Report testuale completo salvato in: {output_txt_report_file}")

        except IOError as e:
            logging.error(f"Impossibile scrivere il file di report testuale: {e}")

    else:
        logging.warning("Nessun metadato valido trovato o la directory era vuota.")