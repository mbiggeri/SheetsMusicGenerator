import re
import json
import random
import math
import logging
from pathlib import Path
import music21 as m21
import os
import time # Lo teniamo per eventuali future operazioni lente

# --- Configurazione Globale ---

# Directory Locali (ASSICURATI CHE SIANO CORRETTE)
BASE_DATA_DIR = Path("./mutopia_data")
LY_DOWNLOAD_DIR = BASE_DATA_DIR / "ly_files"       # Dove sono i file .ly scaricati
MIDI_DOWNLOAD_DIR = BASE_DATA_DIR / "midi_files"     # Dove sono i file .mid scaricati
OUTPUT_DIR = BASE_DATA_DIR / "dataset_splits"    # Dove salvare train/val/test.jsonl
LOG_FILE = BASE_DATA_DIR / "dataset_processing.log" # Nuovo nome per il log

# Metadati essenziali richiesti (come nel tuo script)
ESSENTIAL_METADATA = ['key', 'time_signature']

# Split Proportions (come nel tuo script - include test set)
TRAIN_SPLIT = 0.9
VALIDATION_SPLIT = 0.1
# TEST_SPLIT è il rimanente

# Configurazione Logging
BASE_DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True) # Assicura che la dir di output esista
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info("Inizio elaborazione file locali per creazione dataset Mutopia.")

# --- Funzioni Ausiliarie (Copia/Incolla dallo script precedente) ---

def extract_metadata_from_ly(ly_file_path):
    """Estrae metadati dal blocco \\header di un file LilyPond (.ly)."""
    metadata = {}
    if not ly_file_path or not ly_file_path.exists():
        return metadata
    try:
        with open(ly_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            header_match = re.search(r'\\header\s*\{([^}]+)\}', content, re.DOTALL)
            if header_match:
                header_content = header_match.group(1)
                pattern = re.compile(r'^\s*([a-zA-Z0-9]+)\s*=\s*"(.*?)"\s*$', re.MULTILINE)
                for match in pattern.finditer(header_content):
                    key = match.group(1).lower().strip()
                    value = match.group(2).strip()
                    metadata[key] = value
    except Exception as e:
        logging.warning(f"Errore nella lettura/parsing di {ly_file_path.name}: {e}")
    return metadata

def analyze_music_file(file_path):
    """Analizza un file musicale (MIDI) usando music21 per ottenere tonalità, metro e strumenti."""
    analysis = {'key': None, 'time_signature': None, 'midi_instruments': []} # Aggiunta chiave per strumenti
    if not file_path or not file_path.exists():
        return analysis
    try:
        score = m21.converter.parse(file_path, forceSource=True)
        
        # Analisi Tonalità
        key_analysis = score.analyze('key')
        if key_analysis:
            tonic_name = key_analysis.tonic.name.capitalize()
            if key_analysis.tonic.accidental:
                 tonic_name = tonic_name[0] + key_analysis.tonic.accidental.modifier + tonic_name[1:]
            analysis['key'] = f"{tonic_name} {key_analysis.mode}"
            
        # Analisi Metro
        time_sig_elements = score.flat.getElementsByClass('TimeSignature')
        if time_sig_elements:
            # Prendi il primo time signature significativo, a volte ce ne sono di anacrusis etc.
            # Se ci sono cambi di metro, questo ne prenderà solo il primo.
            # Per una gestione più complessa (es. cambi di metro), servirebbe logica aggiuntiva.
            analysis['time_signature'] = time_sig_elements[0].ratioString 

        # Estrazione Strumenti dalle tracce MIDI
        instrument_names = set() # Usiamo un set per evitare duplicati
        for part in score.parts:
            instrument = part.getInstrument(returnDefault=False) # returnDefault=False per non avere GenericInstrument
            if instrument and hasattr(instrument, 'instrumentName') and instrument.instrumentName:
                # Pulisci un po' il nome, music21 a volte aggiunge numeri o altro
                name = str(instrument.instrumentName).strip()
                # Potresti voler fare una pulizia più aggressiva qui se necessario
                # es. rimuovere "MIDI Instrument" o normalizzare nomi comuni
                instrument_names.add(name)
            elif not instrument and part.partName: # Fallback al nome della traccia se non c'è strumento esplicito
                instrument_names.add(str(part.partName).strip())


        if not instrument_names and len(score.parts) > 0: # Se nessun nome strumento ma ci sono tracce
            # Potremmo provare a inferire dal program change, ma è più complesso.
            # Per ora, se non c'è nome esplicito, lasciamo vuoto o usiamo un placeholder.
            # Oppure, se c'è solo una traccia, potremmo assumere "Unknown Instrument" o "Solo Instrument"
            # Se ci sono più tracce anonime, diventa difficile.
            pass # Lascia vuoto per ora

        analysis['midi_instruments'] = sorted(list(instrument_names)) # Salva come lista ordinata

    except m21.converter.ConverterException as e:
        logging.warning(f"Errore di conversione music21 (file potrebbe essere corrotto o non supportato) per {file_path.name}: {e}")
    except Exception as e:
        logging.warning(f"Errore generico nell'analisi music21 di {file_path.name}: {e}")
    return analysis

def find_corresponding_ly_file(midi_path, ly_base_dir):
    """Tenta di trovare il file .ly corrispondente a un file .mid."""
    # Assumiamo che la struttura delle cartelle sia parallela
    try:
        relative_midi_path = midi_path.relative_to(MIDI_DOWNLOAD_DIR)
    except ValueError:
         logging.warning(f"Il file MIDI {midi_path} non sembra essere dentro {MIDI_DOWNLOAD_DIR}")
         return None # Non possiamo trovare il corrispondente .ly

    potential_ly_dir = ly_base_dir / relative_midi_path.parent
    midi_stem = midi_path.stem

    if potential_ly_dir.exists():
        # Prova corrispondenza esatta dello stem
        exact_ly_path = potential_ly_dir / relative_midi_path.with_suffix(".ly")
        if exact_ly_path.exists() and exact_ly_path.is_file():
            return exact_ly_path

        # Prova a cercare file che iniziano con lo stem (per gestire suffissi tipo -a4, -letter)
        # Questa logica potrebbe necessitare aggiustamenti se i nomi sono molto diversi
        for ly_file in potential_ly_dir.glob(f"{midi_stem}*.ly"):
            if ly_file.is_file():
                logging.debug(f"Trovato .ly corrispondente potenziale: {ly_file.name} per {midi_path.name}")
                return ly_file # Restituisce il primo trovato affidabile

    # logging.debug(f"Nessun file .ly trovato per {midi_path.name} in {potential_ly_dir}")
    return None

def save_dataset_split(data, filename):
    """Salva una porzione del dataset in un file .jsonl."""
    output_path = OUTPUT_DIR / f"{filename}.jsonl"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in data:
                # Assicurati che tutti i dati siano serializzabili in JSON
                # (Path objects non lo sono, li abbiamo convertiti in stringhe relative)
                f.write(json.dumps(entry) + '\n')
        logging.info(f"Salvato {len(data)} voci in {output_path.relative_to(Path('.'))}")
    except TypeError as e:
         logging.error(f"Errore di tipo durante la scrittura JSON in {output_path}: {e}. Controlla i dati.")
         # Potresti voler loggare l'entry problematica qui per debug
         # logging.error(f"Dati problematici (primi 100 caratteri): {str(entry)[:100]}")
    except Exception as e:
        logging.error(f"Errore nel salvataggio di {output_path}: {e}")


# --- Flusso Principale ---

if __name__ == "__main__":

    # 1. Elaborazione dei file locali
    logging.info("--- Fase 1: Elaborazione File Locali e Estrazione Metadati ---")
    raw_dataset = []
    # Cerca ricorsivamente tutti i file .mid nella directory specificata
    midi_files_found = list(MIDI_DOWNLOAD_DIR.rglob("*.mid"))
    logging.info(f"Trovati {len(midi_files_found)} file MIDI locali in {MIDI_DOWNLOAD_DIR} da elaborare.")

    if not midi_files_found:
         logging.warning("Nessun file MIDI trovato localmente. Verifica il percorso MIDI_DOWNLOAD_DIR.")
    else:
        processed_files = 0
        # Itera sui file MIDI trovati
        for midi_file_path in midi_files_found:
            processed_files += 1
            if processed_files % 200 == 0: # Logga ogni 200 file elaborati
                logging.info(f"Elaborato {processed_files}/{len(midi_files_found)} file MIDI...")

            logging.debug(f"Elaborazione: {midi_file_path.relative_to(MIDI_DOWNLOAD_DIR)}")

            # Trova il file .ly corrispondente nella struttura parallela
            ly_file_path = find_corresponding_ly_file(midi_file_path, LY_DOWNLOAD_DIR)
            if not ly_file_path:
                 logging.warning(f"  File .ly corrispondente non trovato per {midi_file_path.name}, uso solo analisi MIDI.")

            # Estrai metadati base dal file .ly (restituisce {} se non trovato)
            base_metadata = extract_metadata_from_ly(ly_file_path)

            # Analizza il file MIDI per tonalità/metro
            analysis_metadata = analyze_music_file(midi_file_path)

            # Combina i metadati (quelli da analisi sovrascrivono quelli da .ly se presenti in entrambi)
            combined_metadata = {**base_metadata, **analysis_metadata}

            # Aggiungi pulizia/standardizzazione se necessario qui
            # Esempio: Rinomina chiavi o normalizza valori
            if 'mutopiacomposer' in combined_metadata and 'composer' not in combined_metadata:
                combined_metadata['composer'] = combined_metadata['mutopiacomposer']
            if 'mutopiatitle' in combined_metadata and 'title' not in combined_metadata:
                combined_metadata['title'] = combined_metadata['mutopiatitle']
            # Rimuovi chiavi non utili se vuoi
            combined_metadata.pop('mutopiacomposer', None)
            combined_metadata.pop('mutopiatitle', None)

            # --- QUI AVVIENE IL CONTROLLO SUI METADATI ESSENZIALI ---
            missing_essentials = [k for k in ESSENTIAL_METADATA if not combined_metadata.get(k)]

            if not missing_essentials:
                # Aggiungi i percorsi relativi per riferimento nel dataset finale
                combined_metadata['midi_relative_path'] = str(midi_file_path.relative_to(BASE_DATA_DIR))
                if ly_file_path:
                    combined_metadata['ly_relative_path'] = str(ly_file_path.relative_to(BASE_DATA_DIR))
                else:
                    combined_metadata['ly_relative_path'] = None

                # Aggiungi la voce al dataset grezzo
                raw_dataset.append({'metadata': combined_metadata}) # Struttura più pulita

            else:
                # Scarta la voce se mancano metadati essenziali
                logging.warning(f"  {midi_file_path.name}: Saltato - Metadati essenziali mancanti: {missing_essentials}")
                # Logga i metadati trovati per debug, se vuoi
                # logging.debug(f"    Metadati trovati: {combined_metadata}")

    logging.info(f"Elaborazione completata. Voci valide nel dataset grezzo: {len(raw_dataset)}")

    # 2. Divisione del dataset
    logging.info("--- Fase 2: Divisione del Dataset ---")
    if not raw_dataset:
        logging.error("Dataset grezzo è vuoto. Impossibile creare split.")
    else:
        # Mescola il dataset prima di dividerlo
        random.shuffle(raw_dataset)

        # Calcola dimensioni degli split
        n = len(raw_dataset)
        n_train = math.floor(TRAIN_SPLIT * n)
        n_val = math.floor(VALIDATION_SPLIT * n)
        # n_test è il resto

        # Crea gli split
        train_data = raw_dataset[:n_train]
        val_data = raw_dataset[n_train : n_train + n_val]
        test_data = raw_dataset[n_train + n_val :] # Mantiene il test set come nello script originale

        logging.info(f"Dimensioni dataset: Totale={n}, Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")

        # Salva gli split su file JSON Lines
        save_dataset_split(train_data, "train")
        save_dataset_split(val_data, "validation")
        save_dataset_split(test_data, "test") # Mantiene il salvataggio del test set

    logging.info("--- Script Terminato ---")