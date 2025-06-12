# config.py
# File di configurazione centralizzato per l'intero progetto.

import miditok
from pathlib import Path

# Programmi MIDI considerati "pianoforte" (usato in modalità "piano_only")
# General MIDI programs 0-7: Acoustic Grand, Bright Acoustic, Electric Grand, Honky-tonk,
# Electric Piano 1, Electric Piano 2, Harpsichord, Clavinet.
PIANO_PROGRAMS = list(range(0, 8))


# =============================================================================
# --- IMPOSTAZIONI DEL TOKENIZER MIDI ---
# =============================================================================

# Strategia di tokenizzazione da usare (es. miditok.REMI, miditok.TSD, etc.)
MIDI_TOKENIZER_STRATEGY = miditok.REMI

# Nomi dei token speciali per il tokenizer MIDI
MIDI_PAD_TOKEN_NAME = "PAD_None"
MIDI_SOS_TOKEN_NAME = "SOS_None"
MIDI_EOS_TOKEN_NAME = "EOS_None"
MIDI_UNK_TOKEN_NAME = "UNK_None"

# QUESTA CONFIGURAZIONE ABILITA REMI+
TOKENIZER_PARAMS = miditok.TokenizerConfig(
    special_tokens=[MIDI_PAD_TOKEN_NAME, MIDI_SOS_TOKEN_NAME, MIDI_EOS_TOKEN_NAME, MIDI_UNK_TOKEN_NAME],
    
    # PARAMETRI CHE DEFINISCONO REMI+
    use_programs=True,                  # Abilita i token Program per gestire gli strumenti.
    one_token_stream_for_programs=False, # Gestisce le tracce in un unico flusso, come richiesto da REMI+.
    use_time_signatures=True,           # Abilita i token per i cambi di tempo (Time Signature).

    # Altri parametri consigliati per una rappresentazione ricca
    use_velocities=True,
    use_chords=True,
    use_tempos=True,
    use_rests=True,
    use_controls=False
)

# =============================================================================
# --- IMPOSTAZIONI DEI TOKEN DEI METADATI ---
# =============================================================================

# Nomi dei token speciali per i metadati
META_PAD_TOKEN_NAME = "<pad_meta>"
META_UNK_TOKEN_NAME = "<unk_meta>"
META_SOS_TOKEN_NAME = "<sos_meta>"
META_EOS_TOKEN_NAME = "<eos_meta>"


# =============================================================================
# --- PARAMETRI DI CHUNKING E PADDING ---
# =============================================================================

# Lunghezza massima della sequenza di token MIDI per ogni chunk
MAX_SEQ_LEN_MIDI = 2048

# Lunghezza minima della sequenza di token MIDI per considerare un chunk valido
MIN_CHUNK_LEN_MIDI = 128

# Lunghezza massima della sequenza di token dei metadati
MAX_SEQ_LEN_META = 128


# =============================================================================
# --- PARAMETRI DI TRASPOSIZIONE (usato in modalità 'piano_only') ---
# =============================================================================

# Tonalità di riferimento per la trasposizione
REFERENCE_KEY_MAJOR = "C"
REFERENCE_KEY_MINOR = "A"


# =============================================================================
# --- FUNZIONE PER LA GESTIONE DEI PERCORSI ---
# =============================================================================

def get_project_paths(base_data_dir: Path):
    """
    Restituisce un dizionario con tutti i percorsi chiave del progetto,
    derivandoli dalla directory di base.
    """
    return {
        "base": base_data_dir,
        "midi_input": base_data_dir / "MIDI",
        "output_splits": base_data_dir / "dataset_splits",
        "binary_chunks": base_data_dir / "dataset_splits" / "binary_chunks",
        "log_file": base_data_dir / "dataset_processing.log",
        "midi_vocab": base_data_dir / "midi_vocab.json",
        "metadata_vocab": base_data_dir / "metadata_vocab.json",
        "metadata_frequency": base_data_dir / "metadata_frequency.json",
        "model_save": base_data_dir / "models" # Aggiunto un percorso di default per i modelli
    }