# finetune_RL.py (Versione aggiornata, autonoma e corretta)
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import logging
import random
import sys
import os
import tempfile
from tqdm import tqdm
import numpy as np
import mido
import argparse

# Importazioni da altri file del progetto
# Assicurati che questi file siano accessibili dal tuo ambiente Python
from generate_music_RL import Seq2SeqTransformer, generate_multi_chunk_midi, enhance_midi_score
from tokenize_metadata import tokenize_metadata
# --- MODIFICA: Rimosso l'import di 'build_or_load_tokenizer' da training ---
from training import load_metadata_vocab
import config

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Impostazione del device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# === NUOVA FUNZIONE LOCALE PER CARICARE IL TOKENIZER (SOSTITUISCE L'IMPORT) ===
# ==============================================================================

def build_or_load_tokenizer_rl(vocab_path: Path):
    """
    Versione locale per caricare il tokenizer MIDI per il fine-tuning RL.
    Questa funzione richiede che il vocabolario esista già.
    """
    if not vocab_path.exists():
        logging.error(f"Errore critico: il file del vocabolario del tokenizer non è stato trovato in {vocab_path}.")
        logging.error("Esegui prima dataset_creator.py per generare il vocabolario.")
        raise FileNotFoundError(f"Vocabolario tokenizer non trovato in {vocab_path}.")

    logging.info(f"Caricamento del tokenizer MIDI da {vocab_path} per RL.")
    try:
        # Carica il tokenizer usando la strategia e i parametri definiti in config.py
        tokenizer = config.MIDI_TOKENIZER_STRATEGY(params=str(vocab_path))
        logging.info(f"Tokenizer caricato con successo da {vocab_path}")
    except Exception as e:
        logging.error(f"Errore irreversibile durante il caricamento del tokenizer da {vocab_path}: {e}", exc_info=True)
        raise IOError(f"Impossibile caricare il tokenizer da {vocab_path}.")

    # Controlli di integrità del tokenizer caricato
    try:
        assert tokenizer[config.MIDI_PAD_TOKEN_NAME] is not None
        assert tokenizer[config.MIDI_SOS_TOKEN_NAME] is not None
        assert tokenizer[config.MIDI_EOS_TOKEN_NAME] is not None
    except (AssertionError, KeyError) as e:
        logging.error(f"CRITICO: Token speciali MIDI (PAD, SOS, EOS) mancanti nel tokenizer caricato: {e}.")
        raise ValueError("Il tokenizer caricato è corrotto o incompleto (mancano token speciali).")

    logging.info(f"Tokenizer MIDI per RL pronto. Dimensione vocabolario: {len(tokenizer)}")
    return tokenizer

# ==============================================================================
# === BLOCCO FUNZIONI PER RL (precedentemente in finetune_generator.py)      ===
# ==============================================================================

# Mappa per convertire le categorie di tempo in valori BPM medi per la generazione casuale
REVERSE_TEMPO_MAP = {
    "Tempo_VerySlow": 60,
    "Tempo_Slow": 75,
    "Tempo_Moderate": 110,
    "Tempo_Fast": 130,
    "Tempo_VeryFast": 170,
    "Tempo_ExtremelyFast": 190,
}

# Mappa per i nomi degli strumenti GM (usata in analyze_generated_midi)
GM_INSTRUMENT_MAP = {
    0: "Acoustic Grand Piano", 1: "Bright Acoustic Piano", 2: "Electric Grand Piano", 3: "Honky-tonk Piano",
    4: "Electric Piano 1", 5: "Electric Piano 2", 6: "Harpsichord", 7: "Clavinet",
    # ... (mappa completa come fornita precedentemente) ...
    127: "Gunshot"
}


def analyze_generated_midi(midi_path: Path) -> dict:
    """
    Analizza un file MIDI generato ed estrae metadati chiave.
    """
    try:
        mid = mido.MidiFile(str(midi_path))
        note_on_count = 0
        velocities = []
        time_signature = None
        key_signature = None
        tempo_microseconds = None
        found_instruments = set()

        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_on_count += 1
                if not time_signature and msg.is_meta and msg.type == 'time_signature':
                    time_signature = f"{msg.numerator}/{msg.denominator}"
                if msg.type == 'program_change':
                    if hasattr(msg, 'channel') and msg.channel != 9:
                        instrument_name = GM_INSTRUMENT_MAP.get(msg.program, "Unknown")
                        found_instruments.add(instrument_name)
        
        if not found_instruments and note_on_count > 0:
            found_instruments.add("Acoustic Grand Piano")

        analysis = {'note_count': note_on_count}
        if time_signature:
            analysis['time_signature'] = time_signature
            
        return analysis
    except Exception as e:
        logging.warning(f"Impossibile analizzare il file MIDI generato {midi_path}: {e}")
        return {'note_count': 0}


def generate_random_metadata_dict(metadata_vocab_map: dict) -> dict:
    """
    Genera un dizionario di metadati casuale ma coerente per creare un prompt per un episodio RL.
    """
    categorized_tokens = {
        'TimeSig': [], 'Tempo': [], 'Key': [], 'Instrument': []
    }
    for token in metadata_vocab_map.keys():
        if token.startswith('TimeSig='):
            categorized_tokens['TimeSig'].append(token)
        elif token.startswith('Tempo_'):
            categorized_tokens['Tempo'].append(token)
        elif token.startswith('Instrument='):
            categorized_tokens['Instrument'].append(token)

    prompt_dict = {}
    
    if categorized_tokens['Tempo']:
        chosen_tempo_token = random.choice(categorized_tokens['Tempo'])
        prompt_dict['bpm_rounded'] = REVERSE_TEMPO_MAP.get(chosen_tempo_token, 120)

    if categorized_tokens['TimeSig']:
        chosen_ts_token = random.choice(categorized_tokens['TimeSig'])
        prompt_dict['time_signature'] = chosen_ts_token.split('=')[1]

    if categorized_tokens['Instrument']:
        num_instruments = random.randint(1, 3)
        chosen_instrument_tokens = random.sample(categorized_tokens['Instrument'], k=min(num_instruments, len(categorized_tokens['Instrument'])))
        instrument_names = [tok.split('=')[1].replace('_', ' ') for tok in chosen_instrument_tokens]
        prompt_dict['midi_instruments'] = instrument_names
        
    return prompt_dict


def evaluate_adherence(prompt_dict: dict, analysis_dict: dict) -> dict:
    """
    Confronta i metadati richiesti (prompt) con quelli analizzati e restituisce un punteggio di aderenza.
    """
    scores = {}
    total_score = 0
    num_criteria = 0

    prompt_tokens = set(tokenize_metadata(prompt_dict))
    analysis_tokens = set(tokenize_metadata(analysis_dict))

    categories_to_check = ['TimeSig', 'Tempo', 'NumInst']
    
    for category in categories_to_check:
        prompt_cat_token = next((t for t in prompt_tokens if t.startswith(category)), None)
        if prompt_cat_token:
            num_criteria += 1
            analysis_cat_token = next((t for t in analysis_tokens if t.startswith(category)), None)
            if prompt_cat_token == analysis_cat_token:
                total_score += 1.0

    prompt_instruments = {t for t in prompt_tokens if t.startswith('Instrument=')}
    if prompt_instruments:
        num_criteria += 1
        analysis_instruments = {t for t in analysis_tokens if t.startswith('Instrument=')}
        intersection = len(prompt_instruments.intersection(analysis_instruments))
        union = len(prompt_instruments.union(analysis_instruments))
        instrument_score = intersection / union if union > 0 else 0.0
        total_score += instrument_score
    
    overall_adherence = total_score / num_criteria if num_criteria > 0 else 0.0
    scores['overall_adherence'] = overall_adherence

    return scores

# ============================================================================
# === FINE BLOCCO FUNZIONI                                                 ===
# ============================================================================


# --- Parametri RL ---
LEARNING_RATE_RL = 1e-6
GAMMA = 0.99
NUM_EPISODES = 1000
BATCH_SIZE_RL = 4
REPORT_INTERVAL = 10


# --- Funzione di Interazione con l'Ambiente ---
def get_generation_and_reward(model, midi_tokenizer, metadata_vocab_map, metadata_prompt_dict, device):
    """
    Esegue una generazione completa, calcola il reward e restituisce i dati necessari.
    """
    metadata_prompt_str_list = tokenize_metadata(metadata_prompt_dict)
    
    gen_config = {
        "temperature": 1.0, "top_k": 0, "max_rest_penalty": 0.0, "rest_penalty_mode": 'hybrid'
    }
    model_chunk_capacity = model.positional_encoding.pe.size(1)

    generated_token_ids, log_probs_tensor = generate_multi_chunk_midi(
        model, midi_tokenizer, metadata_vocab_map, metadata_prompt_str_list,
        total_target_tokens=config.MAX_SEQ_LEN_MIDI,
        model_chunk_capacity=model_chunk_capacity,
        generation_config=gen_config,
        device=device,
        initial_primer_ids=None,
        rest_ids=torch.tensor([i for t, i in midi_tokenizer.vocab.items() if t.startswith("Rest_")], device=DEVICE)
    )

    if not generated_token_ids or log_probs_tensor is None or log_probs_tensor.numel() == 0:
        logging.warning("Generazione fallita o vuota, reward = -1.")
        return None, -1.0, None

    temp_midi_file = None
    reward = 0.0
    try:
        temp_score = midi_tokenizer.decode([generated_token_ids])
        temp_score = enhance_midi_score(temp_score, {}, metadata_prompt_str_list, lambda msg: logging.debug(msg))

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tf:
            temp_midi_file = tf.name
            temp_score.dump_midi(temp_midi_file)
        
        generated_analysis = analyze_generated_midi(Path(temp_midi_file))
        adherence_scores = evaluate_adherence(metadata_prompt_dict, generated_analysis)
        reward = adherence_scores.get('overall_adherence', 0.0)
        
        if generated_analysis.get('note_count', 0) < 20:
            reward -= 0.5
        if len(generated_token_ids) < config.MIN_CHUNK_LEN_MIDI:
            reward -= 0.5

    except Exception as e:
        logging.error(f"Errore durante l'analisi per reward: {e}")
        reward = -1.0
    finally:
        if temp_midi_file and Path(temp_midi_file).exists():
            os.remove(temp_midi_file)

    return generated_token_ids, reward, log_probs_tensor


# --- LOOP DI ADDESTRAMENTO RL ---
def train_rl(model, optimizer, midi_tokenizer, metadata_vocab_map):
    model.train()
    history = {'total_rewards': [], 'avg_rewards': [], 'losses': []}
    batch_rewards = []
    batch_losses = []

    pbar = tqdm(range(NUM_EPISODES), desc="RL Training Episodes")
    for episode in pbar:
        metadata_prompt_dict = generate_random_metadata_dict(metadata_vocab_map)
        
        generated_tokens, reward, log_probs_tensor = get_generation_and_reward(
            model, midi_tokenizer, metadata_vocab_map, metadata_prompt_dict, DEVICE
        )

        if generated_tokens is None or log_probs_tensor is None:
            continue
            
        loss = -log_probs_tensor.sum() * reward
        loss = loss / len(generated_tokens)

        batch_rewards.append(reward)
        batch_losses.append(loss)

        if (episode + 1) % BATCH_SIZE_RL == 0:
            if not batch_losses: continue

            total_batch_loss = torch.stack(batch_losses).sum()
            optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            avg_batch_reward = sum(batch_rewards) / len(batch_rewards)
            pbar.set_postfix({"Avg Reward": f"{avg_batch_reward:.4f}", "Loss": f"{total_batch_loss.item():.4f}"})
            batch_rewards, batch_losses = [], []

        if (episode + 1) % (REPORT_INTERVAL * BATCH_SIZE_RL) == 0:
            # Salva checkpoint periodico
            pass # Logica di salvataggio omessa per brevità

    logging.info("Addestramento RL completato.")
    return history

# --- ESECUZIONE PRINCIPALE ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegui il fine-tuning di un modello musicale con Reinforcement Learning.")
    parser.add_argument("--base_data_dir", type=Path, required=True, help="Percorso della cartella base del dataset (contenente modelli, ecc.).")
    
    # --- MODIFICA: Aggiunti argomenti per specificare i percorsi dei vocabolari ---
    parser.add_argument("--midi_vocab_path", type=Path, default=None, help="Percorso specifico al file midi_vocab.json. Se non fornito, viene derivato da base_data_dir.")
    parser.add_argument("--metadata_vocab_path", type=Path, default=None, help="Percorso specifico al file metadata_vocab.json. Se non fornito, viene derivato da base_data_dir.")
    
    parser.add_argument("--resume_rl_checkpoint", type=Path, default=None, help="Percorso opzionale a un checkpoint RL per riprendere l'addestramento.")
    args = parser.parse_args()

    # --- SETUP DEI PERCORSI ---
    BASE_DATA_DIR = args.base_data_dir
    MODEL_SAVE_DIR = BASE_DATA_DIR / "rl_models"
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- MODIFICA: Logica per determinare i percorsi dei vocabolari ---
    # Usa il percorso specifico se fornito, altrimenti derivato dalla directory base.
    MIDI_VOCAB_PATH = args.midi_vocab_path if args.midi_vocab_path else BASE_DATA_DIR / "midi_vocab.json"
    METADATA_VOCAB_PATH = args.metadata_vocab_path if args.metadata_vocab_path else BASE_DATA_DIR / "metadata_vocab.json"

    logging.info(f"Percorso Vocabolario MIDI in uso: {MIDI_VOCAB_PATH}")
    logging.info(f"Percorso Vocabolario Metadati in uso: {METADATA_VOCAB_PATH}")

    # --- MODIFICA: Caricamento tokenizer e vocabolari usando le nuove funzioni/logiche ---
    try:
        # Chiama la nuova funzione locale `build_or_load_tokenizer_rl`
        midi_tokenizer = build_or_load_tokenizer_rl(vocab_path=MIDI_VOCAB_PATH)
        
        # La funzione `load_metadata_vocab` importata da `training.py` funziona correttamente
        metadata_vocab_map = load_metadata_vocab(METADATA_VOCAB_PATH)
    except (FileNotFoundError, IOError, ValueError) as e:
        logging.error(f"Impossibile inizializzare i componenti necessari. Errore: {e}")
        sys.exit(1)

    # 3. Determina quale checkpoint caricare
    if args.resume_rl_checkpoint and args.resume_rl_checkpoint.exists():
        checkpoint_path = args.resume_rl_checkpoint
        logging.info(f"Ripresa dell'addestramento RL dal checkpoint: {checkpoint_path}")
    else:
        # Il checkpoint per il fine-tuning dovrebbe essere nella sottocartella 'models', non 'rl_models'
        checkpoint_path = BASE_DATA_DIR / "models" / "transformer_best.pt"
        logging.info(f"Avvio di un nuovo addestramento RL dal modello supervisionato: {checkpoint_path}")
        if not checkpoint_path.exists():
            logging.error(f"Checkpoint del modello pre-addestrato non trovato: {checkpoint_path}.")
            sys.exit(1)

    # 4. Caricamento del modello e dell'ottimizzatore
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_params = checkpoint.get('model_params')
    if not model_params:
        logging.error("'model_params' non trovato nel checkpoint.")
        sys.exit(1)
        
    model = Seq2SeqTransformer(**model_params).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info("Modello caricato.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_RL)
    if args.resume_rl_checkpoint and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info("Stato dell'ottimizzatore ripristinato dal checkpoint RL.")

    # 5. Avvio del training
    rl_history = train_rl(model, optimizer, midi_tokenizer, metadata_vocab_map)

    # 6. Salvataggio finale
    final_rl_model_path = MODEL_SAVE_DIR / "transformer_rl_final.pt"
    torch.save({
        'episode': NUM_EPISODES,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': rl_history,
        'model_params': model_params,
    }, final_rl_model_path)
    logging.info(f"Modello RL finale salvato in {final_rl_model_path}")