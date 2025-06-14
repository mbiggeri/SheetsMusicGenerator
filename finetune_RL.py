# finetune_rl.py (Versione completa e autonoma)
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

# Importazioni da altri file del progetto
from generate_music_RL import Seq2SeqTransformer, generate_multi_chunk_midi, enhance_midi_score
from tokenize_metadata import tokenize_metadata
from training import build_or_load_tokenizer, load_metadata_vocab
import config

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Impostazione del device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# === INIZIO BLOCCO FUNZIONI PER RL (da aggiungere a finetune_RL.py) ===
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
                    velocities.append(msg.velocity)
                if not time_signature and msg.is_meta and msg.type == 'time_signature':
                    time_signature = f"{msg.numerator}/{msg.denominator}"
                if not key_signature and msg.is_meta and msg.type == 'key_signature':
                    key_signature = msg.key
                if tempo_microseconds is None and msg.is_meta and msg.type == 'set_tempo':
                    tempo_microseconds = msg.tempo
                if msg.type == 'program_change':
                    if hasattr(msg, 'channel') and msg.channel != 9:
                        instrument_name = GM_INSTRUMENT_MAP.get(msg.program, "Unknown")
                        found_instruments.add(instrument_name)
        
        if not found_instruments and note_on_count > 0:
            found_instruments.add("Acoustic Grand Piano")

        analysis = {'note_count': note_on_count}
        if time_signature:
            analysis['time_signature'] = time_signature
        if key_signature:
            analysis['mido_declared_key_signature'] = key_signature
        if tempo_microseconds:
            bpm = mido.tempo2bpm(tempo_microseconds)
            analysis['bpm_rounded'] = round(bpm / 5) * 5
        if velocities:
            analysis['avg_velocity_rounded'] = round((sum(velocities) / len(velocities)) / 5) * 5
            analysis['velocity_range_rounded'] = round((max(velocities) - min(velocities)) / 5) * 5
        if found_instruments:
            analysis['midi_instruments'] = sorted(list(found_instruments))
            
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
        elif token.startswith('Key='):
            categorized_tokens['Key'].append(token)
        elif token.startswith('Instrument='):
            categorized_tokens['Instrument'].append(token)

    prompt_dict = {}
    
    if categorized_tokens['Tempo']:
        chosen_tempo_token = random.choice(categorized_tokens['Tempo'])
        prompt_dict['bpm_rounded'] = REVERSE_TEMPO_MAP.get(chosen_tempo_token, 120)

    if categorized_tokens['TimeSig']:
        chosen_ts_token = random.choice(categorized_tokens['TimeSig'])
        prompt_dict['time_signature'] = chosen_ts_token.split('=')[1]

    if categorized_tokens['Key']:
        chosen_key_token = random.choice(categorized_tokens['Key'])
        prompt_dict['mido_declared_key_signature'] = chosen_key_token.split('=')[1].replace('sharp', '#').replace('_', ' ')

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

    categories_to_check = ['TimeSig', 'Tempo', 'Key', 'NumInst']
    
    for category in categories_to_check:
        prompt_cat_token = next((t for t in prompt_tokens if t.startswith(category)), None)
        if prompt_cat_token:
            num_criteria += 1
            analysis_cat_token = next((t for t in analysis_tokens if t.startswith(category)), None)
            if prompt_cat_token == analysis_cat_token:
                scores[f'{category}_adherence'] = 1.0
                total_score += 1.0
            else:
                scores[f'{category}_adherence'] = 0.0

    prompt_instruments = {t for t in prompt_tokens if t.startswith('Instrument=')}
    if prompt_instruments:
        num_criteria += 1
        analysis_instruments = {t for t in analysis_tokens if t.startswith('Instrument=')}
        
        intersection = len(prompt_instruments.intersection(analysis_instruments))
        union = len(prompt_instruments.union(analysis_instruments))
        
        instrument_score = intersection / union if union > 0 else 0.0
        scores['instrument_adherence'] = instrument_score
        total_score += instrument_score
    
    overall_adherence = total_score / num_criteria if num_criteria > 0 else 0.0
    scores['overall_adherence'] = overall_adherence

    return scores

# ============================================================================
# === FINE BLOCCO FUNZIONI PER RL                                          ===
# ============================================================================


# --- Parametri RL ---
LEARNING_RATE_RL = 1e-6 
GAMMA = 0.99
NUM_EPISODES = 1000
BATCH_SIZE_RL = 4
REPORT_INTERVAL = 10

# --- Caricamento Configurazioni Globali ---
BASE_DATA_DIR = Path("./mutopia_data_test") 
MODEL_SAVE_DIR = BASE_DATA_DIR / "rl_models"
PRETRAINED_CHECKPOINT_PATH = BASE_DATA_DIR / "models" / "transformer_best.pt"

MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_PATH = BASE_DATA_DIR / "midi_vocab.json"
METADATA_VOCAB_PATH = BASE_DATA_DIR / "metadata_vocab.json"

# Carica il tokenizer e i vocabolari
MIDI_TOKENIZER = build_or_load_tokenizer(force_build=False)
METADATA_VOCAB_MAP = load_metadata_vocab(METADATA_VOCAB_PATH)

MIDI_PAD_ID = MIDI_TOKENIZER[config.MIDI_PAD_TOKEN_NAME]
META_PAD_ID = METADATA_VOCAB_MAP[config.META_PAD_TOKEN_NAME]
MIDI_VOCAB_SIZE = len(MIDI_TOKENIZER)
META_VOCAB_SIZE = len(METADATA_VOCAB_MAP)

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
def train_rl(model, optimizer):
    model.train()
    history = {'total_rewards': [], 'avg_rewards': [], 'losses': []}
    batch_rewards = []
    batch_losses = []

    pbar = tqdm(range(NUM_EPISODES), desc="RL Training Episodes")
    for episode in pbar:
        metadata_prompt_dict = generate_random_metadata_dict(METADATA_VOCAB_MAP)
        
        generated_tokens, reward, log_probs_tensor = get_generation_and_reward(
            model, MIDI_TOKENIZER, METADATA_VOCAB_MAP, metadata_prompt_dict, DEVICE
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
            history['total_rewards'].extend(batch_rewards)
            history['losses'].append(total_batch_loss.item())
            history['avg_rewards'].append(avg_batch_reward)
            pbar.set_postfix({"Avg Reward": f"{avg_batch_reward:.4f}", "Loss": f"{total_batch_loss.item():.4f}"})
            batch_rewards, batch_losses = [], []

        if (episode + 1) % (REPORT_INTERVAL * BATCH_SIZE_RL) == 0:
            if not history['avg_rewards']: continue
            current_avg_reward = sum(history['avg_rewards'][-REPORT_INTERVAL:]) / REPORT_INTERVAL
            logging.info(f"Episodio {episode+1}, Avg Reward: {current_avg_reward:.4f}")
            
            current_model_path = MODEL_SAVE_DIR / f"transformer_rl_episode_{episode+1}.pt"
            torch.save({
                'episode': episode + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_reward': current_avg_reward,
                'model_params': model.state_dict(),
            }, current_model_path)
            logging.info(f"Modello RL salvato in {current_model_path}")

    logging.info("Addestramento RL completato.")
    return history

# --- ESECUZIONE PRINCIPALE ---
if __name__ == "__main__":
    if not PRETRAINED_CHECKPOINT_PATH.exists():
        logging.error(f"Checkpoint del modello pre-addestrato non trovato: {PRETRAINED_CHECKPOINT_PATH}.")
        sys.exit(1)

    logging.info(f"Caricamento modello pre-addestrato da: {PRETRAINED_CHECKPOINT_PATH}")
    checkpoint = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location=DEVICE)
    
    model_params = checkpoint.get('model_params')
    if not model_params:
        logging.error("'model_params' non trovato nel checkpoint.")
        sys.exit(1)
        
    model = Seq2SeqTransformer(**model_params).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info("Modello caricato.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_RL)

    rl_history = train_rl(model, optimizer)

    final_rl_model_path = MODEL_SAVE_DIR / "transformer_rl_final.pt"
    torch.save({
        'episode': NUM_EPISODES,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': rl_history,
        'model_params': model_params,
    }, final_rl_model_path)
    logging.info(f"Modello RL finale salvato in {final_rl_model_path}")