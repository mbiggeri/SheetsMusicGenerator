# finetune_RL.py (versione aggiornata e corretta)
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

from generate_music_RL import Seq2SeqTransformer, generate_multi_chunk_midi, enhance_midi_score
from tokenize_metadata import tokenize_metadata
from training import load_metadata_vocab
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_or_load_tokenizer_rl(vocab_path: Path):
    if not vocab_path.exists():
        logging.error(f"Errore critico: il file del vocabolario del tokenizer non è stato trovato in {vocab_path}.")
        raise FileNotFoundError(f"Vocabolario tokenizer non trovato in {vocab_path}.")
    logging.info(f"Caricamento del tokenizer MIDI da {vocab_path} per RL.")
    try:
        tokenizer = config.MIDI_TOKENIZER_STRATEGY(params=str(vocab_path))
        logging.info(f"Tokenizer caricato con successo da {vocab_path}")
    except Exception as e:
        logging.error(f"Errore irreversibile durante il caricamento del tokenizer da {vocab_path}: {e}", exc_info=True)
        raise IOError(f"Impossibile caricare il tokenizer da {vocab_path}.")
    try:
        assert tokenizer[config.MIDI_PAD_TOKEN_NAME] is not None
        assert tokenizer[config.MIDI_SOS_TOKEN_NAME] is not None
        assert tokenizer[config.MIDI_EOS_TOKEN_NAME] is not None
    except (AssertionError, KeyError) as e:
        logging.error(f"CRITICO: Token speciali MIDI (PAD, SOS, EOS) mancanti nel tokenizer caricato: {e}.")
        raise ValueError("Il tokenizer caricato è corrotto o incompleto.")
    logging.info(f"Tokenizer MIDI per RL pronto. Dimensione vocabolario: {len(tokenizer)}")
    return tokenizer

REVERSE_TEMPO_MAP = {"Tempo_VerySlow": 60, "Tempo_Slow": 75, "Tempo_Moderate": 110, "Tempo_Fast": 130, "Tempo_VeryFast": 170, "Tempo_ExtremelyFast": 190}
GM_INSTRUMENT_MAP = {i: name for i, name in enumerate([
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet", "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba", "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ", "Reed Organ", "Accordion", "Harmonica", "Tango Accordion", "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)", "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics", "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass", "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2", "Violin", "Viola", "Cello", "Contrabass", "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani", "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2", "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit", "Trumpet", "Trombone", "Tuba", "Muted Trumpet", "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2", "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe", "English Horn", "Bassoon", "Clarinet", "Piccolo", "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina", "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)", "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)", "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)", "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)", "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bagpipe", "Fiddle", "Shanai", "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal", "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet", "Telephone Ring", "Helicopter", "Applause", "Gunshot"
])}

def analyze_generated_midi(midi_path: Path) -> dict:
    try:
        mid = mido.MidiFile(str(midi_path))
        note_on_count = 0
        time_signature = None
        found_instruments = set()
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_on_count += 1
                if not time_signature and msg.is_meta and msg.type == 'time_signature':
                    time_signature = f"{msg.numerator}/{msg.denominator}"
                if msg.type == 'program_change' and hasattr(msg, 'channel') and msg.channel != 9:
                    instrument_name = GM_INSTRUMENT_MAP.get(msg.program, "Unknown")
                    found_instruments.add(instrument_name)
        if not found_instruments and note_on_count > 0:
            found_instruments.add("Acoustic Grand Piano")
        analysis = {'note_count': note_on_count, 'midi_instruments': list(found_instruments)}
        if time_signature:
            analysis['time_signature'] = time_signature
        return analysis
    except Exception as e:
        logging.warning(f"Impossibile analizzare il file MIDI generato {midi_path}: {e}")
        return {'note_count': 0, 'midi_instruments': []}

def generate_random_metadata_dict(metadata_vocab_map: dict) -> dict:
    categorized_tokens = {'TimeSig': [], 'Tempo': [], 'Instrument': []}
    for token in metadata_vocab_map.keys():
        if token.startswith('TimeSig='): categorized_tokens['TimeSig'].append(token)
        elif token.startswith('Tempo_'): categorized_tokens['Tempo'].append(token)
        elif token.startswith('Instrument='): categorized_tokens['Instrument'].append(token)
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
    scores = {}
    total_score = 0
    num_criteria = 0
    prompt_tokens = set(tokenize_metadata(prompt_dict))
    analysis_tokens = set(tokenize_metadata(analysis_dict))
    for category in ['TimeSig', 'Tempo', 'NumInst']:
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
    scores['overall_adherence'] = total_score / num_criteria if num_criteria > 0 else 0.0
    return scores

LEARNING_RATE_RL = 1e-6
NUM_EPISODES = 1000
BATCH_SIZE_RL = 4

def get_generation_and_reward(model, midi_tokenizer, metadata_vocab_map, metadata_prompt_dict, device):
    metadata_prompt_str_list = tokenize_metadata(metadata_prompt_dict)
    gen_config = {"temperature": 1.0, "top_k": 0, "max_rest_penalty": 0.0, "rest_penalty_mode": 'hybrid'}
    model_chunk_capacity = model.positional_encoding.pe.size(1)

    # MODIFICA: Attivazione della modalità di training per il calcolo dei gradienti
    generated_token_ids, log_probs_tensor = generate_multi_chunk_midi(
        model, midi_tokenizer, metadata_vocab_map, metadata_prompt_str_list,
        total_target_tokens=config.MAX_SEQ_LEN_MIDI,
        model_chunk_capacity=model_chunk_capacity,
        generation_config=gen_config,
        device=device,
        initial_primer_ids=None,
        rest_ids=torch.tensor([i for t, i in midi_tokenizer.vocab.items() if t.startswith("Rest_")], device=DEVICE),
        training_mode=True  # <-- Attiva i gradienti
    )

    if not generated_token_ids or log_probs_tensor is None or log_probs_tensor.numel() == 0:
        logging.warning("Generazione fallita o vuota, reward = -1.")
        return None, -1.0, None

    temp_midi_file = None
    reward = 0.0
    try:
        # MODIFICA: Correzione della chiamata a decode
        temp_score = midi_tokenizer.decode(generated_token_ids)
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

def train_rl(model, optimizer, midi_tokenizer, metadata_vocab_map):
    model.train()
    pbar = tqdm(range(NUM_EPISODES), desc="RL Training Episodes")
    batch_rewards = []
    batch_losses = []

    # --- MODIFICA: Uso della nuova API torch.amp per compatibilità futura ---
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    for episode in pbar:
        model.eval()

        # --- FIX: Reinserita la generazione del prompt ad ogni episodio ---
        metadata_prompt_dict = generate_random_metadata_dict(metadata_vocab_map)

        # --- MODIFICA: Uso della nuova API torch.amp con device_type ---
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
            generated_tokens, reward, log_probs_tensor = get_generation_and_reward(
                model, midi_tokenizer, metadata_vocab_map, metadata_prompt_dict, DEVICE
            )

            if generated_tokens is None or log_probs_tensor is None:
                continue

            if log_probs_tensor.requires_grad:
                loss = -log_probs_tensor.sum() * reward
                if len(generated_tokens) > 0:
                    loss = loss / len(generated_tokens)
                batch_losses.append(loss)

        model.train()
        batch_rewards.append(reward)

        if (episode + 1) % BATCH_SIZE_RL == 0:
            if not batch_losses:
                logging.warning(f"Batch {episode // BATCH_SIZE_RL} vuoto o senza gradienti. Skip.")
                batch_rewards.clear()
                continue

            total_batch_loss = torch.stack(batch_losses).sum()
            optimizer.zero_grad()
            
            scaler.scale(total_batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            avg_batch_reward = sum(batch_rewards) / len(batch_rewards)
            pbar.set_postfix({"Avg Reward": f"{avg_batch_reward:.4f}", "Loss": f"{total_batch_loss.item():.4f}"})

            batch_rewards.clear()
            batch_losses.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegui il fine-tuning di un modello musicale con Reinforcement Learning.")
    parser.add_argument("--base_data_dir", type=Path, required=True, help="Percorso della cartella base del dataset.")
    parser.add_argument("--midi_vocab_path", type=Path, default=None, help="Percorso specifico al file midi_vocab.json.")
    parser.add_argument("--metadata_vocab_path", type=Path, default=None, help="Percorso specifico al file metadata_vocab.json.")
    parser.add_argument("--resume_rl_checkpoint", type=Path, default=None, help="Percorso opzionale a un checkpoint RL per riprendere.")
    args = parser.parse_args()

    MODEL_SAVE_DIR = args.base_data_dir / "rl_models"
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    MIDI_VOCAB_PATH = args.midi_vocab_path if args.midi_vocab_path else args.base_data_dir / "midi_vocab.json"
    METADATA_VOCAB_PATH = args.metadata_vocab_path if args.metadata_vocab_path else args.base_data_dir / "metadata_vocab.json"

    logging.info(f"Percorso Vocabolario MIDI in uso: {MIDI_VOCAB_PATH}")
    logging.info(f"Percorso Vocabolario Metadati in uso: {METADATA_VOCAB_PATH}")

    try:
        midi_tokenizer = build_or_load_tokenizer_rl(vocab_path=MIDI_VOCAB_PATH)
        metadata_vocab_map = load_metadata_vocab(METADATA_VOCAB_PATH)
    except (FileNotFoundError, IOError, ValueError) as e:
        logging.error(f"Impossibile inizializzare i componenti. Errore: {e}")
        sys.exit(1)

    if args.resume_rl_checkpoint and args.resume_rl_checkpoint.exists():
        checkpoint_path = args.resume_rl_checkpoint
        logging.info(f"Ripresa dell'addestramento RL dal checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = args.base_data_dir / "models" / "transformer_best.pt"
        logging.info(f"Avvio di un nuovo addestramento RL dal modello supervisionato: {checkpoint_path}")
        if not checkpoint_path.exists():
            logging.error(f"Checkpoint del modello pre-addestrato non trovato: {checkpoint_path}.")
            sys.exit(1)

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

    train_rl(model, optimizer, midi_tokenizer, metadata_vocab_map)

    final_rl_model_path = MODEL_SAVE_DIR / "transformer_rl_final.pt"
    torch.save({
        'episode': NUM_EPISODES,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_params': model_params,
    }, final_rl_model_path)
    logging.info(f"Modello RL finale salvato in {final_rl_model_path}")