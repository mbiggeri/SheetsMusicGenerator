# finetune_RL_octuple.py
# Script per il fine-tuning di modelli musicali Octuple con Reinforcement Learning.
# Include la modalità standard (GPU/CPU) e una modalità --multiprocessing su CPU.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import json
import logging
import random
import sys
import os
from tqdm import tqdm
import numpy as np
import argparse
import gc
from symusic import Score
import miditok
import config

# NUOVA IMPORTAZIONE PER IL MULTIPROCESSING
import torch.multiprocessing as mp 

# --- USAGE ---
# Modalità Standard (GPU se disponibile):
# python finetune_RL_octuple.py --base_data_dir "percorso/dati" --model_checkpoint "percorso/modello.pt"

# Modalità Multiprocessing (solo CPU):
# python finetune_RL_octuple.py --base_data_dir "percorso/dati" --model_checkpoint "percorso/modello.pt" --multiprocessing --num_workers 8

# --- CONFIGURAZIONE LOGGING E DEVICE (verrà impostato dinamicamente) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = None # Verrà definito nel blocco main

# Importazioni da altri script
try:
    from generate_music import Seq2SeqTransformerOctuple
except ImportError:
    print("ERRORE: Assicurati che il file 'generate_music.py' sia presente.")
    sys.exit(1)
    
from tokenize_metadata import tokenize_metadata
# La funzione per caricare le dimensioni del vocabolario è in training.py
from training import load_octuple_vocab_sizes

# --- IPERPARAMETRI DI TRAINING RL ---
LEARNING_RATE_RL = 5e-6
NUM_EPISODES = 1000
# Il BATCH_SIZE per l'aggiornamento dei gradienti. Con multiprocessing possiamo usare un valore più alto.
BATCH_SIZE_RL = 16 
RL_MAX_GEN_TOKENS = 300
GRADIENT_CLIP_NORM = 1.0

# --- PESI DELLA FUNZIONE DI REWARD ---
W_ADHERENCE = 1.5
W_NOTE_COUNT = 0.5
W_POLYPHONY_PENALTY = 0.1
W_SILENCE_PENALTY = 0.2
W_AVG_TEMPO = 0.7
W_AVG_VELOCITY = 0.7
POLYPHONY_THRESHOLD = 12
NOTE_COUNT_TARGET = 80

TARGET_TEMPO_MAP = {"Tempo_VerySlow": 60, "Tempo_Slow": 80, "Tempo_Moderate": 110, "Tempo_Fast": 140, "Tempo_VeryFast": 170}
TARGET_VELOCITY_MAP = {"AvgVel_Low": 60, "AvgVel_Mid": 80, "AvgVel_High": 100}

# --- Le funzioni di setup e di calcolo della ricompensa rimangono quasi invariate ---
# (Ho solo rimosso il Device esplicito da alcune chiamate perché ora è globale o specifico del worker)

def build_or_load_tokenizer_octuple_rl(vocab_path: Path):
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabolario tokenizer non trovato in {vocab_path}.")
    logging.info(f"Caricamento del tokenizer Octuple da {vocab_path} per RL.")
    tokenizer = miditok.Octuple(params=str(vocab_path))
    return tokenizer

def generate_random_metadata_dict(metadata_vocab_map: dict, only_piano: bool = False) -> dict:
    """
    Genera un dizionario di metadati casuale per creare un prompt di generazione.
    *** VERSIONE ROBUSTA CHE GARANTISCE UN PROMPT COMPLETO ***
    """
    categorized_tokens = {'TimeSig': [], 'Tempo': [], 'Instrument': []}
    
    # --- 1. Popola le liste di token disponibili dal vocabolario ---
    instrument_pool = []
    if only_piano:
        # CORREZIONE: Filtra i token in base al nome, non a un program number
        for token in metadata_vocab_map.keys():
            if token.startswith('Instrument='):
                # Estrai il nome dello strumento
                instrument_name = token.split('=')[1]
                # Controlla se è un tipo di pianoforte in base al nome
                if 'piano' in instrument_name.lower():
                    instrument_pool.append(token)
    else:
        # Questo blocco era già corretto
        instrument_pool = [t for t in metadata_vocab_map.keys() if t.startswith('Instrument=')]

    timesig_pool = [t for t in metadata_vocab_map.keys() if t.startswith('TimeSig=')]
    tempo_pool = [t for t in metadata_vocab_map.keys() if t.startswith('Tempo_')]

    prompt_dict = {}

    # --- 2. Scegli i valori, usando un default se la lista è vuota ---

    # Scelta dello strumento
    if instrument_pool:
        chosen_instrument_tokens = random.sample(instrument_pool, k=min(1, len(instrument_pool)))
        instrument_names = [tok.split('=')[1].replace('_', ' ') for tok in chosen_instrument_tokens]
        prompt_dict['midi_instruments'] = instrument_names
    else:
        # Fallback a un default se nessun strumento è disponibile (es. con --only_piano e vocabolario senza piani)
        logging.warning("Nessun token strumento valido trovato, uso 'Piano' come default.")
        prompt_dict['midi_instruments'] = ['0'] # Programma 0: Acoustic Grand Piano

    # Scelta del Time Signature
    if timesig_pool:
        prompt_dict['time_signature'] = random.choice(timesig_pool).split('=')[1]
    else:
        # Fallback a un default
        logging.warning("Nessun token TimeSig valido trovato, uso '4/4' come default.")
        prompt_dict['time_signature'] = '4/4'

    # Scelta del Tempo (BPM)
    if tempo_pool:
        # Estrai solo la parte del nome (es. 'Moderate' da 'Tempo_Moderate')
        prompt_dict['bpm'] = random.choice(tempo_pool).split('_')[1]
    else:
        # Fallback a un default
        logging.warning("Nessun token Tempo valido trovato, uso 'Moderate' come default.")
        prompt_dict['bpm'] = 'Moderate'
        
    return prompt_dict

def analyze_generated_score_for_reward(score) -> dict:
    # Questa funzione rimane identica a prima
    analysis = {'note_count': 0, 'max_polyphony': 0, 'longest_silence_ticks': 0, 'average_velocity': 0.0, 'average_tempo': 120.0}
    if not score.tracks or not any(track.notes for track in score.tracks): return analysis
    all_notes = [note for track in score.tracks for note in track.notes]
    if not all_notes: return analysis
    analysis['note_count'] = len(all_notes)
    if analysis['note_count'] > 0: analysis['average_velocity'] = sum(n.velocity for n in all_notes) / len(all_notes)
    if score.tempos: analysis['average_tempo'] = score.tempos[0].tempo
    all_notes.sort(key=lambda n: n.start)
    events = [(note.start, 1) for note in all_notes] + [(note.end, -1) for note in all_notes]
    events.sort(key=lambda x: x[0])
    current_polyphony, max_polyphony = 0, 0
    for _, type in events:
        current_polyphony += type
        if current_polyphony > max_polyphony: max_polyphony = current_polyphony
    analysis['max_polyphony'] = max_polyphony
    longest_silence, last_note_end = 0, 0
    if len(all_notes) > 1:
        last_note_end = all_notes[0].end
        for i in range(1, len(all_notes)):
            current_note = all_notes[i]
            if current_note.start > last_note_end:
                silence = current_note.start - last_note_end
                if silence > longest_silence: longest_silence = silence
            last_note_end = max(last_note_end, current_note.end)
    analysis['longest_silence_ticks'] = longest_silence
    return analysis

def generate_sequence_octuple_with_logprobs(model, midi_tokenizer, metadata_vocab_map, device, metadata_prompt):
    # Funzione modificata per non avere più il 'generation_config' e per usare il device passato
    model.eval()
    sos_meta_id = metadata_vocab_map[config.META_SOS_TOKEN_NAME]
    eos_meta_id = metadata_vocab_map[config.META_EOS_TOKEN_NAME]
    pad_meta_id = metadata_vocab_map[config.META_PAD_TOKEN_NAME]
    unk_meta_id = metadata_vocab_map[config.META_UNK_TOKEN_NAME]
    sos_midi_tuple = tuple(voc[config.MIDI_SOS_TOKEN_NAME] for voc in midi_tokenizer.vocab)
    eos_midi_tuple = tuple(voc[config.MIDI_EOS_TOKEN_NAME] for voc in midi_tokenizer.vocab)
    pad_midi_tuple = tuple(voc[config.MIDI_PAD_TOKEN_NAME] for voc in midi_tokenizer.vocab)
    num_components = len(midi_tokenizer.vocab)
    collected_log_probs, collected_entropies = [], []

    # Eseguiamo l'encoding del prompt senza calcolare gradienti
    with torch.no_grad():
        meta_token_ids = [metadata_vocab_map.get(token, unk_meta_id) for token in metadata_prompt]
        src_seq = torch.tensor([[sos_meta_id] + meta_token_ids + [eos_meta_id]], dtype=torch.long, device=device)
        src_padding_mask = (src_seq == pad_meta_id)
        memory = model.encode(src_seq, src_padding_mask)
        memory_key_padding_mask = src_padding_mask
        
    generated_tuples = [sos_midi_tuple]
    with torch.no_grad():
        initial_input = torch.tensor([sos_midi_tuple], dtype=torch.long, device=device).unsqueeze(0)
        _, cache = model.decode(initial_input, memory, memory_key_padding_mask=memory_key_padding_mask, cache=None)

    for _ in range(RL_MAX_GEN_TOKENS):
        next_token_tuple_in_progress = []
        input_for_component_decoder = torch.tensor([generated_tuples[-1]], dtype=torch.long, device=device).unsqueeze(0)
        for j in range(num_components):
            logits_list, new_cache_step = model.decode(input_for_component_decoder, memory, memory_key_padding_mask=memory_key_padding_mask, cache=cache)
            head_logits = logits_list[j]
            probs = F.softmax(head_logits.float(), dim=-1).squeeze(0).squeeze(0)
            log_probs = F.log_softmax(head_logits.float(), dim=-1).squeeze(0).squeeze(0)
            sampled_id = torch.multinomial(probs, num_samples=1).item()
            collected_log_probs.append(log_probs[sampled_id])
            next_token_tuple_in_progress.append(sampled_id)
            if j < num_components - 1:
                temp_list = next_token_tuple_in_progress + list(pad_midi_tuple[j+1:])
                input_for_component_decoder = torch.tensor([temp_list], dtype=torch.long, device=device).unsqueeze(0)
        final_generated_tuple = tuple(next_token_tuple_in_progress)
        generated_tuples.append(final_generated_tuple)
        cache = new_cache_step
        if final_generated_tuple == eos_midi_tuple: break
            
    final_ids = [list(t) for t in generated_tuples[1:]]
    log_probs_tensor = torch.stack(collected_log_probs) if collected_log_probs else None
    return final_ids, log_probs_tensor, None

def get_generation_and_reward_octuple(model, midi_tokenizer, metadata_vocab_map, device, metadata_prompt_dict):
    metadata_prompt_str_list = tokenize_metadata(metadata_prompt_dict)
    generated_token_ids, log_probs, entropies = generate_sequence_octuple_with_logprobs(model, midi_tokenizer, metadata_vocab_map, device, metadata_prompt_str_list)
    if not generated_token_ids or log_probs is None: return None, -2.0, None, None, None

    try:
        critical_fields = ["Pitch", "Position", "Bar", "Velocity", "Program", "TimeSig"]
        bad_ids_map = {}
        for field_name in critical_fields:
            if field_name in midi_tokenizer.vocab_types_idx:
                component_idx = midi_tokenizer.vocab_types_idx[field_name]
                component_vocab = midi_tokenizer.vocab[component_idx]
                ids_to_discard = {token_id for token_str, token_id in component_vocab.items() if token_str.endswith("_None")}
                if ids_to_discard: bad_ids_map[component_idx] = ids_to_discard
        sanitized_ids = [token_tuple for token_tuple in generated_token_ids if all(token_tuple[idx] not in bad_ids for idx, bad_ids in bad_ids_map.items() if idx < len(token_tuple))]
        if not sanitized_ids:
            logging.warning("La sanificazione ha rimosso tutti i token.")
            return None, -2.0, None, None, None
        
        reward = 0.0
        score = midi_tokenizer.decode(sanitized_ids)
        analysis = analyze_generated_score_for_reward(score)
        
        prompt_tokens = set(metadata_prompt_str_list)
        target_tempo = next((TARGET_TEMPO_MAP[t] for t in prompt_tokens if t in TARGET_TEMPO_MAP), None)
        target_velocity = next((TARGET_VELOCITY_MAP[t] for t in prompt_tokens if t in TARGET_VELOCITY_MAP), None)

        if target_tempo and analysis.get('average_tempo'):
            reward += np.exp(-0.5 * ((analysis['average_tempo'] - target_tempo) / 10.0) ** 2) * W_AVG_TEMPO
        if target_velocity and analysis.get('average_velocity'):
            reward += np.exp(-0.5 * ((analysis['average_velocity'] - target_velocity) / 10.0) ** 2) * W_AVG_VELOCITY
        reward -= max(0, analysis['max_polyphony'] - POLYPHONY_THRESHOLD) * W_POLYPHONY_PENALTY
        reward -= max(0, (analysis['longest_silence_ticks'] - (midi_tokenizer.time_division * 2)) / midi_tokenizer.time_division) * W_SILENCE_PENALTY
        reward += max(0, 1.0 - abs(analysis['note_count'] - NOTE_COUNT_TARGET) / NOTE_COUNT_TARGET) * W_NOTE_COUNT
        if analysis['note_count'] > 10: reward += W_ADHERENCE
            
    except Exception as e:
        logging.error(f"Errore imprevisto durante l'analisi per reward: {e}", exc_info=True)
        return None, -2.0, None, None, None

    return generated_token_ids, reward, log_probs, None, entropies

# --- NUOVE FUNZIONI PER IL MULTIPROCESSING ---

# Variabili globali per i worker. Vengono inizializzate una sola volta per processo.
worker_model = None
worker_tokenizer = None
worker_metadata_map = None
worker_model_params = None

def init_worker(model_state, model_params, tokenizer_obj, metadata_map_obj):
    """Funzione di inizializzazione per ogni worker del pool."""
    global worker_model, worker_tokenizer, worker_metadata_map, worker_model_params
    
    # Ricostruisce il modello e carica lo stato nel worker
    worker_model_params = model_params
    worker_model = Seq2SeqTransformerOctuple(**worker_model_params)
    worker_model.load_state_dict(model_state)
    worker_model.to(torch.device("cpu")) # Il multiprocessing lavora su CPU
    worker_model.eval()
    
    # Rende disponibili gli altri oggetti
    worker_tokenizer = tokenizer_obj
    worker_metadata_map = metadata_map_obj
    logging.info(f"Worker {os.getpid()} inizializzato.")

def run_single_rl_episode(args_tuple):
    """
    La funzione che viene eseguita da ogni worker.
    Simula un singolo episodio RL e restituisce il risultato.
    """
    episode_idx, only_piano = args_tuple
    
    # Genera un prompt casuale
    metadata_prompt_dict = generate_random_metadata_dict(worker_metadata_map, only_piano=only_piano)
    
    # Esegue generazione e calcolo della ricompensa
    # Usa le variabili globali del worker
    _, reward, log_probs, _, _ = get_generation_and_reward_octuple(
        worker_model, 
        worker_tokenizer, 
        worker_metadata_map, 
        torch.device("cpu"),
        metadata_prompt_dict
    )
    
    # Se la generazione fallisce, log_probs è None. Lo gestiamo nel ciclo principale.
    return reward, log_probs

# --- CICLO DI TRAINING RL (MODIFICATO PER SUPPORTARE ENTRAMBE LE MODALITÀ) ---

def train_rl_octuple(model, optimizer, midi_tokenizer, metadata_vocab_map, model_params, args):
    """Ciclo di training principale, con logica per seriale (GPU) o parallelo (CPU)."""
    
    # --- MODALITÀ MULTIPROCESSING ---
    if args.multiprocessing:
        num_workers = args.num_workers if args.num_workers is not None else os.cpu_count()
        logging.info(f"Avvio training RL in modalità MULTIPROCESSING con {num_workers} workers.")
        
        # mp.get_context("spawn") è più robusto su diverse piattaforme (incluso Windows)
        ctx = mp.get_context("spawn")
        
        # Passiamo lo state_dict (un dizionario) invece dell'intero modello, è più sicuro
        init_args = (model.state_dict(), model_params, midi_tokenizer, metadata_vocab_map)

        batch_rewards, batch_losses = [], []
        
        with ctx.Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
            # Creiamo i task da eseguire
            tasks = [(i, args.only_piano) for i in range(NUM_EPISODES)]
            
            pbar = tqdm(pool.imap_unordered(run_single_rl_episode, tasks), total=NUM_EPISODES, desc="RL Training (Multiprocessing)")
            
            for i, (reward, log_probs) in enumerate(pbar):
                if log_probs is None:
                    continue
                
                # Calcolo della loss
                policy_loss = -log_probs.mean() * reward
                batch_losses.append(policy_loss)
                batch_rewards.append(reward)

                # Aggiornamento del modello ogni BATCH_SIZE_RL episodi completati
                if len(batch_losses) >= BATCH_SIZE_RL:
                    total_batch_loss = torch.stack(batch_losses).mean()
                    
                    optimizer.zero_grad()
                    total_batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
                    optimizer.step()

                    avg_batch_reward = np.mean(batch_rewards)
                    pbar.set_postfix({"Avg Reward": f"{avg_batch_reward:.4f}", "Loss": f"{total_batch_loss.item():.4f}"})
                    
                    del total_batch_loss, batch_losses, batch_rewards
                    batch_rewards, batch_losses = [], []
                    gc.collect()

    # --- MODALITÀ STANDARD (SERIALE) ---
    else:
        logging.info(f"Avvio training RL in modalità STANDARD su device: {DEVICE}")
        model.train()
        pbar = tqdm(range(NUM_EPISODES), desc="RL Training Episodes (Standard)")
        batch_rewards, batch_losses = [], []
        scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

        for episode in pbar:
            metadata_prompt_dict = generate_random_metadata_dict(metadata_vocab_map, only_piano=args.only_piano)
            
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                _, reward, log_probs, _, _ = get_generation_and_reward_octuple(
                    model, midi_tokenizer, metadata_vocab_map, DEVICE, metadata_prompt_dict
                )
                if log_probs is None: continue
                policy_loss = -log_probs.mean() * reward

            batch_losses.append(policy_loss)
            batch_rewards.append(reward)

            if (episode + 1) % BATCH_SIZE_RL == 0 and batch_losses:
                total_batch_loss = torch.stack(batch_losses).mean()
                optimizer.zero_grad()
                scaler.scale(total_batch_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                avg_batch_reward = np.mean(batch_rewards)
                pbar.set_postfix({"Avg Reward": f"{avg_batch_reward:.4f}", "Loss": f"{total_batch_loss.item():.4f}"})
                del total_batch_loss, batch_losses, batch_rewards
                batch_rewards, batch_losses = [], []
                gc.collect()
                if DEVICE.type == 'cuda': torch.cuda.empty_cache()


# --- BLOCCO DI ESECUZIONE PRINCIPALE (MODIFICATO) ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegui il fine-tuning di un modello Octuple con RL.")
    parser.add_argument("--base_data_dir", type=Path, required=True)
    parser.add_argument("--model_checkpoint", type=Path, required=True)
    parser.add_argument("--only_piano", action="store_true", help="Limita i prompt a strumenti della famiglia dei pianoforti.")
    
    # NUOVI ARGOMENTI PER IL MULTIPROCESSING
    parser.add_argument("--multiprocessing", action="store_true", help="Usa il multiprocessing su CPU per l'addestramento RL.")
    parser.add_argument("--num_workers", type=int, default=None, help="Numero di processi worker. Default: tutti i core disponibili.")
    
    args = parser.parse_args()

    # IMPOSTAZIONE DINAMICA DEL DEVICE E DELLE OPZIONI
    if args.multiprocessing:
        DEVICE = torch.device("cpu")
        # Aumentiamo il batch size per gli aggiornamenti, dato che processiamo più episodi in parallelo
        BATCH_SIZE_RL = 32 
        logging.info(f"MODALITÀ MULTIPROCESSING ATTIVA. Device impostato su CPU. Batch size per update: {BATCH_SIZE_RL}")
        # Necessario per CUDA in modalità 'spawn'
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
             mp.set_start_method('spawn', force=True)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        BATCH_SIZE_RL = 4 # Batch size più piccolo per la modalità seriale/GPU
        logging.info(f"MODALITÀ STANDARD ATTIVA. Device impostato su {DEVICE}. Batch size per update: {BATCH_SIZE_RL}")

    # Setup percorsi (invariato)
    MIDI_VOCAB_PATH = args.base_data_dir / "midi_vocab.json"
    METADATA_VOCAB_PATH = args.base_data_dir / "metadata_vocab.json"
    RL_MODEL_SAVE_DIR = args.base_data_dir / "rl_models_octuple"
    RL_MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        midi_tokenizer = build_or_load_tokenizer_octuple_rl(vocab_path=MIDI_VOCAB_PATH)
        from training import load_metadata_vocab
        metadata_vocab_map = load_metadata_vocab(METADATA_VOCAB_PATH)
    except Exception as e:
        logging.error(f"Impossibile inizializzare. Errore: {e}")
        sys.exit(1)
        
    # Caricamento modello (ora consapevole del DEVICE)
    checkpoint = torch.load(args.model_checkpoint, map_location=DEVICE)
    model_params = checkpoint.get('model_params')
    if not model_params or 'tgt_vocab_sizes' not in model_params:
        logging.error("Checkpoint non valido per Octuple.")
        sys.exit(1)
        
    model = Seq2SeqTransformerOctuple(**model_params).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info("Modello Octuple da addestrare caricato.")

    # In modalità multiprocessing, il modello principale viene solo aggiornato, non esegue calcoli.
    # In modalità seriale, viene messo in modalità training.
    if not args.multiprocessing:
        model.train()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_RL)
    
    # La funzione di training ora riceve i parametri del modello e gli args del parser
    train_rl_octuple(model, optimizer, midi_tokenizer, metadata_vocab_map, model_params, args)

    # Salvataggio finale (invariato)
    final_rl_model_path = RL_MODEL_SAVE_DIR / "transformer_octuple_rl_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_params': model_params,
    }, final_rl_model_path)
    logging.info(f"Modello RL Octuple finale salvato in {final_rl_model_path}")