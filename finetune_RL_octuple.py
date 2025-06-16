# finetune_RL_octuple.py
# Script per il fine-tuning di modelli musicali Octuple con Reinforcement Learning.
# Include funzioni di reward per coerenza ai metadati, penalità per silenzio e polifonia.

import torch
import torch.nn as nn # Assicurati che sia importato all'inizio del file
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

# --- USAGE:
# python finetune_RL_octuple.py --base_data_dir "percorso/al/tuo/dataset" --model_checkpoint "percorso/al/tuo/modello.pt" --only_piano

# --- EXAMPLE:
# python finetune_RL_octuple.py --base_data_dir "C:\Users\Michael\Desktop\ModelliMusicGenerator\Octuple_onlyPiano_v1" --model_checkpoint "C:\Users\Michael\Desktop\ModelliMusicGenerator\Octuple_onlyPiano_v1\transformer_best_small.pt" --only_piano

# Importa le classi e funzioni necessarie, specifiche per Octuple
# Assicurati che il file 'generate_music.py' sia aggiornato e contenga questa funzione
try:
    from generate_music import Seq2SeqTransformerOctuple, enhance_midi_score
except ImportError:
    print("ERRORE: Assicurati che il file 'generate_music.py' sia presente e contenga la classe 'Seq2SeqTransformerOctuple'.")
    sys.exit(1)
    
from tokenize_metadata import tokenize_metadata
from training import load_metadata_vocab, load_octuple_vocab_sizes


# --- CONFIGURAZIONE GLOBALE ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- IPERPARAMETRI DI TRAINING RL ---
# Puoi modificare questi valori per regolare il comportamento del fine-tuning
LEARNING_RATE_RL = 5e-6
NUM_EPISODES = 1000
BATCH_SIZE_RL = 1  # Ridotto per Octuple che è più pesante in memoria
RL_MAX_GEN_TOKENS = 300
GRADIENT_CLIP_NORM = 1.0

# --- PESI DELLA FUNZIONE DI REWARD ---
# Sperimenta con questi pesi per guidare il modello
W_ADHERENCE = 1.5           # Reward per la coerenza con i metadati del prompt
W_NOTE_COUNT = 0.5          # Bonus per generare un numero ragionevole di note
W_POLYPHONY_PENALTY = 0.1   # Penalità per ogni nota che supera la soglia di polifonia
W_SILENCE_PENALTY = 0.2     # Penalità per i silenzi troppo lunghi
W_AVG_TEMPO = 0.7           # Reward per l'aderenza al tempo target
W_AVG_VELOCITY = 0.7        # Reward per l'aderenza alla velocity media target
POLYPHONY_THRESHOLD = 12    # Numero massimo di note simultanee prima di applicare la penalità
NOTE_COUNT_TARGET = 80      # Numero di note ideale per la lunghezza di generazione data

# Dizionari per mappare i token del prompt a valori numerici target
TARGET_TEMPO_MAP = {"Tempo_VerySlow": 60, "Tempo_Slow": 80, "Tempo_Moderate": 110, "Tempo_Fast": 140, "Tempo_VeryFast": 170}
TARGET_VELOCITY_MAP = {"AvgVel_Low": 60, "AvgVel_Mid": 80, "AvgVel_High": 100}

# --- IPERPARAMETRI PER RL AVANZATO (FACOLTATIVI) ---
# Decommenta e usa queste sezioni per un training più stabile e avanzato
USE_KL_DIVERGENCE_PENALTY = False
W_KL = 0.01 # Peso della penalità KL per evitare catastrophic forgetting

USE_ENTROPY_BONUS = False
W_ENTROPY = 0.01 # Peso del bonus di entropia per incoraggiare l'esplorazione


# --- FUNZIONI DI SETUP ---

def build_or_load_tokenizer_octuple_rl(vocab_path: Path):
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabolario tokenizer non trovato in {vocab_path}.")
    logging.info(f"Caricamento del tokenizer Octuple da {vocab_path} per RL.")
    tokenizer = miditok.Octuple(params=str(vocab_path))
    logging.info(f"Tokenizer Octuple per RL pronto.")
    return tokenizer

def generate_random_metadata_dict(metadata_vocab_map: dict, only_piano: bool = False) -> dict:
    """
    Genera un dizionario di metadati casuale per creare un prompt di generazione.
    *** VERSIONE AGGIORNATA CON FILTRO PER PIANOFORTE ***
    """
    categorized_tokens = {'TimeSig': [], 'Tempo': [], 'Instrument': []}
    
    # --- LOGICA DI FILTRAGGIO DEGLI STRUMENTI ---
    # Se il flag --only_piano è attivo, considera solo gli strumenti nel range 0-7.
    if only_piano:
        piano_family_range = range(0, 8)
        for token in metadata_vocab_map.keys():
            if token.startswith('Instrument='):
                try:
                    # Estrae il numero del programma e lo aggiunge solo se è un pianoforte
                    prog_num = int(token.split('=')[1])
                    if prog_num in piano_family_range:
                        categorized_tokens['Instrument'].append(token)
                except (ValueError, IndexError):
                    continue # Ignora token malformati
            elif token.startswith('TimeSig='):
                categorized_tokens['TimeSig'].append(token)
            elif token.startswith('Tempo_'):
                categorized_tokens['Tempo'].append(token)
    else:
        # Logica originale: usa tutti gli strumenti disponibili nel vocabolario
        for token in metadata_vocab_map.keys():
            if token.startswith('Instrument='): categorized_tokens['Instrument'].append(token)
            elif token.startswith('TimeSig='): categorized_tokens['TimeSig'].append(token)
            elif token.startswith('Tempo_'): categorized_tokens['Tempo'].append(token)

    prompt_dict = {}
    if categorized_tokens['Instrument']:
        num_instruments = 1
        # Assicurati di non chiedere più strumenti di quelli disponibili dopo il filtraggio
        if len(categorized_tokens['Instrument']) > 0:
            chosen_instrument_tokens = random.sample(categorized_tokens['Instrument'], k=min(num_instruments, len(categorized_tokens['Instrument'])))
            instrument_names = [tok.split('=')[1].replace('_', ' ') for tok in chosen_instrument_tokens]
            prompt_dict['midi_instruments'] = instrument_names
            
    # Scegli casualmente gli altri metadati
    if categorized_tokens['TimeSig']:
        prompt_dict['time_signature'] = random.choice(categorized_tokens['TimeSig']).split('=')[1]
    if categorized_tokens['Tempo']:
        prompt_dict['bpm'] = random.choice(categorized_tokens['Tempo']).split('_')[1]

    return prompt_dict


# --- FUNZIONI DI REWARD ---

def analyze_generated_score_for_reward(score) -> dict:
    """
    Analizza un oggetto Score di symusic per calcolare le metriche necessarie per il reward.
    *** VERSIONE AGGIORNATA CON TEMPO E VELOCITY ***
    """
    analysis = {
        'note_count': 0,
        'max_polyphony': 0,
        'longest_silence_ticks': 0,
        'average_velocity': 0.0,
        'average_tempo': 120.0 # Default a 120 BPM
    }
    
    if not score.tracks or not any(track.notes for track in score.tracks):
        return analysis

    all_notes = [note for track in score.tracks for note in track.notes]
    if not all_notes: return analysis

    analysis['note_count'] = len(all_notes)

    # Calcolo Average Velocity
    analysis['average_velocity'] = sum(n.velocity for n in all_notes) / len(all_notes)

    # Calcolo Average Tempo (usando il primo evento di tempo come rappresentativo)
    if score.tempos:
        analysis['average_tempo'] = score.tempos[0].tempo
        
    all_notes.sort(key=lambda n: n.start)

    # Calcolo Polifonia Massima (invariato)
    events = [(note.start, 1) for note in all_notes] + [(note.end, -1) for note in all_notes]
    events.sort(key=lambda x: x[0])
    current_polyphony, max_polyphony = 0, 0
    for _, type in events:
        current_polyphony += type
        if current_polyphony > max_polyphony:
            max_polyphony = current_polyphony
    analysis['max_polyphony'] = max_polyphony

    # Calcolo Silenzio Più Lungo (invariato)
    longest_silence, last_note_end = 0, 0
    if len(all_notes) > 1:
        last_note_end = all_notes[0].end
        for i in range(1, len(all_notes)):
            current_note = all_notes[i]
            if current_note.start > last_note_end:
                silence = current_note.start - last_note_end
                if silence > longest_silence:
                    longest_silence = silence
            last_note_end = max(last_note_end, current_note.end)
    analysis['longest_silence_ticks'] = longest_silence
    
    return analysis


# --- FUNZIONE DI GENERAZIONE MODIFICATA PER RL ---

def generate_sequence_octuple_with_logprobs(model, midi_tokenizer, metadata_vocab_map, generation_config, metadata_prompt, device):
    """
    Funzione di generazione specifica per Octuple che restituisce anche le log-probabilità
    delle azioni (token) scelte, con gestione corretta dei gradienti per RL.
    """
    model.eval()
    
    # Setup iniziale token speciali (invariato)
    sos_meta_id = metadata_vocab_map[config.META_SOS_TOKEN_NAME]
    eos_meta_id = metadata_vocab_map[config.META_EOS_TOKEN_NAME]
    pad_meta_id = metadata_vocab_map[config.META_PAD_TOKEN_NAME]
    unk_meta_id = metadata_vocab_map[config.META_UNK_TOKEN_NAME]
    
    sos_midi_tuple = tuple(voc[config.MIDI_SOS_TOKEN_NAME] for voc in midi_tokenizer.vocab)
    eos_midi_tuple = tuple(voc[config.MIDI_EOS_TOKEN_NAME] for voc in midi_tokenizer.vocab)
    pad_midi_tuple = tuple(voc[config.MIDI_PAD_TOKEN_NAME] for voc in midi_tokenizer.vocab)
    num_components = len(midi_tokenizer.vocab)
    
    collected_log_probs = []
    collected_entropies = []
    
    # --- NUOVA GESTIONE DI NO_GRAD ---
    # 1. Eseguiamo l'encoding del prompt senza calcolare gradienti, perché è solo un input.
    with torch.no_grad():
        meta_token_ids = [metadata_vocab_map.get(token, unk_meta_id) for token in metadata_prompt]
        src_seq = torch.tensor([[sos_meta_id] + meta_token_ids + [eos_meta_id]], dtype=torch.long, device=device)
        src_padding_mask = (src_seq == pad_meta_id)
        memory = model.encode(src_seq, src_padding_mask)
        memory_key_padding_mask = src_padding_mask

    # 2. Il ciclo di generazione ora opera CON i gradienti attivi.
    generated_tuples = [sos_midi_tuple]
    
    # Inizializzazione del cache (fuori dal ciclo principale)
    with torch.no_grad(): # Il primo passo di cache può essere fatto senza gradienti
        initial_input = torch.tensor([sos_midi_tuple], dtype=torch.long, device=device).unsqueeze(0)
        _, cache = model.decode(initial_input, memory, memory_key_padding_mask=memory_key_padding_mask, cache=None)

    for _ in range(RL_MAX_GEN_TOKENS):
        next_token_tuple_in_progress = []
        # L'input per il decoder è l'ultimo token generato
        input_for_component_decoder = torch.tensor([generated_tuples[-1]], dtype=torch.long, device=device).unsqueeze(0)

        for j in range(num_components):
            # QUI NON USIAMO PIÙ NO_GRAD
            logits_list, new_cache_step = model.decode(input_for_component_decoder, memory, memory_key_padding_mask=memory_key_padding_mask, cache=cache)
            head_logits = logits_list[j]
            
            probs = F.softmax(head_logits.float(), dim=-1).squeeze(0).squeeze(0)
            log_probs = F.log_softmax(head_logits.float(), dim=-1).squeeze(0).squeeze(0)

            entropy = -(probs * log_probs).sum()
            collected_entropies.append(entropy)
            
            sampled_id = torch.multinomial(probs, num_samples=1).item()
            
            collected_log_probs.append(log_probs[sampled_id])
            
            next_token_tuple_in_progress.append(sampled_id)
            
            # Per il prossimo componente, usiamo il token appena campionato per l'autoregressione
            if j < num_components - 1:
                temp_list = next_token_tuple_in_progress + list(pad_midi_tuple[j+1:])
                input_for_component_decoder = torch.tensor([temp_list], dtype=torch.long, device=device).unsqueeze(0)

        final_generated_tuple = tuple(next_token_tuple_in_progress)
        generated_tuples.append(final_generated_tuple)
        
        # Aggiorniamo il cache per il prossimo step temporale
        cache = new_cache_step
        
        if final_generated_tuple == eos_midi_tuple:
            break

    # Il resto della funzione (return) rimane invariato
    final_ids = [list(t) for t in generated_tuples[1:]]
    log_probs_tensor = torch.stack(collected_log_probs) if collected_log_probs else None
    entropies_tensor = torch.stack(collected_entropies) if collected_entropies and USE_ENTROPY_BONUS else None
    
    return final_ids, log_probs_tensor, entropies_tensor


# --- FUNZIONE PRINCIPALE DI ORCHESTRAZIONE RL ---

def get_generation_and_reward_octuple(model, ref_model, midi_tokenizer, metadata_vocab_map, metadata_prompt_dict, device):
    metadata_prompt_str_list = tokenize_metadata(metadata_prompt_dict)
    
    # La chiamata a generate_sequence_octuple_with_logprobs era errata.
    # Mancava 'metadata_vocab_map' e gli argomenti erano nell'ordine sbagliato.
    
    # *** CHIAMATA CORRETTA ***
    generated_token_ids, log_probs, entropies = generate_sequence_octuple_with_logprobs(
        model, 
        midi_tokenizer, 
        metadata_vocab_map, # <-- Questo argomento mancava e sfasava gli altri
        {"temperature": 1.1, "top_k": 40}, # generation_config
        metadata_prompt_str_list, # metadata_prompt
        device # device
    )

    if not generated_token_ids or log_probs is None:
        return None, -2.0, None, None, None

    try:
        # --- INIZIO BLOCCO DI SANIFICAZIONE ---
        # Questo blocco previene il crash "invalid literal for int() with base 10: 'None'"
        
        # 1. Identifica gli indici dei componenti critici che non possono essere 'None'
        critical_fields = ["Pitch", "Position", "Bar", "Velocity", "Program", "TimeSig"]
        bad_ids_map = {}
        for field_name in critical_fields:
            if field_name in midi_tokenizer.vocab_types_idx:
                component_idx = midi_tokenizer.vocab_types_idx[field_name]
                component_vocab = midi_tokenizer.vocab[component_idx]
                
                # Trova tutti gli ID che corrispondono a token 'None' (es. PAD_None, Program_None)
                ids_to_discard = {
                    token_id for token_str, token_id in component_vocab.items() 
                    if token_str.endswith("_None")
                }
                if ids_to_discard:
                    bad_ids_map[component_idx] = ids_to_discard

        # 2. Filtra le tuple di token generate, scartando quelle non valide
        sanitized_ids = []
        for token_tuple in generated_token_ids:
            is_valid = True
            for component_idx, bad_ids_set in bad_ids_map.items():
                if component_idx < len(token_tuple) and token_tuple[component_idx] in bad_ids_set:
                    is_valid = False
                    break
            if is_valid:
                sanitized_ids.append(token_tuple)
        
        # Se la sanificazione ha rimosso tutti i token, la generazione non è valida.
        if not sanitized_ids:
            logging.warning("La sanificazione ha rimosso tutti i token. La generazione è vuota.")
            return None, -2.0, None, None, None # Ricompensa negativa per generazioni vuote/invalide
            
        # --- FINE BLOCCO DI SANIFICAZIONE ---

        # 3. Procedi con la decodifica e l'analisi usando i token puliti
        reward = 0.0
        score = midi_tokenizer.decode(sanitized_ids)
        analysis = analyze_generated_score_for_reward(score)
        
        # ESTRAZIONE DEI VALORI TARGET DAL PROMPT
        prompt_tokens = set(metadata_prompt_str_list)
        target_tempo = next((TARGET_TEMPO_MAP[t] for t in prompt_tokens if t in TARGET_TEMPO_MAP), None)
        target_velocity = next((TARGET_VELOCITY_MAP[t] for t in prompt_tokens if t in TARGET_VELOCITY_MAP), None)

        # REWARD PER TEMPO E VELOCITY
        if target_tempo and analysis['average_tempo'] is not None:
            sigma_tempo = 10.0 
            tempo_reward = np.exp(-0.5 * ((analysis['average_tempo'] - target_tempo) / sigma_tempo) ** 2)
            reward += tempo_reward * W_AVG_TEMPO

        if target_velocity and analysis['average_velocity'] is not None:
            sigma_velocity = 10.0
            velocity_reward = np.exp(-0.5 * ((analysis['average_velocity'] - target_velocity) / sigma_velocity) ** 2)
            reward += velocity_reward * W_AVG_VELOCITY

        # PENALITÀ E BONUS ESISTENTI
        polyphony_penalty = max(0, analysis['max_polyphony'] - POLYPHONY_THRESHOLD)
        reward -= polyphony_penalty * W_POLYPHONY_PENALTY 
        
        silence_penalty = max(0, analysis['longest_silence_ticks'] - (midi_tokenizer.time_division * 2))
        reward -= (silence_penalty / midi_tokenizer.time_division) * W_SILENCE_PENALTY
        
        note_count_reward = 1.0 - abs(analysis['note_count'] - NOTE_COUNT_TARGET) / NOTE_COUNT_TARGET
        reward += max(0, note_count_reward) * W_NOTE_COUNT

        if analysis['note_count'] > 10:
            reward += W_ADHERENCE
            
    except Exception as e:
        # Questo blocco ora catturerà solo errori imprevisti, non più il crash comune
        logging.error(f"Errore imprevisto durante l'analisi per reward: {e}", exc_info=True)
        return None, -2.0, None, None, None

    kl_div = None # Placeholder
    return generated_token_ids, reward, log_probs, kl_div, entropies


# --- CICLO DI TRAINING RL ---

def train_rl_octuple(model, ref_model, optimizer, midi_tokenizer, metadata_vocab_map, only_piano: bool = False):
    """ *** VERSIONE AGGIORNATA CON STAMPA DI DEBUG E PASSAGGIO DEL FLAG *** """
    model.train()
    pbar = tqdm(range(NUM_EPISODES), desc="RL Training Episodes (Octuple)")
    batch_rewards, batch_losses = [], []
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    for episode in pbar:
        # Passa il flag 'only_piano' alla funzione che genera il prompt
        metadata_prompt_dict = generate_random_metadata_dict(metadata_vocab_map, only_piano=only_piano)
        
        # *** STAMPA DI DEBUG AGGIUNTA ***
        # Stampa il prompt ogni N episodi per non inondare la console
        if (episode + 1) % 10 == 0:
             print(f"\n--- Episodio {episode + 1} | Prompt: {metadata_prompt_dict} ---")

        # ... il resto della logica del ciclo di training rimane invariato ...
        # (chiama get_generation_and_reward_octuple, calcola la loss, etc.)

        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
            _, reward, log_probs, kl_div, entropies = get_generation_and_reward_octuple(
                model, ref_model, midi_tokenizer, metadata_vocab_map, metadata_prompt_dict, DEVICE
            )

            if log_probs is None: continue
            
            policy_loss = -log_probs.mean() * reward
            
            if kl_div is not None and USE_KL_DIVERGENCE_PENALTY:
                policy_loss += W_KL * kl_div
            if entropies is not None and USE_ENTROPY_BONUS:
                policy_loss -= W_ENTROPY * entropies.mean()

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


# --- BLOCCO DI ESECUZIONE ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegui il fine-tuning di un modello Octuple con RL.")
    parser.add_argument("--base_data_dir", type=Path, required=True)
    parser.add_argument("--model_checkpoint", type=Path, required=True)
    parser.add_argument("--only_piano", action="store_true", help="Limita i prompt di generazione casuale a includere solo strumenti della famiglia dei pianoforti (programmi 0-7).")
    args = parser.parse_args()

    # Setup percorsi
    MIDI_VOCAB_PATH = args.base_data_dir / "midi_vocab.json"
    METADATA_VOCAB_PATH = args.base_data_dir / "metadata_vocab.json"
    MIDI_VOCAB_SIZES_PATH = args.base_data_dir / "midi_vocab_sizes.json"
    RL_MODEL_SAVE_DIR = args.base_data_dir / "rl_models_octuple"
    RL_MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        midi_tokenizer = build_or_load_tokenizer_octuple_rl(vocab_path=MIDI_VOCAB_PATH)
        metadata_vocab_map = load_metadata_vocab(METADATA_VOCAB_PATH)
    except Exception as e:
        logging.error(f"Impossibile inizializzare. Errore: {e}")
        sys.exit(1)
        
    # Caricamento modello da addestrare
    checkpoint = torch.load(args.model_checkpoint, map_location=DEVICE)
    model_params = checkpoint.get('model_params')
    if not model_params or 'tgt_vocab_sizes' not in model_params:
        logging.error("Checkpoint non valido per Octuple.")
        sys.exit(1)
        
    model = Seq2SeqTransformerOctuple(**model_params).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info("Modello Octuple da addestrare caricato.")

    # Caricamento modello di riferimento (per KL divergence facoltativa)
    ref_model = None
    if USE_KL_DIVERGENCE_PENALTY:
        ref_model = Seq2SeqTransformerOctuple(**model_params).to(DEVICE)
        ref_model.load_state_dict(checkpoint['model_state_dict'])
        ref_model.eval() # Il modello di riferimento è solo per inferenza
        logging.info("Modello di riferimento per KL Divergence caricato.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_RL)
    
    train_rl_octuple(model, ref_model, optimizer, midi_tokenizer, metadata_vocab_map, only_piano=args.only_piano)

    final_rl_model_path = RL_MODEL_SAVE_DIR / "transformer_octuple_rl_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_params': model_params,
    }, final_rl_model_path)
    logging.info(f"Modello RL Octuple finale salvato in {final_rl_model_path}")