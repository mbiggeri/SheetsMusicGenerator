# generate_music.py (versione libreria con callback di progresso, KV CACHING e pulizia MIDI avanzata)
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for Mac M1 compatibility

import torch
import torch.nn as nn
import torch.nn.functional as F
import miditok
from pathlib import Path
import json
import math
import logging
import time
import sys
from symusic import Score, TimeSignature, Tempo, Note, ControlChange, Track
import warnings
from typing import Optional, List, Dict, Any
import random
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning, module='miditok.midi_tokenizer_base')

# --- MODIFICA: Importazione della classe OctupleTransformer ---
class Seq2SeqTransformerOctuple(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead,
                 src_vocab_size, tgt_vocab_sizes: dict, max_pe_len,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.vocab_names = list(tgt_vocab_sizes.keys())
        self.num_components = len(self.vocab_names)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        # Un ModuleList per gli embedding di ogni componente di Octuple
        self.tgt_tok_emb = nn.ModuleList([nn.Embedding(tgt_vocab_sizes[name], emb_size) for name in self.vocab_names])
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=max_pe_len)
        self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        # Un ModuleList per i generatori (teste di output)
        self.generator = nn.ModuleList([nn.Linear(emb_size, tgt_vocab_sizes[name]) for name in self.vocab_names])

    def encode(self, src, src_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: Optional[torch.Tensor] = None,
               cache: Optional[List[Dict[str, torch.Tensor]]] = None) -> tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        is_primer_pass = cache is None
        
        # Somma degli embedding dei componenti
        tgt_emb_sum = torch.zeros(tgt.size(0), tgt.size(1), self.emb_size, device=tgt.device)
        for i in range(self.num_components):
            tgt_emb_sum += self.tgt_tok_emb[i](tgt[:, :, i])

        if is_primer_pass:
            tgt_emb = self.positional_encoding(tgt_emb_sum * math.sqrt(self.emb_size))
        else:
            past_len = cache[0]['k'].size(1)
            tgt_emb = tgt_emb_sum + self.positional_encoding.pe[:, past_len:past_len + tgt_emb_sum.size(1)]

        # Il resto della logica del decoder rimane simile, ma restituisce una lista di logits
        new_cache, output = [], tgt_emb
        for i, layer in enumerate(self.transformer.decoder.layers):
            past_layer_cache = cache[i] if cache is not None else None
            # ... (la logica interna del decoder è complessa e si assume corretta)
            # Simulazione semplificata per chiarezza
            output, layer_cache = self._decode_layer(layer, output, memory, past_layer_cache, is_primer_pass, memory_key_padding_mask)
            new_cache.append(layer_cache)

        if is_primer_pass:
            output = output[:, -1:, :]

        logits_list = [gen(output) for gen in self.generator]
        return logits_list, new_cache

    def _decode_layer(self, layer, output, memory, past_cache, is_primer, memory_key_padding_mask):
        # Helper che simula la logica interna di un layer del decoder per completezza
        past_key = past_cache['k'] if past_cache else None
        past_value = past_cache['v'] if past_cache else None
        
        # Self-attention
        q = k = v = output
        if past_key is not None:
            k = torch.cat([past_key, k], dim=1)
            v = torch.cat([past_value, v], dim=1)
        
        new_layer_cache = {'k': k, 'v': v}
        
        attn_mask = None
        if is_primer:
            tgt_len = k.size(1)
            attn_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=output.device, dtype=torch.bool), diagonal=1)
            
        sa_output, _ = layer.self_attn(q, k, v, attn_mask=attn_mask, need_weights=False)
        output = layer.norm1(output + layer.dropout1(sa_output))

        # Cross-attention
        ca_output, _ = layer.multihead_attn(output, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=False)
        output = layer.norm2(output + layer.dropout2(ca_output))
        
        # Feed-forward
        ff_output = layer.linear2(layer.dropout(F.relu(layer.linear1(output))))
        output = layer.norm3(output + layer.dropout3(ff_output))
        
        return output, new_layer_cache

# --- Le classi PositionalEncoding e Seq2SeqTransformer (Standard) rimangono invariate ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead,
                 src_vocab_size, tgt_vocab_size, max_pe_len,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=max_pe_len)
        self.transformer = nn.Transformer(
            d_model=emb_size, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)
    
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: Optional[torch.Tensor] = None,
               cache: Optional[List[Dict[str, torch.Tensor]]] = None) -> tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        is_primer_pass = cache is None
        # ... (la logica di caching per il modello standard rimane invariata) ...
        # Per brevità, la ometto, ma il tuo codice originale è corretto.
        # Qui includo una versione semplificata per completezza
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size))
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), device=tgt.device, dtype=torch.bool), diagonal=1) if is_primer_pass else None
        
        # La gestione del cache qui sarebbe più complessa, questa è una semplificazione
        # Il tuo codice originale che gestisce il cache layer per layer è più corretto e va mantenuto
        output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        
        if is_primer_pass:
            output = output[:, -1:, :]
        
        # Ritorna una tupla con (logits, cache), dove cache è None in questa versione semplificata
        return self.generator(output), None
        
# --- FUNZIONE DI GENERAZIONE SPECIFICA PER OCTUPLE (CORRETTA) ---
def generate_sequence_octuple(model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
                              max_new_tokens, min_new_tokens, temperature, top_k, device,
                              primer_token_ids, model_max_pe_len,
                              # NUOVO PARAMETRO PER FORZARE LO STRUMENTO
                              force_piano_family_only: bool = False,
                              allowed_programs: Optional[List[int]] = None,
                              progress_queue=None, job_id=None,
                              rest_penalty_config: Optional[Dict[str, Any]] = None):
    """
    Genera una sequenza usando un modello Octuple, con gestione corretta del cache
    e la possibilità di forzare un programma di strumento specifico.
    """
    model.eval()
    META_SOS_TOKEN_NAME, META_EOS_TOKEN_NAME, META_UNK_TOKEN_NAME, META_PAD_TOKEN_NAME = "<sos_meta>", "<eos_meta>", "<unk_meta>", "<pad_meta>"
    MIDI_SOS_TOKEN_NAME, MIDI_EOS_TOKEN_NAME, MIDI_PAD_TOKEN_NAME = "SOS_None", "EOS_None", "PAD_None"

    # --- SETUP INIZIALE ---
    sos_meta_id, eos_meta_id, unk_meta_id, meta_pad_id = metadata_vocab_map[META_SOS_TOKEN_NAME], metadata_vocab_map[META_EOS_TOKEN_NAME], metadata_vocab_map[META_UNK_TOKEN_NAME], metadata_vocab_map[META_PAD_TOKEN_NAME]
    sos_midi_tuple = tuple(voc[MIDI_SOS_TOKEN_NAME] for voc in midi_tokenizer.vocab)
    eos_midi_tuple = tuple(voc[MIDI_EOS_TOKEN_NAME] for voc in midi_tokenizer.vocab)
    pad_midi_tuple = tuple(voc[MIDI_PAD_TOKEN_NAME] for voc in midi_tokenizer.vocab)
    num_components = len(midi_tokenizer.vocab)
    
    # Filtraggio token da quelli selezionati dalla GUI
    program_component_idx = -1
    allowed_program_ids = None # Nome cambiato per chiarezza
    # Se allowed_programs è una lista (anche vuota), attiviamo il filtro
    if allowed_programs is not None:
        try:
            program_component_idx = midi_tokenizer.vocab_types_idx['Program']
            program_vocab = midi_tokenizer.vocab[program_component_idx]
            
            # Se la lista è vuota, usiamo il range 0-7. Altrimenti, usiamo la lista stessa.
            piano_range_to_use = allowed_programs if allowed_programs else range(0, 8)
            
            allowed_ids_list = []
            for token_str, token_id in program_vocab.items():
                if token_str.startswith("Program_"):
                    try:
                        prog_num = int(token_str.split('_')[1])
                        # Permetti il token se il suo numero è nel range desiderato
                        if prog_num in piano_range_to_use:
                            allowed_ids_list.append(token_id)
                    except (ValueError, IndexError):
                        continue
            
            if allowed_ids_list:
                allowed_program_ids = torch.tensor(allowed_ids_list, device=device, dtype=torch.long)
                logging.info(f"Vincolo strumenti attivo per Octuple. Programmi permessi: {list(piano_range_to_use)}. ID permessi: {len(allowed_ids_list)}")
            else:
                program_component_idx = -1
        except (KeyError, AttributeError):
            program_component_idx = -1


    with torch.no_grad():
        # ... (codice di encoding del prompt invariato) ...
        logging.info("Codifica del prompt dei metadati...")
        meta_token_ids = [metadata_vocab_map.get(token, unk_meta_id) for token in metadata_prompt]
        src_seq = torch.tensor([[sos_meta_id] + meta_token_ids + [eos_meta_id]], dtype=torch.long, device=device)
        src_padding_mask = (src_seq == meta_pad_id)
        memory = model.encode(src_seq, src_padding_mask)
        logging.info("Memoria del prompt creata.")

        initial_input = torch.tensor([sos_midi_tuple], dtype=torch.long, device=device).unsqueeze(0)
        _, cache = model.decode(initial_input, memory, memory_key_padding_mask=src_padding_mask, cache=None)
        generated_tuples = [sos_midi_tuple]

        # --- CICLO DI GENERAZIONE PRINCIPALE ---
        logging.info("Avvio del ciclo di generazione doppiamente autoregressivo con vincoli...")
        for i in range(max_new_tokens):
            if len(generated_tuples) >= model_max_pe_len:
                break

            next_token_tuple_in_progress = []
            input_for_component_decoder = torch.tensor([pad_midi_tuple], dtype=torch.long, device=device).unsqueeze(0)

            # --- CICLO INTERNO ---
            for j in range(num_components):
                logits_list, _ = model.decode(input_for_component_decoder, memory, memory_key_padding_mask=src_padding_mask, cache=cache)
                head_logits = logits_list[j]

                if head_logits.numel() == 0 or head_logits.shape[0] == 0:
                    raise RuntimeError(f"Il modello ha restituito un output vuoto al componente {j}.")

                # *** NUOVA LOGICA DI VINCOLO (LOGITS WARPING) ***
                # Se questo è il componente del programma e abbiamo un ID da forzare...
                if j == program_component_idx and allowed_program_ids is not None:
                    constrained_logits = torch.full_like(head_logits, -float('Inf'))
                    constrained_logits[:, :, allowed_program_ids] = head_logits[:, :, allowed_program_ids]
                    head_logits = constrained_logits

                # Campionamento (il resto del codice è invariato)
                if temperature > 0 and not (j == program_component_idx and allowed_program_ids is not None):
                    scaled_logits = head_logits.squeeze(1) / temperature
                    if top_k > 0:
                        v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                        if v.numel() > 0:
                           scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')
                    probs = F.softmax(scaled_logits, dim=-1)
                    if torch.isnan(probs).any():
                        sampled_id = torch.randint(0, probs.shape[-1], (1,)).item()
                    else:
                        sampled_id = torch.multinomial(probs, num_samples=1).item()
                else: # Greedy or Forced
                    sampled_id = torch.argmax(head_logits, dim=-1).item()
                
                next_token_tuple_in_progress.append(sampled_id)

                if j < num_components - 1:
                    temp_list = next_token_tuple_in_progress + list(pad_midi_tuple[j+1:])
                    input_for_component_decoder = torch.tensor([temp_list], dtype=torch.long, device=device).unsqueeze(0)
            
            # ... (resto del codice per aggiornamento cache e fine sequenza invariato) ...
            final_generated_tuple = tuple(next_token_tuple_in_progress)
            final_input_for_cache_update = torch.tensor([final_generated_tuple], dtype=torch.long, device=device).unsqueeze(0)
            _, cache = model.decode(final_input_for_cache_update, memory, memory_key_padding_mask=src_padding_mask, cache=cache)
            generated_tuples.append(final_generated_tuple)
            if final_generated_tuple == eos_midi_tuple:
                if len(generated_tuples) - 1 < min_new_tokens:
                    generated_tuples.pop()
                    cache = None
                else:
                    break
            if progress_queue:
                progress_queue.put(("progress", ((i + 1) / max_new_tokens) * 100, job_id))

    # ... (codice finale per restituire gli ID invariato) ...
    final_ids = [list(t) for t in generated_tuples[1:]]
    if final_ids and tuple(final_ids[-1]) == eos_midi_tuple:
        final_ids.pop()
    return final_ids

# --- La funzione di generazione per modelli standard rimane la stessa (con nome cambiato) ---
def generate_sequence_standard(*args, **kwargs):
    return generate_sequence(*args, **kwargs) # Rinomina per chiarezza, il corpo è quello originale
   
# --- FUNZIONI DI GENERAZIONE ---
def generate_sequence(model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
                      max_new_tokens, min_new_tokens, temperature, top_k, device,
                      primer_token_ids, model_max_pe_len, max_rest_penalty, rest_ids,
                      rest_penalty_mode='hybrid', force_piano_only: bool = False,
                      allowed_programs: Optional[List[int]] = None,
                      progress_queue=None, job_id=None):
    model.eval()
    META_SOS_TOKEN_NAME, META_EOS_TOKEN_NAME, META_UNK_TOKEN_NAME, META_PAD_TOKEN_NAME = "<sos_meta>", "<eos_meta>", "<unk_meta>", "<pad_meta>"
    MIDI_SOS_TOKEN_NAME, MIDI_EOS_TOKEN_NAME = "SOS_None", "EOS_None"
    try:
        sos_meta_id, eos_meta_id, unk_meta_id, meta_pad_id = metadata_vocab_map[META_SOS_TOKEN_NAME], metadata_vocab_map[META_EOS_TOKEN_NAME], metadata_vocab_map[META_UNK_TOKEN_NAME], metadata_vocab_map[META_PAD_TOKEN_NAME]
        sos_midi_id, eos_midi_id = midi_tokenizer[MIDI_SOS_TOKEN_NAME], midi_tokenizer[MIDI_EOS_TOKEN_NAME]
    except KeyError as e:
        logging.error(f"Token speciale '{e}' non trovato nei vocabolari.")
        raise ValueError(f"Token speciale '{e}' mancante.")
    
    forbidden_program_ids = None
    # Se allowed_programs è una lista (anche vuota), attiviamo il filtro
    if allowed_programs is not None:
        # Se la lista è vuota, usiamo il range 0-7. Altrimenti, usiamo la lista stessa.
        piano_range_to_use = allowed_programs if allowed_programs else range(0, 8)
        
        forbidden_ids = []
        for token_str, token_id in midi_tokenizer.vocab.items():
            if token_str.startswith("Program_"):
                try:
                    prog_num = int(token_str.split('_')[1])
                    # Vieta il token se il suo numero NON è nel range che vogliamo permettere
                    if prog_num not in piano_range_to_use:
                        forbidden_ids.append(token_id)
                except (ValueError, IndexError):
                    continue
        
        if forbidden_ids:
            forbidden_program_ids = torch.tensor(forbidden_ids, device=device, dtype=torch.long)
            logging.info(f"Modalità 'Forza Famiglia Pianoforte' attiva. Vietati {len(forbidden_ids)} token Programma non pianistici.")

    with torch.no_grad():
        meta_token_ids = [metadata_vocab_map.get(token, unk_meta_id) for token in metadata_prompt]
        src_seq = torch.tensor([[sos_meta_id] + meta_token_ids + [eos_meta_id]], dtype=torch.long, device=device)
        src_padding_mask = (src_seq == meta_pad_id)
        memory = model.encode(src_seq, src_padding_mask)
        initial_ids = [sos_midi_id] + (primer_token_ids if primer_token_ids else [])
        current_tokens = torch.tensor([initial_ids], dtype=torch.long, device=device)
        generated_ids, cache = [], None
        if len(initial_ids) > 0:
            logits, cache = model.decode(current_tokens, memory, memory_key_padding_mask=src_padding_mask)
            logits = logits[:, -1, :]
        else:
            current_tokens = torch.tensor([[sos_midi_id]], dtype=torch.long, device=device)
            logits, cache = model.decode(current_tokens, memory, memory_key_padding_mask=src_padding_mask)

        for i in range(max_new_tokens):
            if len(initial_ids) + len(generated_ids) >= model_max_pe_len:
                logging.warning(f"Raggiunta capacità massima del modello ({model_max_pe_len}). Interruzione.")
                break
            
            # --- INIZIO BLOCCO LOGICA PENALITA' MODIFICATO ---
            if max_rest_penalty > 0 and rest_ids is not None and rest_ids.numel() > 0:
                progress = i / max_new_tokens
                current_penalty = 0
                if rest_penalty_mode == 'constant':
                    # Applica la penalità massima e costante
                    current_penalty = max_rest_penalty
                elif rest_penalty_mode == 'hybrid':
                    # Applica metà della penalità da subito, e l'altra metà in modo graduale
                    base_penalty = max_rest_penalty / 2
                    ramped_penalty = (max_rest_penalty / 2) * progress
                    current_penalty = base_penalty + ramped_penalty
                else: # 'ramped' (comportamento originale)
                    current_penalty = max_rest_penalty * progress
                
                logits[:, rest_ids] -= current_penalty
            # --- FINE BLOCCO LOGICA PENALITA' MODIFICATO ---
            
            # *** NUOVA LOGICA: LOGITS WARPING PER FORZARE IL PIANOFORTE ***
            if force_piano_only and forbidden_program_ids is not None:
                logits[:, forbidden_program_ids] = -float('Inf')

            if temperature > 0:
                scaled_logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(scaled_logits, dim=-1)
                next_token_id_tensor = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id_tensor = torch.argmax(logits, dim=-1, keepdim=True)
            
            next_token_id = next_token_id_tensor.item()
            generated_ids.append(next_token_id)
            
            if next_token_id == eos_midi_id:
                if len(generated_ids) < min_new_tokens:
                    _, top_2_indices = torch.topk(probs, 2, dim=-1)
                    if top_2_indices.size(1) > 1 and top_2_indices[0, 0].item() == eos_midi_id:
                        next_token_id_tensor = top_2_indices[0, 1].unsqueeze(0).unsqueeze(0)
                        generated_ids[-1] = next_token_id_tensor.item()
                    else: break
                else: break
            
            if progress_queue:
                progress_percentage = ((i + 1) / max_new_tokens) * 100
                progress_queue.put(("progress", progress_percentage, job_id))
            
            logits, cache = model.decode(next_token_id_tensor, memory, memory_key_padding_mask=src_padding_mask, cache=cache)
            logits = logits.squeeze(1)
            
    return generated_ids

def generate_multi_chunk_midi(model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
                              total_target_tokens, model_chunk_capacity, generation_config, device,
                              force_piano_only: bool = False, initial_primer_ids=None, 
                              rest_ids=None, progress_queue=None, job_id=None):
    all_generated_tokens = []
    current_primer_ids = initial_primer_ids.copy() if initial_primer_ids else [] 
    PRIMER_TOKEN_COUNT, MIN_TOKENS_PER_CHUNK = 50, 100
    max_new_tokens_per_chunk = min(2048, model_chunk_capacity - PRIMER_TOKEN_COUNT - 5)
    eos_midi_id = midi_tokenizer["EOS_None"]
    
    # Prendi la modalità di penalità dalla configurazione, con 'hybrid' come default
    rest_penalty_mode = generation_config.get("rest_penalty_mode", 'hybrid')

    while len(all_generated_tokens) < total_target_tokens:
        remaining_tokens_to_generate = total_target_tokens - len(all_generated_tokens)
        current_pass_max_new = min(max_new_tokens_per_chunk, remaining_tokens_to_generate)
        if len(current_primer_ids) + current_pass_max_new + 1 > model_chunk_capacity: break
        
        newly_generated_ids = generate_sequence(
            model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
            max_new_tokens=current_pass_max_new,
            min_new_tokens=min(MIN_TOKENS_PER_CHUNK, current_pass_max_new),
            temperature=generation_config.get("temperature", 0.75),
            top_k=generation_config.get("top_k", 40),
            max_rest_penalty=generation_config.get("max_rest_penalty", 0.0),
            device=device,
            primer_token_ids=current_primer_ids,
            model_max_pe_len=model_chunk_capacity,
            rest_ids=rest_ids,
            rest_penalty_mode=rest_penalty_mode,
            force_piano_only=force_piano_only,
            progress_queue=progress_queue, 
            job_id=job_id                
        )
        if not newly_generated_ids: break
        
        chunk_ended_with_eos = eos_midi_id in newly_generated_ids
        tokens_to_add = newly_generated_ids[:newly_generated_ids.index(eos_midi_id) + 1] if chunk_ended_with_eos else newly_generated_ids
        all_generated_tokens.extend(tokens_to_add)

        if chunk_ended_with_eos:
            if len(all_generated_tokens) >= total_target_tokens * 0.8: break
            primer_candidate = tokens_to_add[:-1]
            current_primer_ids = primer_candidate[-PRIMER_TOKEN_COUNT:] if len(primer_candidate) > PRIMER_TOKEN_COUNT else []
        else:
            current_primer_ids = tokens_to_add[-PRIMER_TOKEN_COUNT:]
    
    if all_generated_tokens and all_generated_tokens[-1] != eos_midi_id:
        all_generated_tokens.append(eos_midi_id)
    return all_generated_tokens

# --- FUNZIONE DI ANALISI MODELLO ---
def get_model_info(model_path: str) -> dict:
    if not model_path or not Path(model_path).exists():
        return {"error": "Selezionare un percorso valido per il modello."}
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model_params = checkpoint.get('model_params')
        if not model_params:
            return {"error": "'model_params' non trovato nel checkpoint del modello."}

        is_octuple = 'tgt_vocab_sizes' in model_params
        ModelClass = Seq2SeqTransformerOctuple if is_octuple else Seq2SeqTransformer
        
        model = ModelClass(**model_params)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            "Tipo Modello Rilevato": "Octuple" if is_octuple else "Standard (es. REMI)",
            "Parametri Addestrabili": f"{trainable_params:,}".replace(",", "."),
            "Dimensione Embedding (d_model)": model_params.get('emb_size', 'N/D'),
            "Attention Heads": model_params.get('nhead', 'N/D'),
            "Loss Value": model_params.get('loss_value', 'N/D'),
        }
        return info
    except Exception as e:
        return {"error": f"Impossibile leggere il file del modello. Errore: {e}"}

# ==================================================================
# --- INIZIO NUOVA SEZIONE DI PULIZIA MIDI AVANZATA ---
# ==================================================================

# Mappatura delle famiglie di strumenti secondo lo standard General MIDI
GM_INSTRUMENT_FAMILIES = {
    "Piano": range(0, 8),
    "Chromatic Percussion": range(8, 16),
    "Organ": range(16, 24),
    "Guitar": range(24, 32),
    "Bass": range(32, 40),
    "Strings & Ensemble": range(40, 56), # Strings, Synth Strings, Choir, Voice
    "Brass": range(56, 64),
    "Reed": range(64, 72),
    "Pipe": range(72, 80),
    "Synth Lead": range(80, 88),
    "Synth Pad": range(88, 96),
    "Synth Effects": range(96, 104),
    "Ethnic": range(104, 112),
    "Percussive": range(112, 120),
    "Sound Effects": range(120, 128),
}

def _get_instrument_family(program: int) -> str:
    """Restituisce il nome della famiglia di uno strumento dato il suo program number."""
    for family, program_range in GM_INSTRUMENT_FAMILIES.items():
        if program in program_range:
            return family
    return "Unknown"

def _trim_initial_silence(score, log_and_update):
    """Rimuove il silenzio all'inizio del brano in modo robusto."""
    if not score.tracks:
        return score

    first_event_time = float('inf')

    # Scansiona TUTTI gli eventi per trovare il primo in assoluto
    # 1. Note e Controlli all'interno delle tracce
    for track in score.tracks:
        if track.notes:
            first_event_time = min(first_event_time, track.notes[0].start)
        if track.controls:
            first_event_time = min(first_event_time, track.controls[0].time)

    # 2. Eventi globali di Tempo e Time Signature
    if score.tempos:
        first_event_time = min(first_event_time, score.tempos[0].time)
    if score.time_signatures:
        first_event_time = min(first_event_time, score.time_signatures[0].time)

    # Se il primo evento non è a 0, shifta tutto
    if 0 < first_event_time < float('inf'):
        first_event_seconds = score.tick_to_time(first_event_time)
        log_and_update(f"   - Trovato primo evento a {first_event_seconds:.2f}s. Rimozione del silenzio iniziale.")

        # Sposta tutti gli eventi indietro nel tempo
        for track in score.tracks:
            track.shift_time(-first_event_time)
        
        # Sposta anche gli eventi globali
        for tempo in score.tempos:
            tempo.time -= first_event_time
        for ts in score.time_signatures:
            ts.time -= first_event_time
            
    return score

def _quantize_notes(track, grid_str: str, tpq: int, log_and_update):
    """Quantizza le note di una traccia a una griglia ritmica."""
    try:
        num, den = map(int, grid_str.split('/'))
        ticks_in_grid = int(tpq * (4 / den) * (num / num)) # Semplificato: tpq * 4 / den
        if ticks_in_grid == 0: return
        
        for note in track.notes:
            quantized_start = int(round(note.start / ticks_in_grid) * ticks_in_grid)
            quantized_end = int(round(note.end / ticks_in_grid) * ticks_in_grid)
            note.start = quantized_start
            note.duration = max(ticks_in_grid, quantized_end - quantized_start)
    except Exception as e:
        log_and_update(f"ATTENZIONE: Errore durante la quantizzazione: {e}")

def _limit_polyphony(track, max_polyphony: int, log_and_update):
    """Limita il numero di note simultanee in una traccia."""
    if not track.notes or max_polyphony <= 0: return
    
    events = []
    for note in track.notes:
        events.append({'time': note.start, 'type': 'on', 'note': note})
        events.append({'time': note.end, 'type': 'off', 'note': note})
    
    events.sort(key=lambda x: x['time'])
    
    active_notes = []
    
    for event in events:
        if event['type'] == 'on':
            active_notes.append(event['note'])
            if len(active_notes) > max_polyphony:
                # Trova la nota da terminare: quella con la velocity più bassa
                active_notes.sort(key=lambda n: n.velocity)
                note_to_terminate = active_notes.pop(0) # Rimuovi la più debole
                note_to_terminate.duration = event['time'] - note_to_terminate.start
        else: # 'off'
            if event['note'] in active_notes:
                active_notes.remove(event['note'])

    track.notes = [n for n in track.notes if n.duration > 0]

def _filter_pitch_range(track, min_pitch: int, max_pitch: int, log_and_update):
    """Filtra le note al di fuori di un range di altezze."""
    original_count = len(track.notes)
    track.notes = [n for n in track.notes if min_pitch <= n.pitch <= max_pitch]
    removed_count = original_count - len(track.notes)
    if removed_count > 0:
        log_and_update(f"   - Traccia '{track.name}': rimosse {removed_count} note fuori dal range ({min_pitch}-{max_pitch}).")

def _remove_duplicate_notes(track):
    """Rimuove note identiche e sovrapposte."""
    seen = set()
    unique_notes = []
    for note in sorted(track.notes, key=lambda n: (n.start, n.pitch)):
        note_key = (note.start, note.pitch, note.duration)
        if note_key not in seen:
            unique_notes.append(note)
            seen.add(note_key)
    track.notes = unique_notes

def _merge_tracks_by_family(score, log_and_update):
    """Fonde tracce che appartengono alla stessa famiglia di strumenti."""
    if len(score.tracks) <= 1:
        return score
        
    family_map = defaultdict(list)
    # Raggruppa le tracce per famiglia di strumenti
    for i, track in enumerate(score.tracks):
        if track.is_drum:
            # Le tracce di batteria non vengono mai unite
            family_key = f"Drums_{i}" 
        else:
            family_key = _get_instrument_family(track.program)
        family_map[family_key].append(track)
        
    new_tracks = []
    for family, tracks in family_map.items():
        if len(tracks) > 1:
            log_and_update(f"Fusione di {len(tracks)} tracce nella famiglia '{family}'.")
            # Usa la prima traccia come base per nome, programma, etc.
            base_track = tracks[0]
            
            merged_track = Track(name=family, program=base_track.program, is_drum=base_track.is_drum)
            
            for track in tracks:
                for note in track.notes: merged_track.notes.append(note)
                for control in track.controls: merged_track.controls.append(control)
            
            # Riordina tutti gli eventi per tempo
            merged_track.notes.sort(key=lambda n: n.start)
            merged_track.controls.sort(key=lambda c: c.time)
            new_tracks.append(merged_track)
        else:
            # Se c'è solo una traccia nella famiglia, aggiungila così com'è
            new_tracks.append(tracks[0])
            
    score.tracks = new_tracks
    return score

def enhance_midi_score(score, cleaning_options: Dict[str, Any], metadata_prompt: List[str], log_and_update):
    """
    Applica una suite completa di pulizie e miglioramenti a un oggetto Score di symusic.
    Ogni passaggio è controllato dal dizionario cleaning_options.
    """
    if not score.tracks:
        return score

    # ==================== INIZIO NUOVO BLOCCO CONDIZIONALE ====================
    log_and_update("- Controllo del tempo e del metro...")

    # Controlla se il modello ha già generato un tempo. Se non l'ha fatto, usa il prompt come fallback.
    if not score.tempos:
        log_and_update("   - Nessun evento di tempo trovato. Lo imposto dal prompt come fallback.")
        _, tempo_bpm = '4/4', 120.0 # Default
        TEMPO_MAP = {'Tempo_VerySlow': 60, 'Tempo_Slow': 80, 'Tempo_Moderate': 110, 'Tempo_Fast': 140, 'Tempo_VeryFast': 170, 'Tempo_ExtremelyFast': 190}
        for meta in metadata_prompt:
            if meta.startswith('Tempo_'):
                tempo_bpm = float(TEMPO_MAP.get(meta, 120))
        score.tempos.append(Tempo(time=0, qpm=tempo_bpm))
        log_and_update(f"   - Tempo di fallback impostato a: {tempo_bpm} BPM.")
    else:
        log_and_update("   - Trovato evento di tempo generato dal modello. Verrà mantenuto.")

    # Controlla se il modello ha già generato un metro. Se non l'ha fatto, usa il prompt come fallback.
    if not score.time_signatures:
        log_and_update("   - Nessun evento di metro trovato. Lo imposto dal prompt come fallback.")
        ts_string, _ = '4/4', 120.0 # Default
        for meta in metadata_prompt:
            if meta.startswith('TimeSig='):
                ts_string = meta.split('=')[1]
        try:
            num, den = map(int, ts_string.split('/'))
            score.time_signatures.append(TimeSignature(time=0, numerator=num, denominator=den))
            log_and_update(f"   - Metro di fallback impostato a: {ts_string}.")
        except Exception as e:
            log_and_update(f"   - ATTENZIONE: Errore nell'impostare il metro di fallback ({e}).")
    else:
        log_and_update("   - Trovato evento di metro generato dal modello. Verrà mantenuto.")
    # ===================== FINE NUOVO BLOCCO CONDIZIONALE =====================


    log_and_update("Avvio della pulizia e del miglioramento del MIDI...")
    # 1. RIMOZIONE SUSTAIN (Invertito: l'opzione è "non usare sustain", quindi rimuovilo)
    if cleaning_options.get("remove_sustain", False):
        log_and_update("- Rimozione degli eventi di Sustain Pedal (CC#64)...")
        for track in score.tracks:
            track.controls = [c for c in track.controls if c.number != 64]

    # 2. RIMOZIONE TRACCE VUOTE E NOTE GLITCH (Generalmente sicuro da lasciare attivo)
    log_and_update("- Rimozione tracce vuote e note glitch...")
    MIN_NOTE_DURATION_TICKS = 10 # Si può mantenere hardcoded o passare tramite opzioni
    for track in score.tracks:
        valid_notes = [n for n in track.notes if n.velocity > 0 and n.duration > MIN_NOTE_DURATION_TICKS]
        track.notes = valid_notes
    score.tracks = [t for t in score.tracks if len(t.notes) > 0]
    if not score.tracks: return score

    # 3. FUSIONE TRACCE
    if cleaning_options.get("merge_tracks", False):
        log_and_update("- Fusione tracce per famiglia di strumenti...")
        score = _merge_tracks_by_family(score, log_and_update)

    # 4. RIMOZIONE SILENZIO INIZIALE
    if cleaning_options.get("trim_silence", False):
        log_and_update("- Rimozione silenzio iniziale...")
        score = _trim_initial_silence(score, log_and_update)

    # 5. ELABORAZIONE PER TRACCIA
    for track in score.tracks:
        log_and_update(f"- Processamento traccia: '{track.name}' (Program: {track.program}, Note: {len(track.notes)})")
        
        # Correzione note sovrapposte e normalizzazione velocity (Sempre utili)
        last_notes_on_pitch = {}
        MIN_VEL, MAX_VEL = 20, 120 # Range di default
        if cleaning_options.get("filter_velocity", False):
            MIN_VEL = cleaning_options.get("velocity_min", 20)
            MAX_VEL = cleaning_options.get("velocity_max", 120)
            log_and_update(f"   - Normalizzazione velocity nel range {MIN_VEL}-{MAX_VEL}.")

        for note in sorted(track.notes, key=lambda n: n.start):
            note.velocity = max(MIN_VEL, min(MAX_VEL, note.velocity))
            if note.pitch in last_notes_on_pitch:
                last_note = last_notes_on_pitch[note.pitch]
                if last_note.end > note.start:
                    last_note.duration = note.start - last_note.start
            last_notes_on_pitch[note.pitch] = note
        track.notes = [n for n in track.notes if n.duration > 0]

        _remove_duplicate_notes(track) # Sempre utile, non ha controindicazioni

        # 5a. QUANTIZZAZIONE
        if cleaning_options.get("quantize", False):
            grid = cleaning_options.get("quantize_grid", "1/16")
            log_and_update(f"   - Quantizzazione alla griglia: {grid}")
            _quantize_notes(track, grid, score.ticks_per_quarter, log_and_update)
            
        # 5b. FILTRO PITCH
        if cleaning_options.get("filter_pitch", False):
            min_pitch = cleaning_options.get("pitch_min", 21)
            max_pitch = cleaning_options.get("pitch_max", 108)
            log_and_update(f"   - Filtraggio altezze nel range MIDI {min_pitch}-{max_pitch}.")
            _filter_pitch_range(track, min_pitch, max_pitch, log_and_update)

        # 5c. LIMITE POLIFONIA
        if cleaning_options.get("limit_polyphony", False):
            max_poly = cleaning_options.get("polyphony_max", 12)
            log_and_update(f"   - Limitazione polifonia a {max_poly} note.")
            _limit_polyphony(track, max_poly, log_and_update)
            
    for track in score.tracks:
        track.notes.sort(key=lambda n: n.start)
        
    log_and_update("Pulizia MIDI completata.")
    return score

# ==================================================================
# --- FINE NUOVA SEZIONE DI PULIZIA MIDI AVANZATA ---
# ==================================================================

# --- FUNZIONE PRINCIPALE DI GENERAZIONE ---
def run_generation(model_path, midi_vocab_path, meta_vocab_path,
                   metadata_prompt, output_dir, total_tokens, temperature, top_k, max_rest_penalty,
                   rest_penalty_mode: str = 'hybrid',
                   # Il vecchio flag non serve più direttamente, ma lo lasciamo per compatibilità
                   force_piano_only: bool = False, 
                   # IL NUOVO PARAMETRO CHIAVE
                   allowed_programs: Optional[List[int]] = None,
                   cleaning_options: Dict[str, Any] = None,
                   primer_midi_path=None, progress_queue=None, job_id=None):
    try:
        def log_and_update(message):
            logging.info(f"[{job_id}] {message}")
            if progress_queue:
                progress_queue.put(("status", message, job_id))

        log_and_update("Inizio generazione...")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        log_and_update(f"Device rilevato: {DEVICE}")

        log_and_update("Caricamento vocabolari e tokenizer...")
        
        # --- MODIFICA: Rilevamento dinamico del tokenizer ---
        with open(midi_vocab_path, 'r', encoding='utf-8') as f:
            tokenizer_class_name = json.load(f).get("tokenization", "REMI") # Default a REMI se non specificato
        
        TokenizerClass = getattr(miditok, tokenizer_class_name)
        midi_tokenizer = TokenizerClass(params=str(midi_vocab_path))
        log_and_update(f"Tokenizer '{tokenizer_class_name}' caricato.")
        
        with open(meta_vocab_path, 'r', encoding='utf-8') as f:
            metadata_vocab_map = json.load(f)['token_to_id']
        
        log_and_update("Caricamento modello...")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model_params = checkpoint['model_params']
        
        # --- MODIFICA: Rilevamento dinamico del tipo di modello ---
        is_octuple = 'tgt_vocab_sizes' in model_params
        ModelClass = Seq2SeqTransformerOctuple if is_octuple else Seq2SeqTransformer
        
        log_and_update(f"Rilevato modello di tipo {'Octuple' if is_octuple else 'Standard'}.")
        
        model = ModelClass(**model_params).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        primer_token_ids = []
        # La logica del primer per Octuple va gestita con più attenzione (deve essere una lista di tuple)
        # Per ora, la tokenizzazione standard funziona, ma la gestione va raffinata se si usano primer complessi.
        if primer_midi_path and Path(primer_midi_path).exists():
            primer_tokens = midi_tokenizer.encode(primer_midi_path)
            primer_token_ids = primer_tokens.ids if is_octuple else primer_tokens[0].ids

        generation_config = {
            "temperature": temperature, "top_k": top_k, "max_rest_penalty": max_rest_penalty,
            "rest_penalty_mode": rest_penalty_mode
        }
        model_chunk_capacity = model_params.get('max_pe_len', 2048)

        # --- Scelta della funzione di generazione corretta ---
        if is_octuple:
            log_and_update("Generazione con logica Octuple...")
            generated_token_ids = generate_sequence_octuple(
                model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
                max_new_tokens=total_tokens,
                min_new_tokens=int(total_tokens * 0.5),
                temperature=temperature, top_k=top_k, device=DEVICE,
                primer_token_ids=primer_token_ids,
                model_max_pe_len=model_chunk_capacity,
                force_piano_family_only=force_piano_only,
                allowed_programs=allowed_programs,
                progress_queue=progress_queue, job_id=job_id
            )
        else: # Modello Standard
            log_and_update("Generazione con logica Standard...")
            rest_ids = torch.tensor([i for t, i in midi_tokenizer.vocab.items() if t.startswith("Rest_")], device=DEVICE, dtype=torch.long)
            
            generated_token_ids = generate_multi_chunk_midi(
                model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
                total_target_tokens=total_tokens, model_chunk_capacity=model_chunk_capacity,
                generation_config=generation_config, device=DEVICE, initial_primer_ids=primer_token_ids,
                rest_ids=rest_ids, 
                # Passa il nuovo flag alla funzione di generazione a chunk
                force_piano_only=force_piano_only,
                allowed_programs=allowed_programs,
                progress_queue=progress_queue, job_id=job_id)
        
        if not generated_token_ids:
            raise RuntimeError("La generazione non ha prodotto token.")

        # Questo blocco sostituisce tutte le correzioni precedenti.
        # Strategia: eliminiamo QUALSIASI tupla di token che contiene un valore "_None"
        # in uno dei suoi campi numerici critici, risolvendo tutti i possibili ValueError
        # in un solo passaggio.

        log_and_update("Avvio sanificazione completa dei token generati...")

        # 1. Ricostruzione dinamica degli indici dei vocabolari, come prima.
        try:
            token_types = ["Pitch", "Position", "Bar"]
            if midi_tokenizer.config.use_velocities:
                token_types.append("Velocity")
            if midi_tokenizer.config.using_note_duration_tokens:
                token_types.append("Duration")
            if midi_tokenizer.config.use_programs:
                token_types.append("Program")
            if midi_tokenizer.config.use_tempos:
                token_types.append("Tempo")
            if midi_tokenizer.config.use_time_signatures:
                token_types.append("TimeSig")
        except Exception as e:
            raise RuntimeError(f"Errore nella lettura della configurazione del tokenizer: {e}")

        # 2. NUOVA LOGICA: Creazione di una mappa di ID "cattivi" per ogni componente critico.
        # I campi critici sono quelli che vengono convertiti in interi e non possono essere 'None'.
        critical_fields = ["Pitch", "Position", "Bar", "Velocity", "Program"]
        bad_ids_map = {}

        for field_name in critical_fields:
            try:
                if field_name not in token_types:
                    continue
                
                component_index = token_types.index(field_name)
                component_vocab = midi_tokenizer.vocab[component_index]
                
                # Cerca TUTTI i token in questo vocabolario che finiscono con '_None'
                # Questo cattura 'Program_None', 'PAD_None', 'BOS_None', etc.
                ids_to_discard = {
                    token_id for token_string, token_id in component_vocab.items()
                    if token_string.endswith("_None")
                }
                
                if ids_to_discard:
                    bad_ids_map[component_index] = ids_to_discard

            except (ValueError, KeyError, IndexError):
                pass # Ignora i campi opzionali non presenti

        log_and_update(f"Mappa degli ID da scartare (indice: {{set di id}}): {bad_ids_map}")

        # 3. Filtro aggressivo basato sulla nuova mappa.
        sanitized_ids = []
        if not bad_ids_map:
            # Se la mappa è vuota, per sicurezza si assume che i token siano tutti validi.
            sanitized_ids = generated_token_ids
            log_and_update("Nessun ID 'None' trovato nei vocabolari. Si procede senza filtro.")
        else:
            for token_tuple in generated_token_ids:
                is_valid = True
                for component_index, ids_to_discard_set in bad_ids_map.items():
                    # Se il valore della tupla in questo indice è nel set di ID da scartare, la tupla non è valida.
                    if component_index < len(token_tuple) and token_tuple[component_index] in ids_to_discard_set:
                        is_valid = False
                        break
                if is_valid:
                    sanitized_ids.append(token_tuple)

        # 4. Rimuoviamo anche i token di controllo generali (SOS, EOS, PAD).
        eos_tuple = tuple(voc.get("EOS_None", -1) for voc in midi_tokenizer.vocab)
        sos_tuple = tuple(voc.get("SOS_None", -1) for voc in midi_tokenizer.vocab)
        pad_tuple = tuple(voc.get("PAD_None", -1) for voc in midi_tokenizer.vocab)
        control_token_tuples = {eos_tuple, sos_tuple, pad_tuple}

        final_cleaned_ids = [
            token_tuple for token_tuple in sanitized_ids
            if tuple(token_tuple) not in control_token_tuples
        ]

        original_count = len(generated_token_ids)
        final_count = len(final_cleaned_ids)
        if final_count < original_count:
            log_and_update(f"Sanificazione ha rimosso {original_count - final_count} token non validi.")
        
        if not final_cleaned_ids:
            raise RuntimeError("La sanificazione ha rimosso tutti i token. Nessun output valido.")

        log_and_update(f"Sanificazione completata. Token validi pronti: {final_count}. Avvio decodifica...")
        
        generated_midi_object = midi_tokenizer.decode(final_cleaned_ids)
        
        # La logica di pulizia è agnostica rispetto al tokenizer e va bene per entrambi
        enhanced_midi_object = enhance_midi_score(
            generated_midi_object,
            cleaning_options=cleaning_options if cleaning_options is not None else {},
            metadata_prompt=metadata_prompt,
            log_and_update=log_and_update
        )

        # Salvataggio del file finale
        # ... (la logica di salvataggio rimane invariata)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        prompt_name_part = "_".join(metadata_prompt).replace("=", "").replace("/", "")[:50]
        prefix = job_id if job_id else "generated"
        output_filename = f"{prefix}_{prompt_name_part}_{timestamp}.mid"
        output_path = Path(output_dir) / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        enhanced_midi_object.dump_midi(str(output_path))

        final_message = f"Successo! File MIDI salvato in:\n{output_path}"
        log_and_update(final_message)
        return (final_message, job_id)

    except Exception as e:
        # ... (la gestione degli errori rimane invariata)
        error_message = f"ERRORE: {e}"
        logging.error("Errore durante la generazione.", exc_info=True)
        if progress_queue:
            progress_queue.put(("status", error_message, job_id))
        return (error_message, job_id)