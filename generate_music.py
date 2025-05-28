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
from symusic import Score # Importa Score direttamente da symusic

# --- Configurazione / Costanti Essenziali per la Generazione ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MODIFICA QUESTI PERCORSI ===
# Percorso del checkpoint del modello addestrato (.pt)
PATH_MODELLO_CHECKPOINT = Path(r"C:\Users\Michael\Downloads\transformer_mutopia_epoch3_valloss1.9265_20250528-113813.pt")
# Percorso del file di vocabolario MIDI (.json) usato per addestrare il modello sopra
PATH_VOCAB_MIDI = Path(r"C:\Users\Michael\Desktop\SheetsMusicGenerator\ModelloPiccolo\midi_vocab.json")
# Percorso del file di vocabolario dei metadati (.json) usato per addestrare il modello sopra
PATH_VOCAB_METADATA = Path(r"C:\Users\Michael\Desktop\SheetsMusicGenerator\ModelloPiccolo\metadata_vocab.json")
# Directory dove salvare i file MIDI generati
GENERATION_OUTPUT_DIR = Path("./generated_midi_inference")
# ================================

# Strategia di Tokenizzazione (DEVE CORRISPONDERE A QUELLA USATA PER L'ADDDESTRAMENTO DEL MODELLO CARICATO)
# Ad esempio, se il modello è stato addestrato con REMI:
MIDI_TOKENIZER_STRATEGY = miditok.REMI
# Se era CPWord:
# MIDI_TOKENIZER_STRATEGY = miditok.CPWord # Attenzione: CPWord ha mostrato problemi di multi-vocabolario

# Token Speciali MIDI (devono corrispondere a quelli usati per l'addestramento)
MIDI_PAD_TOKEN_NAME = "PAD_None"
MIDI_SOS_TOKEN_NAME = "SOS_None"
MIDI_EOS_TOKEN_NAME = "EOS_None"
MIDI_UNK_TOKEN_NAME = "UNK_None"

# Token Speciali per Metadati (devono corrispondere)
META_PAD_TOKEN_NAME = "<pad_meta>"
META_UNK_TOKEN_NAME = "<unk_meta>"
META_SOS_TOKEN_NAME = "<sos_meta>"
META_EOS_TOKEN_NAME = "<eos_meta>"

# Lunghezze massime sequenza (idealmente prese dal checkpoint o corrispondenti al training)
# Queste sono usate in generate_sequence e PositionalEncoding.
# Se il checkpoint non le salva esplicitamente per PositionalEncoding, assicurati che siano abbastanza grandi.
MAX_SEQ_LEN_MIDI = 1024
MAX_SEQ_LEN_META = 128

PRIMER_TOKEN_COUNT = 50  # Quanti token usare dalla fine del chunk precedente come primer
                          # Deve essere minore di effective_model_max_len
MIN_TOKENS_PER_CHUNK = 100 # Lunghezza minima per un chunk prima di accettare EOS
                           # o per continuare la generazione

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Crea directory di output se non esiste
GENERATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Definizioni Helper, Classi Modello e Funzioni (copiate da training.py) ---

def tokenize_metadata(metadata_dict):
    tokens = []
    if 'style' in metadata_dict and metadata_dict['style']:
        tokens.append(f"Style={metadata_dict['style'].replace(' ', '_')}")
    if 'key' in metadata_dict and metadata_dict['key']:
        tokens.append(f"Key={metadata_dict['key'].replace(' ', '_')}")
    if 'time_signature' in metadata_dict and metadata_dict['time_signature']:
        tokens.append(f"TimeSig={metadata_dict['time_signature']}")
    if 'title' in metadata_dict and metadata_dict['title']:
        clean_title = metadata_dict['title'].replace(' ', '_').lower()
        clean_title = ''.join(c for c in clean_title if c.isalnum() or c == '_')
        tokens.append(f"Title={clean_title[:30]}")
    return tokens

def load_midi_tokenizer_for_inference(vocab_path):
    logging.info(f"Caricamento tokenizer MIDI da {vocab_path}")
    if not Path(vocab_path).exists():
        logging.error(f"File vocabolario MIDI non trovato: {vocab_path}")
        sys.exit(1)
    try:
        # La configurazione del tokenizer (use_programs, ecc.) è definita nel file params.
        # Non è necessario passare una TokenizerConfig qui se il file params è completo.
        tokenizer = MIDI_TOKENIZER_STRATEGY(params=str(vocab_path))
        logging.info(f"Tokenizer MIDI caricato con successo. Strategia: {MIDI_TOKENIZER_STRATEGY.__name__}")

        # Verifica ID token speciali essenziali
        for token_name in [MIDI_PAD_TOKEN_NAME, MIDI_SOS_TOKEN_NAME, MIDI_EOS_TOKEN_NAME, MIDI_UNK_TOKEN_NAME]:
            try:
                _ = tokenizer[token_name] # Tenta di accedere
            except KeyError:
                logging.error(f"ERRORE CRITICO: Token speciale MIDI '{token_name}' non trovato nel vocabolario caricato da {vocab_path}.")
                sys.exit(1)
        return tokenizer
    except Exception as e:
        logging.error(f"Errore nel caricare il tokenizer MIDI da {vocab_path}: {e}", exc_info=True)
        sys.exit(1)

def load_metadata_vocab_for_inference(vocab_path):
    logging.info(f"Caricamento vocabolario Metadati da {vocab_path}")
    if not Path(vocab_path).exists():
        logging.error(f"File vocabolario Metadati non trovato: {vocab_path}")
        sys.exit(1)
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        token_to_id = vocab_data['token_to_id']
        
        # Verifica token speciali essenziali per metadati
        for token_name in [META_PAD_TOKEN_NAME, META_UNK_TOKEN_NAME, META_SOS_TOKEN_NAME, META_EOS_TOKEN_NAME]:
            if token_name not in token_to_id:
                logging.error(f"ERRORE CRITICO: Token speciale Metadati '{token_name}' non trovato nel vocabolario caricato da {vocab_path}.")
                sys.exit(1)
        return token_to_id
    except Exception as e:
        logging.error(f"Errore nel caricare il vocabolario Metadati da {vocab_path}: {e}", exc_info=True)
        sys.exit(1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000): # max_len dovrebbe essere abbastanza grande
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
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

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, memory_key_padding_mask=None, tgt_mask=None):
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size))

        if tgt_mask is None:
             tgt_len = tgt.size(1)
             tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool), diagonal=1)

        if memory_key_padding_mask is None:
             memory_key_padding_mask = src_padding_mask

        outs = self.transformer(src_emb, tgt_emb,
                                tgt_mask=tgt_mask,
                                src_key_padding_mask=src_padding_mask,
                                tgt_key_padding_mask=tgt_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)

    def decode(self, tgt, memory, tgt_mask, tgt_padding_mask=None, memory_key_padding_mask=None):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size))
        return self.transformer.decoder(tgt_emb, memory,
                                         tgt_mask=tgt_mask,
                                         tgt_key_padding_mask=tgt_padding_mask,
                                         memory_key_padding_mask=memory_key_padding_mask)

def generate_sequence(model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
                      max_new_tokens, # Rinominato da max_len a max_new_tokens
                      min_new_tokens, # Rinominato da min_len a min_new_tokens
                      temperature=0.5, top_k=None, device=DEVICE,
                      primer_token_ids=None, # Nuovo parametro per il primer
                      model_max_pe_len=5000): # Capacità massima del Positional Encoding del modello
    model.eval()
    try:
        sos_meta_id = metadata_vocab_map[META_SOS_TOKEN_NAME]
        eos_meta_id = metadata_vocab_map[META_EOS_TOKEN_NAME]
        unk_meta_id = metadata_vocab_map[META_UNK_TOKEN_NAME]
        meta_pad_id = metadata_vocab_map[META_PAD_TOKEN_NAME]

        sos_midi_id = midi_tokenizer[MIDI_SOS_TOKEN_NAME]
        eos_midi_id = midi_tokenizer[MIDI_EOS_TOKEN_NAME]
    except KeyError as e:
        logging.error(f"Errore critico: Token speciale '{e}' non trovato nei vocabolari.")
        sys.exit(1)

    meta_token_ids = [metadata_vocab_map.get(token, unk_meta_id) for token in metadata_prompt]
    # Assicurati che la lunghezza dei metadati non superi la capacità del modello per i metadati.
    # MAX_SEQ_LEN_META dovrebbe essere preso da model_params se disponibile, o usato come nel training.
    src_seq = torch.tensor([[sos_meta_id] + meta_token_ids[:MAX_SEQ_LEN_META-2] + [eos_meta_id]], dtype=torch.long, device=device)
    src_padding_mask = (src_seq == meta_pad_id)

    with torch.no_grad():
        memory = model.encode(src_seq, src_padding_mask)
        memory_key_padding_mask = src_padding_mask
        
        if primer_token_ids and len(primer_token_ids) > 0:
            # Inizia con SOS + primer. La lunghezza del primer deve essere considerata
            # rispetto a model_max_pe_len.
            initial_ids = [sos_midi_id] + primer_token_ids
        else:
            initial_ids = [sos_midi_id]
        
        tgt_tokens = torch.tensor([initial_ids], dtype=torch.long, device=device)
        
        generated_ids_this_chunk = []
        generated_eos_in_chunk = False

        # Il loop genera fino a max_new_tokens *nuovi* token
        # La lunghezza totale di tgt_tokens non deve superare model_max_pe_len
        for i in range(max_new_tokens):
            current_total_len = tgt_tokens.size(1)
            if current_total_len >= model_max_pe_len:
                logging.warning(f"Raggiunta capacità massima del modello ({model_max_pe_len}) per il chunk corrente. Interruzione anticipata del chunk.")
                break

            tgt_mask_step = torch.triu(torch.ones(current_total_len, current_total_len, device=device, dtype=torch.bool), diagonal=1)
            tgt_padding_mask_step = torch.zeros_like(tgt_tokens, dtype=torch.bool, device=device) # No padding in generation step

            decoder_output = model.decode(tgt=tgt_tokens, memory=memory, tgt_mask=tgt_mask_step,
                                          tgt_padding_mask=tgt_padding_mask_step,
                                          memory_key_padding_mask=memory_key_padding_mask)
            logits = model.generator(decoder_output)
            last_logits = logits[:, -1, :]

            if temperature > 0:
                last_logits = last_logits / temperature

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)), dim=-1)
                last_logits[last_logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(last_logits, dim=-1)
            next_token_id_tensor = torch.multinomial(probs, num_samples=1)
            next_token_id = next_token_id_tensor.item()

            # Gestione EOS e min_new_tokens
            if next_token_id == eos_midi_id and len(generated_ids_this_chunk) < min_new_tokens:
                if probs.size(-1) > 1:
                    top_2_probs, top_2_indices = torch.topk(probs, 2, dim=-1)
                    if top_2_indices[0, 0].item() == eos_midi_id:
                        if top_2_indices.size(1) > 1:
                            next_token_id_tensor = top_2_indices[0, 1].unsqueeze(0).unsqueeze(0)
                            next_token_id = next_token_id_tensor.item()
                        else: # Solo EOS possibile, raro
                            logging.warning("Forzato a generare EOS nonostante min_new_tokens non raggiunta (solo EOS possibile).")
                            generated_eos_in_chunk = True
                            # Non aggiungere a generated_ids_this_chunk qui, lo facciamo dopo il check
                    # else: il più probabile non era EOS, next_token_id è già corretto
                # else: solo una scelta possibile (EOS)
            
            generated_ids_this_chunk.append(next_token_id)
            tgt_tokens = torch.cat((tgt_tokens, next_token_id_tensor), dim=1)

            if next_token_id == eos_midi_id and len(generated_ids_this_chunk) >= min_new_tokens:
                logging.info(f"Token EOS generato nel chunk dopo {len(generated_ids_this_chunk)} nuovi token (min_new_tokens raggiunta).")
                generated_eos_in_chunk = True
                break 
            
            # Se abbiamo generato EOS ma min_new_tokens non era raggiunta, generated_eos_in_chunk potrebbe essere True
            # ma il break non è avvenuto. Resettiamo generated_eos_in_chunk se il token attuale *non* è EOS.
            if next_token_id != eos_midi_id:
                generated_eos_in_chunk = False # Se abbiamo forzato a non usare EOS prima, e ora non è EOS.

    # Rimuove l'ultimo token se è EOS ma è stato forzato e min_len non era soddisfatta
    # e il modello continua a generare. Questa logica è complessa.
    # Per ora, ci fidiamo che `generated_ids_this_chunk` contenga la sequenza corretta.
    
    # La funzione restituisce solo i NUOVI token generati in questo chunk.
    return generated_ids_this_chunk

def generate_multi_chunk_midi(
    model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
    total_target_tokens, # Lunghezza totale desiderata per il brano
    model_chunk_capacity, # Max token per un singolo pass del modello (da model_params['max_pe_len'])
    generation_config, # Dizionario con temp, top_k, etc.
    device=DEVICE
):
    all_generated_tokens = []
    current_primer_ids = []
    sos_midi_id = midi_tokenizer[MIDI_SOS_TOKEN_NAME]
    eos_midi_id = midi_tokenizer[MIDI_EOS_TOKEN_NAME]

    # Definisci la lunghezza massima di *nuovi* token da generare per chunk
    # Lascia spazio per SOS e un primer potenziale nel calcolo della capacità del modello
    # Esempio: se model_chunk_capacity è 4096, potremmo puntare a generare chunk di 4000 nuovi token,
    # lasciando spazio per SOS e un primer di ~95 tokens.
    # max_new_tokens_per_chunk = model_chunk_capacity - PRIMER_TOKEN_COUNT - 5 # Piccolo buffer
    # Oppure, più semplicemente, imposta un target ragionevole per i nuovi token per chunk:
    max_new_tokens_per_chunk = min(2048, model_chunk_capacity - PRIMER_TOKEN_COUNT - 5) # Esempio
    if max_new_tokens_per_chunk <= MIN_TOKENS_PER_CHUNK:
        logging.error(f"max_new_tokens_per_chunk ({max_new_tokens_per_chunk}) troppo piccolo. Controlla model_chunk_capacity e PRIMER_TOKEN_COUNT.")
        return []


    while len(all_generated_tokens) < total_target_tokens:
        remaining_tokens_to_generate = total_target_tokens - len(all_generated_tokens)
        
        # Determina quanti *nuovi* token generare in questo chunk
        # Non superare max_new_tokens_per_chunk e non generare più del necessario
        current_pass_max_new = min(max_new_tokens_per_chunk, remaining_tokens_to_generate)
        
        # Assicurati che la lunghezza totale (primer + nuovi token) non superi la capacità del modello
        if len(current_primer_ids) + current_pass_max_new + 1 > model_chunk_capacity: # +1 per SOS
            current_pass_max_new = model_chunk_capacity - len(current_primer_ids) - 1
            if current_pass_max_new < MIN_TOKENS_PER_CHUNK / 2 : # Se rimane pochissimo spazio, forse meglio fermarsi
                 logging.warning("Spazio insufficiente nel chunk per generare una quantità significativa di nuovi token. Interruzione.")
                 break

        if current_pass_max_new < MIN_TOKENS_PER_CHUNK / 2 and len(all_generated_tokens) > 0 : # Evita chunk finali troppo piccoli se abbiamo già qualcosa
            logging.info("Chunk finale sarebbe troppo corto, interruzione generazione.")
            break


        logging.info(f"Generazione nuovo chunk. Tokens totali finora: {len(all_generated_tokens)}/{total_target_tokens}. "
                     f"Primer attuale: {len(current_primer_ids)} tokens. Target nuovi token per questo chunk: {current_pass_max_new}")

        newly_generated_ids = generate_sequence(
            model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
            max_new_tokens=current_pass_max_new,
            min_new_tokens=min(MIN_TOKENS_PER_CHUNK, current_pass_max_new), # Min per questo chunk
            temperature=generation_config.get("temperature", 0.7),
            top_k=generation_config.get("top_k", 40),
            device=device,
            primer_token_ids=current_primer_ids,
            model_max_pe_len=model_chunk_capacity
        )

        if not newly_generated_ids:
            logging.warning("generate_sequence ha restituito un chunk vuoto. Interruzione.")
            break

        # Rimuovi l'EOS se presente e non siamo all'ultimo pezzo o non abbiamo raggiunto total_target_tokens
        chunk_ended_with_eos = False
        if eos_midi_id in newly_generated_ids:
            chunk_ended_with_eos = True
            # Se EOS è generato, considera questo chunk terminato lì.
            eos_index = newly_generated_ids.index(eos_midi_id)
            tokens_to_add = newly_generated_ids[:eos_index + 1] # Include EOS
        else:
            tokens_to_add = newly_generated_ids
        
        all_generated_tokens.extend(tokens_to_add)

        logging.info(f"Chunk generato di {len(tokens_to_add)} tokens. Totale attuale: {len(all_generated_tokens)}.")

        if chunk_ended_with_eos and len(all_generated_tokens) >= total_target_tokens * 0.8: # Se EOS e siamo vicini al target
            logging.info("EOS generato e lunghezza vicina al target. Generazione terminata.")
            break
        elif chunk_ended_with_eos: # Se EOS ma siamo ancora lontani dal target, proviamo a continuare
            logging.info("EOS generato nel chunk, ma la lunghezza totale non è ancora raggiunta. Si tenta di continuare.")
            # Prepara un primer *senza* l'EOS precedente per il prossimo chunk
            # Questo potrebbe portare a un cambio di "frase" musicale.
            # Oppure, potresti decidere di fermarti qui se preferisci non forzare la continuazione dopo un EOS.
            # Per ora, continuiamo.
            if len(tokens_to_add) > PRIMER_TOKEN_COUNT : # Assicurati ci siano abbastanza token per il primer
                 # Prendi gli ultimi token *prima* dell'EOS se EOS era l'ultimo, o semplicemente gli ultimi
                 primer_candidate = tokens_to_add[:-1] if tokens_to_add[-1] == eos_midi_id else tokens_to_add
                 current_primer_ids = primer_candidate[-PRIMER_TOKEN_COUNT:]
            else: # Chunk troppo corto per un primer significativo, ricomincia senza primer
                 current_primer_ids = []

            if not current_primer_ids and len(all_generated_tokens) < total_target_tokens : # Se non possiamo fare un primer, e non abbiamo finito
                 logging.warning("Chunk terminato con EOS troppo corto per un primer, ma non abbiamo finito. Prossimo chunk senza primer.")

        else: # No EOS nel chunk, continua normalmente
            if len(tokens_to_add) >= PRIMER_TOKEN_COUNT:
                current_primer_ids = tokens_to_add[-PRIMER_TOKEN_COUNT:]
            else: # Il chunk generato è più corto del primer richiesto
                current_primer_ids = tokens_to_add # Usa l'intero chunk come primer
        
        if not newly_generated_ids: # Se il chunk era vuoto o solo EOS forzato subito.
            break


    # Assicurati che la sequenza finale (se non già terminata con EOS) abbia un EOS
    # e non superi la total_target_tokens di troppo.
    if all_generated_tokens:
        if all_generated_tokens[-1] != eos_midi_id:
            if len(all_generated_tokens) >= total_target_tokens:
                all_generated_tokens = all_generated_tokens[:total_target_tokens-1] + [eos_midi_id]
            else:
                all_generated_tokens.append(eos_midi_id)
        else: # Finisce già con EOS
            if len(all_generated_tokens) > total_target_tokens:
                 all_generated_tokens = all_generated_tokens[:total_target_tokens-1] + [eos_midi_id]


    return all_generated_tokens

# --- Esecuzione Principale per la Sola Generazione ---

if __name__ == "__main__":
    logging.info("--- Script di Generazione Avviato ---")

    # 1. Carica Tokenizer e Vocabolari (invariato)
    logging.info(f"Caricamento MIDI Tokenizer da: {PATH_VOCAB_MIDI}")
    midi_tokenizer = load_midi_tokenizer_for_inference(PATH_VOCAB_MIDI)
    
    logging.info(f"Caricamento Vocabolario Metadati da: {PATH_VOCAB_METADATA}")
    metadata_vocab_map = load_metadata_vocab_for_inference(PATH_VOCAB_METADATA)

    # 2. Carica il Checkpoint del Modello (invariato)
    if not PATH_MODELLO_CHECKPOINT.exists():
        logging.error(f"File checkpoint modello non trovato: {PATH_MODELLO_CHECKPOINT}")
        sys.exit(1)
    
    logging.info(f"Caricamento checkpoint modello da: {PATH_MODELLO_CHECKPOINT}")
    checkpoint = torch.load(PATH_MODELLO_CHECKPOINT, map_location=DEVICE)

    model_params = checkpoint.get('model_params')
    if not model_params:
        logging.error("ERRORE: 'model_params' non trovato nel checkpoint.")
        sys.exit(1)
    
    # Determina la capacità massima di un chunk dal modello
    # USARE model_params['max_pe_len'] è FONDAMENTALE
    effective_model_chunk_capacity = model_params.get('max_pe_len')
    if not effective_model_chunk_capacity:
        logging.error("ERRORE: 'max_pe_len' non trovato in model_params del checkpoint. Impossibile determinare la capacità del chunk.")
        # Fallback a MAX_SEQ_LEN_MIDI se max_pe_len non è nel checkpoint, ma è rischioso.
        # Aggiungere max_pe_len ai model_params salvati durante il training è la soluzione migliore.
        # Per ora, se non c'è, usiamo un valore di default e logghiamo un warning severo.
        logging.warning(f"max_pe_len non trovato nel checkpoint! Si usa MAX_SEQ_LEN_MIDI ({MAX_SEQ_LEN_MIDI}) come capacità del chunk, "
                        "ma potrebbe causare errori se è maggiore della capacità reale del modello.")
        effective_model_chunk_capacity = MAX_SEQ_LEN_MIDI # Fallback Rischioso

    logging.info(f"Capacità massima del modello per singolo chunk (da max_pe_len): {effective_model_chunk_capacity}")


    # Verifica consistenza dimensioni vocabolari (opzionale ma consigliato) - invariato
    if model_params['src_vocab_size'] != len(metadata_vocab_map):
        logging.warning(f"Dimensione vocabolario metadati del checkpoint ({model_params['src_vocab_size']}) "
                        f"diversa da quella caricata ({len(metadata_vocab_map)}).")
    if model_params['tgt_vocab_size'] != len(midi_tokenizer):
        logging.warning(f"Dimensione vocabolario MIDI del checkpoint ({model_params['tgt_vocab_size']}) "
                        f"diversa da quella caricata ({len(midi_tokenizer)}).")

    # 3. Istanzia e Carica il Modello (invariato)
    logging.info(f"Istanziazione modello con parametri: {model_params}")
    model = Seq2SeqTransformer(**model_params).to(DEVICE)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        logging.error(f"Errore durante il caricamento di model_state_dict: {e}")
        sys.exit(1)
        
    model.eval()
    logging.info("Modello caricato e impostato in modalità valutazione.")

    # 4. Prepara Prompt e Genera
    TOTAL_MIDI_TARGET_LENGTH = 8000 # Esempio: lunghezza totale desiderata in token
                                      # Aumenta questo valore per brani più lunghi.
                                      # Se è più grande di effective_model_chunk_capacity, verranno generati più chunk.

    example_metadata_prompt_list = [
        ["Style=Folk", "Key=A_minor", "TimeSig=4/4", "Instrument=Piano", "Instrument=Flute", "BPM_120", "AvgVel_Media_45-80"],
        ["Style=Classical", "Key=C_Major", "TimeSig=4/4", "Instrument=Violin", "Instrument=Cello", "BPM_80", "VelRange_Ampio_55+"],
        ["Style=Techno", "Key=G_Major", "TimeSig=4/4", "Instrument=Synth_Bass_1", "Instrument=Synth_Drum", "BPM_140"],
    ]

    generation_params = {
        "temperature": 0.75,
        "top_k": 40
    }

    for idx, example_metadata_prompt in enumerate(example_metadata_prompt_list):
        logging.info(f"--- Generazione {idx+1}/{len(example_metadata_prompt_list)} ---")
        logging.info(f"Prompt metadati: {example_metadata_prompt}")
        logging.info(f"Target lunghezza totale MIDI: {TOTAL_MIDI_TARGET_LENGTH} tokens.")

        try:
            # Usa la nuova funzione multi-chunk
            generated_token_ids = generate_multi_chunk_midi(
                model, midi_tokenizer, metadata_vocab_map, example_metadata_prompt,
                total_target_tokens=TOTAL_MIDI_TARGET_LENGTH,
                model_chunk_capacity=effective_model_chunk_capacity,
                generation_config=generation_params,
                device=DEVICE
            )

            if generated_token_ids:
                logging.info(f"Generati {len(generated_token_ids)} token MIDI totali.")
                
                generated_midi_object = midi_tokenizer.decode(generated_token_ids)
                
                if generated_midi_object:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    # ... (logica per il nome del file invariata)
                    prompt_name_part = "_".join(example_metadata_prompt).replace("=", "").replace("/", "").replace("Style", "").replace("Key", "").replace("TimeSig","").replace("Title","").replace("Instrument","").replace("BPM","").replace("AvgVel","").replace("VelRange","")
                    prompt_name_part = ''.join(c for c in prompt_name_part if c.isalnum() or c == '_')[:70] # Aumentato un po'
                    
                    output_filename_path = GENERATION_OUTPUT_DIR / f"generated_long_{prompt_name_part}_{timestamp}_{idx}.mid"
                    GENERATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    
                    generated_midi_object.dump_midi(str(output_filename_path))
                    logging.info(f"File MIDI (potenzialmente multi-chunk) salvato in: {output_filename_path}")
                else:
                    logging.warning(f"midi_tokenizer.decode ha restituito None per prompt '{example_metadata_prompt}'.")
            else:
                logging.warning(f"Generazione per prompt '{example_metadata_prompt}' fallita o ha prodotto una sequenza vuota.")
        
        except Exception as e:
            logging.error(f"Errore durante la generazione o il salvataggio per prompt '{example_metadata_prompt}': {e}", exc_info=True)
        
        if idx < len(example_metadata_prompt_list) - 1:
            time.sleep(1)

    logging.info("--- Script di Generazione Terminato ---")