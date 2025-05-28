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
                      max_len=10000, min_len=5000, temperature=0.5, top_k=None, device=DEVICE,
                      log_progress_every_n_tokens=100): # Aggiunto min_len e log_progress_every_n_tokens
    model.eval()
    try:
        sos_meta_id = metadata_vocab_map[META_SOS_TOKEN_NAME]
        eos_meta_id = metadata_vocab_map[META_EOS_TOKEN_NAME]
        unk_meta_id = metadata_vocab_map[META_UNK_TOKEN_NAME]
        meta_pad_id = metadata_vocab_map[META_PAD_TOKEN_NAME]

        sos_midi_id = midi_tokenizer[MIDI_SOS_TOKEN_NAME]
        eos_midi_id = midi_tokenizer[MIDI_EOS_TOKEN_NAME]
    except KeyError as e:
        logging.error(f"Errore critico: Token speciale '{e}' non trovato nei vocabolari durante la preparazione per la generazione.")
        sys.exit(1)

    meta_token_ids = [metadata_vocab_map.get(token, unk_meta_id) for token in metadata_prompt]
    src_seq = torch.tensor([[sos_meta_id] + meta_token_ids[:MAX_SEQ_LEN_META-2] + [eos_meta_id]], dtype=torch.long, device=device)
    src_padding_mask = (src_seq == meta_pad_id)

    logging.info(f"Avvio generazione per prompt: {metadata_prompt}. Lunghezza max: {max_len}, min: {min_len}") # Log iniziale

    with torch.no_grad():
        memory = model.encode(src_seq, src_padding_mask)
        memory_key_padding_mask = src_padding_mask
        tgt_tokens = torch.tensor([[sos_midi_id]], dtype=torch.long, device=device)
        generated_eos = False 

        for i in range(max_len - 1): # max_len -1 perché iniziamo con SOS e generiamo un token alla volta
            tgt_len = tgt_tokens.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool), diagonal=1)
            tgt_padding_mask_step = torch.zeros_like(tgt_tokens, dtype=torch.bool, device=device)

            decoder_output = model.decode(tgt=tgt_tokens, memory=memory, tgt_mask=tgt_mask,
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
            next_token_id = torch.multinomial(probs, num_samples=1)

            current_length_no_sos = tgt_tokens.size(1) - 1 # Lunghezza attuale dei token generati (escluso SOS)

            if next_token_id.item() == eos_midi_id and current_length_no_sos < min_len:
                if probs.size(-1) > 1:
                    top_2_probs, top_2_indices = torch.topk(probs, 2, dim=-1)
                    if top_2_indices[0, 0].item() == eos_midi_id:
                        if top_2_indices.size(1) > 1:
                            next_token_id = top_2_indices[0, 1].unsqueeze(0).unsqueeze(0)
                        else:
                            logging.warning("Forzato a generare EOS nonostante min_len non raggiunta, poche opzioni.")
                            generated_eos = True 
                    # else: Non era EOS, quindi next_token_id è già buono
                # else: solo una scelta possibile

            tgt_tokens = torch.cat((tgt_tokens, next_token_id), dim=1)
            current_length_generated = tgt_tokens.size(1) -1 

            # --- MODIFICA: LOGGING PROGRESSO ---
            if (i + 1) % log_progress_every_n_tokens == 0:
                logging.info(f"Generazione in corso... Token generati: {current_length_generated}/{max_len}")
            # --- FINE MODIFICA ---

            if next_token_id.item() == eos_midi_id and not generated_eos and current_length_generated >= min_len:
                logging.info(f"Token EOS generato dopo {current_length_generated} token (e min_len raggiunta).")
                generated_eos = True
                break 

            if generated_eos and current_length_generated < min_len : 
                generated_eos = False 

            if current_length_generated >= max_len: 
                logging.info(f"Raggiunta lunghezza massima di generazione ({max_len}).")
                if not generated_eos: # Se non abbiamo ancora generato EOS
                     logging.info("Aggiunta forzata del token EOS a max_len.")
                     # Rimuovi l'ultimo token se non è EOS e aggiungi EOS
                     # Questa logica è già gestita dopo il ciclo, ma potremmo volerla qui se max_len è rigido.
                     # Per ora, il post-processing gestirà l'aggiunta di EOS se necessario.
                break
    
    final_tokens = tgt_tokens[0, 1:].tolist() # Rimuovi SOS iniziale

    if not generated_eos and len(final_tokens) >= min_len :
        if final_tokens and final_tokens[-1] != eos_midi_id:
            if len(final_tokens) < max_len :
                logging.info(f"Aggiunta token EOS alla fine della sequenza (lunghezza: {len(final_tokens)}).")
                final_tokens.append(eos_midi_id)
            elif len(final_tokens) == max_len: # Se siamo già a max_len, sostituisci l'ultimo con EOS
                logging.info(f"Sostituzione ultimo token con EOS a max_len (lunghezza: {len(final_tokens)}).")
                final_tokens[-1] = eos_midi_id
    elif generated_eos and final_tokens and final_tokens[-1] != eos_midi_id :
        # Questo scenario dovrebbe essere raro se EOS è stato generato e il ciclo è uscito correttamente.
        # Ma per sicurezza, se EOS era stato flaggato ma non è l'ultimo token (e siamo oltre min_len).
        if len(final_tokens) >= min_len:
            logging.warning(f"EOS era stato flaggato ma non è l'ultimo token. Assicurando EOS alla fine (lunghezza attuale: {len(final_tokens)}).")
            if len(final_tokens) < max_len:
                final_tokens.append(eos_midi_id)
            else:
                final_tokens[-1] = eos_midi_id


    if len(final_tokens) < min_len:
        logging.warning(f"Lunghezza finale generata ({len(final_tokens)}) è inferiore a min_len ({min_len}) richiesta.")
    
    logging.info(f"Generazione completata. Numero finale di token (inclusi SOS/EOS se presenti): {len(final_tokens)}")
    return final_tokens

# --- Esecuzione Principale per la Sola Generazione ---

if __name__ == "__main__":
    logging.info("--- Script di Generazione Avviato ---")

    # 1. Carica Tokenizer e Vocabolari
    logging.info(f"Caricamento MIDI Tokenizer da: {PATH_VOCAB_MIDI}")
    midi_tokenizer = load_midi_tokenizer_for_inference(PATH_VOCAB_MIDI)
    
    logging.info(f"Caricamento Vocabolario Metadati da: {PATH_VOCAB_METADATA}")
    metadata_vocab_map = load_metadata_vocab_for_inference(PATH_VOCAB_METADATA)

    # 2. Carica il Checkpoint del Modello
    if not PATH_MODELLO_CHECKPOINT.exists():
        logging.error(f"File checkpoint modello non trovato: {PATH_MODELLO_CHECKPOINT}")
        sys.exit(1)
    
    logging.info(f"Caricamento checkpoint modello da: {PATH_MODELLO_CHECKPOINT}")
    checkpoint = torch.load(PATH_MODELLO_CHECKPOINT, map_location=DEVICE)

    model_params = checkpoint.get('model_params')
    if not model_params:
        logging.error("ERRORE: 'model_params' non trovato nel checkpoint. Impossibile ricostruire il modello.")
        sys.exit(1)
    
    if model_params['src_vocab_size'] != len(metadata_vocab_map):
        logging.warning(f"Dimensione vocabolario metadati del checkpoint ({model_params['src_vocab_size']}) "
                        f"diversa da quella caricata ({len(metadata_vocab_map)}). "
                        "Assicurati che i file di vocabolario siano corretti.")
    if model_params['tgt_vocab_size'] != len(midi_tokenizer):
        logging.warning(f"Dimensione vocabolario MIDI del checkpoint ({model_params['tgt_vocab_size']}) "
                        f"diversa da quella caricata ({len(midi_tokenizer)}). "
                        "Assicurati che i file di vocabolario siano corretti.")

    # 3. Istanzia e Carica il Modello
    logging.info(f"Istanziazione modello con parametri: {model_params}")
    # Assicurati che max_pe_len sia presente nei model_params o definito globalmente
    if 'max_pe_len' not in model_params:
        logging.warning("'max_pe_len' non trovato in model_params. Utilizzo di MAX_SEQ_LEN_MIDI + MAX_SEQ_LEN_META + 100 come fallback.")
        model_params['max_pe_len'] = MAX_SEQ_LEN_MIDI + MAX_SEQ_LEN_META + 100 # Fallback ragionevole
        
    model = Seq2SeqTransformer(**model_params).to(DEVICE)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        logging.error(f"Errore durante il caricamento di model_state_dict: {e}")
        logging.error("Possibili cause: discrepanza nell'architettura del modello o nei nomi dei layer.")
        logging.error("Verifica che i parametri in 'model_params' corrispondano esattamente all'architettura salvata.")
        sys.exit(1)
        
    model.eval()
    logging.info("Modello caricato e impostato in modalità valutazione.")

    # 4. Prepara Prompt e Genera
    example_metadata_prompt_list = [
        ["Style=Folk", "Key=A_minor", "TimeSig=4/4", "Instrument=Piano", "Instrument=Flute"],
        ["Style=Classical", "Key=C_Major", "TimeSig=4/4", "Instrument=Violin", "Instrument=Cello"],
    ]

    # === NUOVA CONFIGURAZIONE PER IL LOGGING DEL PROGRESSO ===
    # Quanti token generare prima di stampare un messaggio di log sul progresso
    LOG_PROGRESS_EVERY_N_TOKENS = 200 # Puoi aggiustare questo valore
    # === ============================================= ===

    for idx, example_metadata_prompt in enumerate(example_metadata_prompt_list):
        logging.info(f"--- Generazione {idx+1}/{len(example_metadata_prompt_list)} ---")
        # Non c'è bisogno di loggare qui il prompt, viene fatto dentro generate_sequence

        try:
            generated_token_ids = generate_sequence(
                model, midi_tokenizer, metadata_vocab_map, example_metadata_prompt,
                max_len=MAX_SEQ_LEN_MIDI, # Usa la costante definita sopra
                min_len=2000, # Esempio di min_len, aggiusta se necessario
                temperature=0.75,
                top_k=40,
                device=DEVICE,
                log_progress_every_n_tokens=LOG_PROGRESS_EVERY_N_TOKENS # Passa la frequenza di log
            )

            if generated_token_ids:
                logging.info(f"Generati {len(generated_token_ids)} token MIDI per prompt {idx+1}.")
                
                generated_midi_object = midi_tokenizer.decode(generated_token_ids)
                
                if generated_midi_object:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    prompt_name_part = "_".join(example_metadata_prompt).replace("=", "").replace("/", "").replace("Style", "").replace("Key", "").replace("TimeSig","").replace("Title","")
                    prompt_name_part = ''.join(c for c in prompt_name_part if c.isalnum() or c == '_')[:50]
                    
                    output_filename_path = GENERATION_OUTPUT_DIR / f"generated_{prompt_name_part}_{timestamp}_{idx}.mid"

                    GENERATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    
                    logging.info(f"Tentativo di salvare il MIDI generato in: {output_filename_path}")
                    generated_midi_object.dump_midi(str(output_filename_path))
                    logging.info(f"File MIDI salvato in: {output_filename_path}")
                else:
                    logging.warning(f"midi_tokenizer.decode ha restituito None per prompt '{example_metadata_prompt}' (prompt {idx+1}), impossibile salvare.")

            else:
                logging.warning(f"Generazione per prompt '{example_metadata_prompt}' (prompt {idx+1}) fallita o ha prodotto una sequenza vuota di token ID.")
        
        except Exception as e:
            logging.error(f"Errore durante la generazione o il salvataggio per prompt '{example_metadata_prompt}' (prompt {idx+1}): {e}", exc_info=True)
        
        if idx < len(example_metadata_prompt_list) - 1:
            logging.info(f"Attesa di 1 secondo prima della prossima generazione...")
            time.sleep(1)

    logging.info("--- Script di Generazione Terminato ---")