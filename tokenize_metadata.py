import re

# Inserisci qui tutte le costanti condivise, se vuoi
# Questa variabile deve corrispondere a quella in config.py per attivare la logica corretta.
PROCESSING_MODE = "genres" 
META_PAD_TOKEN_NAME = "<pad_meta>"
META_UNK_TOKEN_NAME = "<unk_meta>"
META_SOS_TOKEN_NAME = "<sos_meta>"
META_EOS_TOKEN_NAME = "<eos_meta>"

def tokenize_metadata(metadata_dict):
    """
    Versione UNICA e ROBUSTA per la tokenizzazione dei metadati.
    Usata sia da dataset_creator.py che da training.py.
    """
    tokens = []

    # Se la modalità è "genres", estraiamo SOLO il genere e ci fermiamo.
    if PROCESSING_MODE == "genres":
        if 'genre' in metadata_dict and metadata_dict['genre']:
            # Pulisce il nome del genere per renderlo un token valido
            # Esempio: "Classic Rock" -> "Genre=Classic_Rock"
            genre_name = str(metadata_dict['genre']).strip().replace(' ', '_')
            clean_genre = re.sub(r'[^a-zA-Z0-9_]', '', genre_name) # Rimuove caratteri non validi
            if clean_genre:
                tokens.append(f"Genre={clean_genre}")
        
        # Restituisce solo il token del genere (o una lista vuota se non trovato)
        return tokens
    # =================================================
    
    key_to_tokenize_str = None

    # Logica robusta per la tonalità
    if PROCESSING_MODE == "piano_only" and 'transposed_to_key' in metadata_dict and metadata_dict['transposed_to_key']:
        raw_transposed_info = metadata_dict['transposed_to_key'].lower()
        # Controlla se le sottostringhe chiave sono presenti, ignorando il formato esatto.
        if "c major" in raw_transposed_info and "a minor" in raw_transposed_info:
            key_to_tokenize_str = "Target_Cmaj_Amin"
        else:
            # Fallback per chiavi trasposte che non sono la destinazione standard
            temp_key = str(raw_transposed_info).replace(' ', '_').replace('/', '_').replace('#','sharp')
            key_to_tokenize_str = re.sub(r'[^a-zA-Z0-9_]', '', temp_key)
            if not key_to_tokenize_str:
                key_to_tokenize_str = "Unknown_Transposed_Key"
                   
    if not key_to_tokenize_str:
        key_source = metadata_dict.get('key') or \
                     metadata_dict.get('music21_detected_key') or \
                     metadata_dict.get('mido_declared_key_signature')
        if key_source:
             key_to_tokenize_str = str(key_source)

    if key_to_tokenize_str:
        clean_key_token = str(key_to_tokenize_str).replace(' ', '_').replace('#', 'sharp')
        clean_key_token = re.sub(r'[^a-zA-Z0-9_]', '', clean_key_token)
        if clean_key_token:
            tokens.append(f"Key={clean_key_token}")

    # 2. Metro (Time Signature) - Invariato
    if 'time_signature' in metadata_dict and metadata_dict['time_signature']:
        tokens.append(f"TimeSig={metadata_dict['time_signature']}")

    # 3. BPM (Battiti Per Minuto) - Nuovo, categorizzato
    if 'bpm_rounded' in metadata_dict and metadata_dict['bpm_rounded'] is not None:
        bpm = metadata_dict['bpm_rounded']
        if bpm <= 0: # Improbabile ma gestiamo
            token_bpm = "Tempo_Unknown"
        elif bpm <= 60:
            token_bpm = "Tempo_VerySlow"  # (es. Largo, Grave)
        elif bpm <= 76: # Fino a Adagio
            token_bpm = "Tempo_Slow"
        elif bpm <= 108: # Fino a Andante/Moderato
            token_bpm = "Tempo_Moderate"
        elif bpm <= 132: # Fino a Allegro
            token_bpm = "Tempo_Fast"
        elif bpm <= 168: # Fino a Vivace/Presto
            token_bpm = "Tempo_VeryFast"
        else: # Prestissimo
            token_bpm = "Tempo_ExtremelyFast"
        tokens.append(token_bpm)

    # 4. Velocity Media - Nuovo, categorizzato
    if 'avg_velocity_rounded' in metadata_dict and metadata_dict['avg_velocity_rounded'] is not None:
        avg_vel = metadata_dict['avg_velocity_rounded']
        if avg_vel < 0 : token_avg_vel = "AvgVel_Unknown" # Improbabile
        elif avg_vel <= 35: # Pianissimo (pp) / Piano (p)
            token_avg_vel = "AvgVel_VeryLow"
        elif avg_vel <= 60: # MezzoPiano (mp)
            token_avg_vel = "AvgVel_Low"
        elif avg_vel <= 85: # MezzoForte (mf)
            token_avg_vel = "AvgVel_Medium"
        elif avg_vel <= 110: # Forte (f)
            token_avg_vel = "AvgVel_High"
        else: # Fortissimo (ff)
            token_avg_vel = "AvgVel_VeryHigh"
        tokens.append(token_avg_vel)

    # 5. Range di Velocity - Nuovo, categorizzato
    if 'velocity_range_rounded' in metadata_dict and metadata_dict['velocity_range_rounded'] is not None:
        vel_range = metadata_dict['velocity_range_rounded']
        if vel_range < 0: token_vel_range = "VelRange_Unknown" # Improbabile
        elif vel_range <= 15: # Poca variazione dinamica
            token_vel_range = "VelRange_Narrow"
        elif vel_range <= 40:
            token_vel_range = "VelRange_Medium"
        elif vel_range <= 70:
            token_vel_range = "VelRange_Wide"
        else: # Variazione dinamica molto ampia
            token_vel_range = "VelRange_VeryWide"
        tokens.append(token_vel_range)

    # 6. Numero di Strumenti (Opzionale, ma può essere utile per lo "stile")
    num_instruments = 0
    if 'midi_instruments' in metadata_dict and isinstance(metadata_dict['midi_instruments'], list):
        num_instruments = len(metadata_dict['midi_instruments'])
    
    if num_instruments == 0:
        token_num_inst = "NumInst_None" # O potresti ometterlo
    elif num_instruments == 1:
        token_num_inst = "NumInst_Solo"
    elif num_instruments == 2:
        token_num_inst = "NumInst_Duet"
    elif num_instruments <= 4: # Trio, Quartetto
        token_num_inst = "NumInst_SmallChamber"
    elif num_instruments <= 8: # Ensemble piccolo/medio
        token_num_inst = "NumInst_MediumEnsemble"
    else: # Ensemble grande
        token_num_inst = "NumInst_LargeEnsemble"
    tokens.append(token_num_inst)


    # 7. Nomi degli Strumenti (Logica precedente, adattata per chiarezza)
    instrument_tokens_added_from_midi_list = False
    if 'midi_instruments' in metadata_dict and isinstance(metadata_dict['midi_instruments'], list) and metadata_dict['midi_instruments']:
        # Priorità alla lista di strumenti direttamente dal MIDI se presente e valida e non vuota
        for instrument_name in metadata_dict['midi_instruments']:
            if instrument_name and isinstance(instrument_name, str): # Verifica aggiuntiva
                # Pulisci il nome dello strumento per creare un token valido
                # Rimuovi caratteri speciali, spazi, normalizza case se necessario.
                # Qui usiamo una pulizia semplice.
                clean_instrument_name = instrument_name.replace(' ', '_').replace('(', '').replace(')', '').replace('#','sharp')
                clean_instrument_name = re.sub(r'[^a-zA-Z0-9_]', '', clean_instrument_name) # Mantieni solo alfanumerici e underscore
                if clean_instrument_name: # Assicurati che non sia vuoto dopo la pulizia
                    tokens.append(f"Instrument={clean_instrument_name}")
                    instrument_tokens_added_from_midi_list = True
                    
    # 8. Uso del Sustain Pedal e del Pitch Bend (aggiunge il token solo se usati)
    if metadata_dict.get('sustain_pedal_used', False):
        tokens.append("Sustain=Used")

    if metadata_dict.get('pitch_bend_used', False):
        tokens.append("PitchBend=Used")
        
    
    # Fallback a mutopiainstrument se midi_instruments non ha prodotto token
    # (o se vuoi che 'mutopiainstrument' aggiunga/sovrascriva - modifica la logica di conseguenza)
    if not instrument_tokens_added_from_midi_list and 'mutopiainstrument' in metadata_dict and metadata_dict['mutopiainstrument']:
        instrument_string = metadata_dict['mutopiainstrument']
        # Sostituisci " and " e altri separatori comuni
        instrument_string_normalized = re.sub(r'\s+(?:and|,|&)\s+', ',', instrument_string, flags=re.IGNORECASE)
        instrument_string_normalized = re.sub(r'[()]', '', instrument_string_normalized) # Rimuovi parentesi

        instrument_names_from_ly = [name.strip() for name in instrument_string_normalized.split(',') if name.strip()]
        
        for instrument_name in instrument_names_from_ly:
            if instrument_name:
                clean_instrument_name = instrument_name.replace(' ', '_').replace('#','sharp')
                clean_instrument_name = re.sub(r'[^a-zA-Z0-9_]', '', clean_instrument_name)
                if clean_instrument_name:
                    tokens.append(f"Instrument={clean_instrument_name}")
    
    return tokens