# tokenize_metadata.py
import re

# --- Rimuoviamo la costante globale PROCESSING_MODE ---

def tokenize_metadata(metadata_dict):
    """
    Versione UNICA e ROBUSTA per la tokenizzazione dei metadati.
    Questa funzione ora tokenizza TUTTI i metadati presenti nel dizionario
    passato come argomento, in modo che sia flessibile e modulare.
    """
    tokens = []

    # 1. Genere (se presente)
    if 'genre' in metadata_dict and metadata_dict['genre']:
        genre_name = str(metadata_dict['genre']).strip().replace(' ', '_')
        clean_genre = re.sub(r'[^a-zA-Z0-9_]', '', genre_name)
        if clean_genre:
            tokens.append(f"Genre={clean_genre}")
        
        return tokens  # Se il genere è presente, ritorniamo subito i token
    
    key_to_tokenize_str = None

    # 2. Logica robusta per la tonalità
    # Se il file è stato trasposto, usiamo l'indicazione della trasposizione.
    if 'transposed_to_key' in metadata_dict and metadata_dict['transposed_to_key']:
        raw_transposed_info = metadata_dict['transposed_to_key'].lower()
        if "c major" in raw_transposed_info and "a minor" in raw_transposed_info:
            key_to_tokenize_str = "Target_Cmaj_Amin"
        else:
            temp_key = str(raw_transposed_info).replace(' ', '_').replace('/', '_').replace('#','sharp')
            key_to_tokenize_str = re.sub(r'[^a-zA-Z0-9_]', '', temp_key) or "Unknown_Transposed_Key"
                   
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

    # 3. Metro (Time Signature)
    if 'time_signature' in metadata_dict and metadata_dict['time_signature']:
        tokens.append(f"TimeSig={metadata_dict['time_signature']}")

    # 4. BPM (Battiti Per Minuto), categorizzato
    if 'bpm_rounded' in metadata_dict and metadata_dict['bpm_rounded'] is not None:
        bpm = metadata_dict['bpm_rounded']
        if bpm <= 60: token_bpm = "Tempo_VerySlow"
        elif bpm <= 76: token_bpm = "Tempo_Slow"
        elif bpm <= 108: token_bpm = "Tempo_Moderate"
        elif bpm <= 132: token_bpm = "Tempo_Fast"
        elif bpm <= 168: token_bpm = "Tempo_VeryFast"
        else: token_bpm = "Tempo_ExtremelyFast"
        tokens.append(token_bpm)

    # 5. Velocity Media, categorizzata
    if 'avg_velocity_rounded' in metadata_dict and metadata_dict['avg_velocity_rounded'] is not None:
        avg_vel = metadata_dict['avg_velocity_rounded']
        if avg_vel <= 35: token_avg_vel = "AvgVel_VeryLow"
        elif avg_vel <= 60: token_avg_vel = "AvgVel_Low"
        elif avg_vel <= 85: token_avg_vel = "AvgVel_Medium"
        elif avg_vel <= 110: token_avg_vel = "AvgVel_High"
        else: token_avg_vel = "AvgVel_VeryHigh"
        tokens.append(token_avg_vel)

    # 6. Range di Velocity, categorizzato
    if 'velocity_range_rounded' in metadata_dict and metadata_dict['velocity_range_rounded'] is not None:
        vel_range = metadata_dict['velocity_range_rounded']
        if vel_range <= 15: token_vel_range = "VelRange_Narrow"
        elif vel_range <= 40: token_vel_range = "VelRange_Medium"
        elif vel_range <= 70: token_vel_range = "VelRange_Wide"
        else: token_vel_range = "VelRange_VeryWide"
        tokens.append(token_vel_range)

    # 7. Numero di Strumenti e Nomi
    if 'midi_instruments' in metadata_dict and isinstance(metadata_dict['midi_instruments'], list):
        num_instruments = len(metadata_dict['midi_instruments'])
        if num_instruments == 1: tokens.append("NumInst_Solo")
        elif num_instruments == 2: tokens.append("NumInst_Duet")
        elif num_instruments <= 4: tokens.append("NumInst_SmallChamber")
        elif num_instruments <= 8: tokens.append("NumInst_MediumEnsemble")
        elif num_instruments > 8: tokens.append("NumInst_LargeEnsemble")
        
        for instrument_name in metadata_dict['midi_instruments']:
            if instrument_name and isinstance(instrument_name, str):
                clean_instrument_name = re.sub(r'[^a-zA-Z0-9_]', '', instrument_name.replace(' ', '_').replace('(', '').replace(')', '').replace('#','sharp'))
                if clean_instrument_name:
                    tokens.append(f"Instrument={clean_instrument_name}")
    
    # ... (eventuale logica di fallback per mutopiainstrument, se necessaria) ...

    return tokens