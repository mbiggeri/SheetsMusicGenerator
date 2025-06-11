import argparse
from pathlib import Path
import sys

# --- USAGE: python rename_midi.py /path/to/directory -- Questo script rinomina le estensioni dei file MIDI in minuscolo.

def normalize_midi_extensions(root_directory: Path):
    """
    Scansiona ricorsivamente una cartella, rinomina i file con estensioni
    MIDI in .mid (minuscolo) ed elimina i file con estensioni miste se
    esiste già una versione con estensione .mid.

    Args:
        root_directory (Path): Il percorso della cartella da scansionare.
    """
    # 1. Validazione dell'input
    if not root_directory.is_dir():
        print(f"Errore: Il percorso specificato '{root_directory}' non è una cartella valida.", file=sys.stderr)
        sys.exit(1)

    print(f"Scansione della cartella '{root_directory}' e delle sue sottocartelle...")
    print("ATTENZIONE: I file duplicati con estensione maiuscola/mista verranno ELIMINATI.")

    # 2. Ricerca dei file
    files_to_check = [f for f in root_directory.rglob('*') if f.is_file()]

    # --- MODIFICA: Aggiunto contatore per i file eliminati ---
    renamed_count = 0
    deleted_count = 0
    skipped_count = 0

    # 3. Iterazione e rinomina/eliminazione
    for file_path in files_to_check:
        # Controlla se l'estensione in minuscolo è '.mid' ma quella originale non lo è
        if file_path.suffix.lower() == '.mid' and file_path.suffix != '.mid':
            
            new_file_path = file_path.with_suffix('.mid')

            # --- INIZIO BLOCCO DI LOGICA MODIFICATO ---
            # 4. Controllo di sicurezza: se un file con il nuovo nome esiste già, elimina l'originale.
            if new_file_path.exists():
                try:
                    # Elimina il file originale (es. 'brano.MID')
                    file_path.unlink()
                    print(f"  [ELIMINATO] '{file_path.name}' perché esiste già una versione con estensione minuscola ('{new_file_path.name}').")
                    deleted_count += 1
                except OSError as e:
                    print(f"  [ERRORE] Impossibile eliminare '{file_path.name}'. Dettagli: {e}")
                    skipped_count += 1
                
                # Prosegui al file successivo in ogni caso
                continue
            # --- FINE BLOCCO DI LOGICA MODIFICATO ---

            # 5. Se non esiste un duplicato, esegue la rinomina (logica originale)
            try:
                file_path.rename(new_file_path)
                print(f"  [RINOMINATO] '{file_path.name}' -> '{new_file_path.name}'")
                renamed_count += 1
            except OSError as e:
                print(f"  [ERRORE] Impossibile rinominare '{file_path.name}'. Dettagli: {e}")
                skipped_count += 1

    # --- MODIFICA: Aggiornato il riepilogo finale ---
    # 6. Riepilogo finale
    print("\n--- Operazione Completata ---")
    print(f"File rinominati con successo: {renamed_count}")
    print(f"File duplicati eliminati: {deleted_count}")
    if skipped_count > 0:
        print(f"File saltati (per errori): {skipped_count}")
    if renamed_count == 0 and deleted_count == 0 and skipped_count == 0:
        print("Nessun file con estensione da correggere o eliminare è stato trovato.")

def main():
    """
    Funzione principale per gestire gli argomenti da riga di comando.
    """
    parser = argparse.ArgumentParser(
        description="Normalizza ricorsivamente le estensioni dei file MIDI (.mid) ed elimina i duplicati con estensioni maiuscole/miste."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Il percorso della cartella radice che contiene i file MIDI da normalizzare."
    )
    
    args = parser.parse_args()
    
    target_directory = Path(args.directory)
    
    normalize_midi_extensions(target_directory)


if __name__ == "__main__":
    main()