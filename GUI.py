import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Listbox, MULTIPLE, END, StringVar
import json
from pathlib import Path
import threading
import logging
import random
import os
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from queue import Empty

try:
    from generate_music import run_generation, get_model_info
except ImportError:
    messagebox.showerror("Errore", "File 'generate_music.py' non trovato. Assicurati che sia nella stessa cartella.")
    exit()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MusicGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Generatore Musicale Transformer v2.0")
        self.root.geometry("900x750")
        self.root.minsize(850, 700)

        # --- Variabili di Stato (invariate) ---
        self.model_path = tk.StringVar()
        self.midi_vocab_path = tk.StringVar()
        self.meta_vocab_path = tk.StringVar()
        self.meta_freq_vocab_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path("./generated_midi_from_gui").resolve()))
        self.primer_midi_path = tk.StringVar()
        self.profiles_path = tk.StringVar()
        self.random_instruments_var = tk.BooleanVar(value=False)
        self.generation_mode = tk.StringVar(value="Genere")
        self.selected_profile = tk.StringVar()
        self.selected_genre = tk.StringVar()
        self.rest_penalty_mode = StringVar(value='hybrid')
        
        # Variabili Parametri Generazione
        self.total_tokens_var = tk.StringVar(value="1024")
        self.temperature_var = tk.StringVar(value="1.0")
        self.top_k_var = tk.StringVar(value="40")
        self.max_rest_penalty_var = tk.DoubleVar(value=3.5)
        self.num_workers_var = tk.StringVar(value=str(max(1, os.cpu_count() // 2)))

        # Variabili Pulizia MIDI
        self.clean_remove_sustain_var = tk.BooleanVar(value=True)
        self.clean_merge_tracks_var = tk.BooleanVar(value=True)
        self.clean_trim_silence_var = tk.BooleanVar(value=True)
        self.clean_quantize_var = tk.BooleanVar(value=True)
        self.clean_quantize_grid_var = tk.StringVar(value="1/16")
        self.clean_limit_polyphony_var = tk.BooleanVar(value=True)
        self.clean_polyphony_max_var = tk.StringVar(value="12")
        self.clean_filter_pitch_var = tk.BooleanVar(value=True)
        self.clean_pitch_min_var = tk.StringVar(value="21")
        self.clean_pitch_max_var = tk.StringVar(value="108")
        self.force_piano_only_var = tk.BooleanVar(value=False)
        self.clean_filter_velocity_var = tk.BooleanVar(value=True)
        self.clean_velocity_min_var = tk.StringVar(value="20")
        self.clean_velocity_max_var = tk.StringVar(value="120")

        # --- Gestione Coda e Parallelismo ---
        self.jobs_list = []
        self.job_counter = 0
        self.executor = None
        self.active_futures = {}
        self.manager = multiprocessing.Manager()
        self.progress_queue = self.manager.Queue()

        # Dizionari per widget e dati
        self.control_vars = {}
        self.combobox_widgets = {}
        self.metadata_options = {}
        self.profiles = []

        self.setup_styles()
        self.create_widgets()
        self.toggle_ui_mode()
        self.check_progress_queue()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styles(self):
        """Imposta uno stile Ttk più moderno."""
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use('clam')
        except tk.TclError:
            # 'clam' potrebbe non essere disponibile su tutti i sistemi, usa un fallback
            pass
        self.style.configure("TLabelFrame.Label", font=('Segoe UI', 11, 'bold'))
        self.style.configure("TButton", font=('Segoe UI', 10))
        self.style.configure("Status.TLabel", font=('Segoe UI', 9))
        self.style.configure("Path.TLabel", foreground="#007acc", font=('Segoe UI', 9))
        self.style.configure("Desc.TLabel", foreground="gray", font=('Segoe UI', 9))

    def create_widgets(self):
        """Crea la struttura principale dell'interfaccia con le schede."""
        main_pane = ttk.Frame(self.root, padding=10)
        main_pane.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(main_pane)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Creazione delle singole schede
        config_tab = self.create_tab(notebook, "1. Configurazione")
        generation_tab = self.create_tab(notebook, "2. Generazione")
        cleaning_tab = self.create_tab(notebook, "3. Pulizia MIDI")
        queue_tab = self.create_tab(notebook, "4. Coda & Stato")

        # Popolamento di ogni scheda
        self.populate_config_tab(config_tab)
        self.populate_generation_tab(generation_tab)
        self.populate_cleaning_tab(cleaning_tab)
        self.populate_queue_tab(queue_tab)

    def create_tab(self, notebook, text):
        """Crea un frame per una scheda e lo aggiunge al notebook."""
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text=text)
        return frame
        
    def populate_config_tab(self, tab):
        """Popola la scheda di configurazione con i selettori di file."""
        tab.columnconfigure(1, weight=1)
        
        file_frame = ttk.LabelFrame(tab, text="Percorsi File Essenziali", padding=10)
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        file_frame.columnconfigure(1, weight=1)

        self.create_file_selector(file_frame, "Modello (.pt):", self.model_path, self.browse_model, 0, has_analyze_button=True)
        self.create_file_selector(file_frame, "Vocabolario MIDI (.json):", self.midi_vocab_path, self.browse_midi_vocab, 1)
        self.create_file_selector(file_frame, "Vocabolario Metadati (.json):", self.meta_vocab_path, self.browse_meta_vocab, 2)
        
        ttk.Button(file_frame, text="Carica Vocabolari e Analizza Modello", command=self.load_all_data).grid(row=3, column=0, columnspan=3, pady=10)
        
        info_frame = ttk.LabelFrame(tab, text="Informazioni Modello", padding=10)
        info_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.model_info_label = ttk.Label(info_frame, text="Nessun modello analizzato.", justify=tk.LEFT, style="Status.TLabel")
        self.model_info_label.pack(fill=tk.X)
        
        optional_frame = ttk.LabelFrame(tab, text="File Opzionali", padding=10)
        optional_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        optional_frame.columnconfigure(1, weight=1)
        
        self.create_file_selector(optional_frame, "Vocabolario Frequenze (GUI):", self.meta_freq_vocab_path, self.browse_meta_freq_vocab, 0)
        self.create_file_selector(optional_frame, "Primer MIDI:", self.primer_midi_path, self.browse_primer_midi, 1)
        self.create_file_selector(optional_frame, "File Profili Consigliati:", self.profiles_path, self.browse_profiles, 2)
        self.create_file_selector(optional_frame, "Cartella di Output:", self.output_dir, self.browse_output_dir, 3, is_dir=True)

    def populate_generation_tab(self, tab):
        """Popola la scheda di generazione con modalità e parametri."""
        tab.columnconfigure(0, weight=1, minsize=400)
        tab.columnconfigure(1, weight=1, minsize=400)
        
        # --- COLONNA SINISTRA: MODALITA' ---
        left_col = ttk.Frame(tab)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        mode_frame = ttk.LabelFrame(left_col, text="Modalità di Generazione", padding=10)
        mode_frame.pack(fill=tk.X, expand=True, pady=5)
        
        ttk.Radiobutton(mode_frame, text="Genere", variable=self.generation_mode, value="Genere", command=self.toggle_ui_mode).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="Guidata", variable=self.generation_mode, value="Guidata", command=self.toggle_ui_mode).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="Manuale", variable=self.generation_mode, value="Manuale", command=self.toggle_ui_mode).pack(side=tk.LEFT, padx=10)

        # --- Frame per ogni modalità ---
        # Genere
        self.genre_frame = self.create_genre_frame(left_col)
        self.genre_frame.pack(fill=tk.X, expand=True, pady=5)
        # Guidata
        self.profile_frame = self.create_guided_frame(left_col)
        self.profile_frame.pack(fill=tk.X, expand=True, pady=5)
        # Manuale
        self.manual_frame = self.create_manual_frame(left_col)
        self.manual_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # --- COLONNA DESTRA: PARAMETRI ---
        right_col = ttk.Frame(tab)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        params_frame = ttk.LabelFrame(right_col, text="Parametri di Generazione", padding=10)
        params_frame.pack(fill=tk.X, expand=True, pady=5)
        params_frame.columnconfigure(1, weight=1)

        self.create_param_entry(params_frame, "Lunghezza base (token):", self.total_tokens_var, 0)
        self.create_param_entry(params_frame, "Temperatura:", self.temperature_var, 1)
        self.create_param_entry(params_frame, "Top-K:", self.top_k_var, 2)
        ttk.Label(params_frame, text="Max Rest Penalty:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        penalty_slider_frame = ttk.Frame(params_frame)
        penalty_slider_frame.grid(row=3, column=1, sticky="ew", padx=5)
        penalty_slider_frame.columnconfigure(0, weight=1)

        penalty_slider = ttk.Scale(
            penalty_slider_frame,
            from_=0.0,
            to=15.0,  # Il nostro valore massimo ragonevole
            orient=tk.HORIZONTAL,
            variable=self.max_rest_penalty_var
        )
        penalty_slider.grid(row=0, column=0, sticky="ew")

        # Etichetta per mostrare il valore corrente dello slider
        penalty_value_label = ttk.Label(penalty_slider_frame, font=('Segoe UI', 9, 'bold'))
        penalty_value_label.grid(row=0, column=1, padx=(5, 0))

        # Funzione per aggiornare l'etichetta con il valore formattato
        def update_penalty_label(*args):
            penalty_value_label.config(text=f"{self.max_rest_penalty_var.get():.1f}")

        # Lega l'aggiornamento al cambiamento della variabile e imposta il valore iniziale
        self.max_rest_penalty_var.trace_add('write', update_penalty_label)
        update_penalty_label() # Chiamata iniziale

        # Sposta il frame della modalità di penalità alla riga successiva
        penalty_frame = ttk.Frame(params_frame)
        penalty_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew") # row era 4, rimane 4
        ttk.Radiobutton(penalty_frame, text="Ibrida", variable=self.rest_penalty_mode, value='hybrid').pack(side=tk.LEFT, padx=5, expand=True)
        ttk.Radiobutton(penalty_frame, text="Costante", variable=self.rest_penalty_mode, value='constant').pack(side=tk.LEFT, padx=5, expand=True)
        ttk.Radiobutton(penalty_frame, text="Progressiva", variable=self.rest_penalty_mode, value='ramped').pack(side=tk.LEFT, padx=5, expand=True)
        
        ttk.Label(params_frame, text="Processi CPU Paralleli:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.workers_entry = ttk.Entry(params_frame, textvariable=self.num_workers_var, width=10)
        self.workers_entry.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Separator(params_frame, orient='horizontal').grid(row=6, column=0, columnspan=2, sticky='ew', pady=10)
        force_piano_check = ttk.Checkbutton(params_frame, text="Forza solo Famiglia Pianoforte", variable=self.force_piano_only_var)
        force_piano_check.grid(row=7, column=0, columnspan=2, sticky="w", padx=5, pady=5)

    def populate_cleaning_tab(self, tab):
        """Popola la scheda con le opzioni di pulizia MIDI in modo corretto e robusto."""
        tab.columnconfigure(0, weight=1)
        
        frame = ttk.LabelFrame(tab, text="Opzioni di Pulizia e Miglioramento MIDI", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # Configura le colonne per il layout a griglia
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        # Helper per abilitare/disabilitare i widget collegati a una checkbox
        def link_checkbox_to_widgets(checkbox_var, widgets_to_toggle):
            def toggle_state(*args):
                state = tk.NORMAL if checkbox_var.get() else tk.DISABLED
                for widget in widgets_to_toggle:
                    widget.config(state=state)
            
            checkbox_var.trace_add('write', toggle_state)
            toggle_state() # Imposta lo stato iniziale

        # --- RIGA 1 ---
        # Opzione: Rimuovi Silenzio Iniziale
        trim_container = ttk.Frame(frame)
        trim_container.grid(row=0, column=0, sticky="w", pady=5)
        ttk.Checkbutton(trim_container, variable=self.clean_trim_silence_var, text="Rimuovi Silenzio Iniziale").pack(side=tk.LEFT)
        ttk.Label(trim_container, text="(Elimina le pause all'inizio)", style="Desc.TLabel").pack(side=tk.LEFT, padx=10)

        # Opzione: Quantizza
        quantize_container = ttk.Frame(frame)
        quantize_container.grid(row=0, column=1, sticky="w", pady=5)
        ttk.Checkbutton(quantize_container, variable=self.clean_quantize_var, text="Quantizza Note").pack(side=tk.LEFT)
        q_subframe = ttk.Frame(quantize_container)
        q_subframe.pack(side=tk.LEFT, padx=10)
        q_label = ttk.Label(q_subframe, text="Griglia:")
        q_label.pack(side=tk.LEFT)
        q_combo = ttk.Combobox(q_subframe, textvariable=self.clean_quantize_grid_var, values=["1/4", "1/8", "1/16", "1/32"], width=6)
        q_combo.pack(side=tk.LEFT)
        link_checkbox_to_widgets(self.clean_quantize_var, [q_label, q_combo])

        # --- RIGA 2 ---
        # Opzione: Rimuovi Pedale Sustain
        sustain_container = ttk.Frame(frame)
        sustain_container.grid(row=1, column=0, sticky="w", pady=5)
        ttk.Checkbutton(sustain_container, variable=self.clean_remove_sustain_var, text="Rimuovi Pedale Sustain").pack(side=tk.LEFT)
        ttk.Label(sustain_container, text="(Rimuove eventi CC64)", style="Desc.TLabel").pack(side=tk.LEFT, padx=10)

        # Opzione: Limita Polifonia
        poly_container = ttk.Frame(frame)
        poly_container.grid(row=1, column=1, sticky="w", pady=5)
        ttk.Checkbutton(poly_container, variable=self.clean_limit_polyphony_var, text="Limita Polifonia").pack(side=tk.LEFT)
        p_subframe = ttk.Frame(poly_container)
        p_subframe.pack(side=tk.LEFT, padx=10)
        p_label = ttk.Label(p_subframe, text="Max Note:")
        p_label.pack(side=tk.LEFT)
        p_entry = ttk.Entry(p_subframe, textvariable=self.clean_polyphony_max_var, width=5)
        p_entry.pack(side=tk.LEFT)
        link_checkbox_to_widgets(self.clean_limit_polyphony_var, [p_label, p_entry])

        # --- RIGA 3 ---
        # Opzione: Fondi Tracce Simili
        merge_container = ttk.Frame(frame)
        merge_container.grid(row=2, column=0, sticky="w", pady=5)
        ttk.Checkbutton(merge_container, variable=self.clean_merge_tracks_var, text="Fondi Tracce Simili").pack(side=tk.LEFT)
        ttk.Label(merge_container, text="(Raggruppa strumenti per famiglia)", style="Desc.TLabel").pack(side=tk.LEFT, padx=10)

        # Opzione: Filtra Altezze Note
        pitch_container = ttk.Frame(frame)
        pitch_container.grid(row=2, column=1, sticky="w", pady=5)
        ttk.Checkbutton(pitch_container, variable=self.clean_filter_pitch_var, text="Filtra Altezze Note").pack(side=tk.LEFT)
        pi_subframe = ttk.Frame(pitch_container)
        pi_subframe.pack(side=tk.LEFT, padx=10)
        pi_label1 = ttk.Label(pi_subframe, text="Range:")
        pi_label1.pack(side=tk.LEFT)
        pi_entry1 = ttk.Entry(pi_subframe, textvariable=self.clean_pitch_min_var, width=4)
        pi_entry1.pack(side=tk.LEFT)
        pi_label2 = ttk.Label(pi_subframe, text="-")
        pi_label2.pack(side=tk.LEFT)
        pi_entry2 = ttk.Entry(pi_subframe, textvariable=self.clean_pitch_max_var, width=4)
        pi_entry2.pack(side=tk.LEFT)
        link_checkbox_to_widgets(self.clean_filter_pitch_var, [pi_label1, pi_entry1, pi_label2, pi_entry2])

        # --- RIGA 4 ---
        # Opzione: Normalizza Velocity
        vel_container = ttk.Frame(frame)
        vel_container.grid(row=3, column=1, sticky="w", pady=5)
        ttk.Checkbutton(vel_container, variable=self.clean_filter_velocity_var, text="Normalizza Velocity").pack(side=tk.LEFT)
        v_subframe = ttk.Frame(vel_container)
        v_subframe.pack(side=tk.LEFT, padx=10)
        v_label1 = ttk.Label(v_subframe, text="Range:")
        v_label1.pack(side=tk.LEFT)
        v_entry1 = ttk.Entry(v_subframe, textvariable=self.clean_velocity_min_var, width=4)
        v_entry1.pack(side=tk.LEFT)
        v_label2 = ttk.Label(v_subframe, text="-")
        v_label2.pack(side=tk.LEFT)
        v_entry2 = ttk.Entry(v_subframe, textvariable=self.clean_velocity_max_var, width=4)
        v_entry2.pack(side=tk.LEFT)
        link_checkbox_to_widgets(self.clean_filter_velocity_var, [v_label1, v_entry1, v_label2, v_entry2])

    def populate_queue_tab(self, tab):
        """Popola la scheda della coda di lavori."""
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        
        queue_frame = ttk.LabelFrame(tab, text="Coda di Lavori", padding=10)
        queue_frame.grid(row=0, column=0, sticky="nsew", pady=5)
        queue_frame.rowconfigure(0, weight=1)
        queue_frame.columnconfigure(0, weight=1)

        self.jobs_listbox = Listbox(queue_frame, height=10)
        self.jobs_listbox.grid(row=0, column=0, sticky="nsew")
        jobs_scrollbar = ttk.Scrollbar(queue_frame, orient="vertical", command=self.jobs_listbox.yview)
        jobs_scrollbar.grid(row=0, column=1, sticky="ns")
        self.jobs_listbox.config(yscrollcommand=jobs_scrollbar.set)
        
        button_frame = ttk.Frame(tab)
        button_frame.grid(row=1, column=0, pady=10, sticky="ew")
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        self.add_job_button = ttk.Button(button_frame, text="Aggiungi Lavoro alla Coda", command=self.add_job_to_queue, style="Accent.TButton")
        self.add_job_button.grid(row=0, column=0, padx=5, sticky="ew")
        
        self.start_queue_button = ttk.Button(button_frame, text="Avvia Esecuzione Coda", command=self.start_processing_queue)
        self.start_queue_button.grid(row=0, column=1, padx=5, sticky="ew")

        status_frame = ttk.LabelFrame(tab, text="Stato e Progresso", padding=10)
        status_frame.grid(row=2, column=0, sticky="ew", pady=5)
        status_frame.columnconfigure(0, weight=1)
        
        self.progress_bar = ttk.Progressbar(status_frame, orient='horizontal', length=400, mode='determinate')
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=5)

        self.status_label = ttk.Label(status_frame, text="Pronto.", wraplength=750, justify=tk.LEFT, style="Status.TLabel")
        self.status_label.grid(row=1, column=0, sticky="ew", pady=5)
    
    # --- Funzioni Helper per la creazione di widget ---

    def create_file_selector(self, frame, label_text, string_var, command, row, is_dir=False, has_analyze_button=False):
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        entry = ttk.Label(frame, textvariable=string_var, style="Path.TLabel", anchor="w")
        entry.grid(row=row, column=1, sticky="ew", padx=5)
        
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=row, column=2, sticky="e", padx=5)
        ttk.Button(button_frame, text="Sfoglia...", command=lambda: command(is_dir)).pack(side=tk.LEFT)
        if has_analyze_button:
            ttk.Button(button_frame, text="Analizza", command=self.analyze_model).pack(side=tk.LEFT, padx=(5,0))
            
    def create_param_entry(self, frame, text, var, row):
        ttk.Label(frame, text=text).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frame, textvariable=var, width=12).grid(row=row, column=1, sticky="w", padx=5)

    # --- Funzioni Helper per i Frame delle Modalità di Generazione ---

    def create_genre_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Modalità Genere", padding=10)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="Genere:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.genre_combobox = ttk.Combobox(frame, textvariable=self.selected_genre, state="readonly")
        self.genre_combobox.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.genre_combobox['values'] = ["Caricare vocabolari..."]
        self.genre_combobox.set(self.genre_combobox['values'][0])
        ttk.Label(frame, text="N° Strumenti (stima):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.num_instruments_genre_mode = tk.StringVar(value="3")
        ttk.Entry(frame, textvariable=self.num_instruments_genre_mode, width=10).grid(row=1, column=1, sticky="w", padx=5)
        return frame

    def create_guided_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Modalità Guidata (Profili)", padding=10)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="Profilo:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.profile_combobox = ttk.Combobox(frame, textvariable=self.selected_profile, state="readonly")
        self.profile_combobox.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.profile_combobox['values'] = ["Caricare file profili..."]
        self.profile_combobox.set(self.profile_combobox['values'][0])
        self.profile_combobox.bind("<<ComboboxSelected>>", self.on_profile_select)
        return frame

    def create_manual_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Modalità Manuale", padding=10)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        
        # Parametri
        params_subframe = ttk.Frame(frame)
        params_subframe.grid(row=0, column=0, sticky="nswe", padx=(0, 5))
        params_subframe.columnconfigure(1, weight=1)
        self.manual_params_frame = params_subframe # For toggle_ui_mode
        self.create_combobox(params_subframe, "Tonalità:", "Key", 0)
        self.create_combobox(params_subframe, "Tempo:", "Tempo", 1)
        self.create_combobox(params_subframe, "Dinamica:", "AvgVel", 2)
        self.create_combobox(params_subframe, "Range Din.:", "VelRange", 3)
        self.create_combobox(params_subframe, "Metro:", "TimeSig", 4)
        ttk.Button(params_subframe, text="Seleziona Casuali", command=self.randomize_metadata).grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")

        # Strumenti
        inst_subframe = ttk.Frame(frame)
        inst_subframe.grid(row=0, column=1, sticky="nswe", padx=(5, 0))
        inst_subframe.rowconfigure(1, weight=1)
        inst_subframe.columnconfigure(0, weight=1)
        self.inst_frame = inst_subframe # For toggle_ui_mode
        
        random_inst_check = ttk.Checkbutton(inst_subframe, text="Scegli strumenti casualmente", variable=self.random_instruments_var, command=self.toggle_instrument_list_state)
        random_inst_check.grid(row=0, column=0, columnspan=2, sticky="w")
        
        self.instrument_listbox = Listbox(inst_subframe, selectmode=MULTIPLE, height=8, exportselection=False)
        self.instrument_listbox.grid(row=1, column=0, sticky="nsew")
        inst_scrollbar = ttk.Scrollbar(inst_subframe, orient=tk.VERTICAL, command=self.instrument_listbox.yview)
        inst_scrollbar.grid(row=1, column=1, sticky="ns")
        self.instrument_listbox.config(yscrollcommand=inst_scrollbar.set)
        self.instrument_listbox.insert(END, "Caricare vocabolari...")
        
        return frame

    def create_combobox(self, frame, label_text, category_key, row):
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.control_vars[category_key] = tk.StringVar()
        combo = ttk.Combobox(frame, textvariable=self.control_vars[category_key], state="readonly")
        combo.grid(row=row, column=1, sticky="ew", padx=5)
        combo['values'] = ["..."]
        combo.set(combo['values'][0])
        self.combobox_widgets[category_key] = combo

    # --- Logica di Funzionamento (la maggior parte è invariata, solo i riferimenti ai widget cambiano) ---
    
    def on_closing(self):
        if self.executor:
            messagebox.showinfo("Attenzione", "Chiusura dei processi di generazione in corso... L'applicazione si chiuderà a breve.")
            self.executor.shutdown(wait=True, cancel_futures=True)
        if self.manager:
            self.manager.shutdown()
        self.root.destroy()
        
    def browse_file(self, string_var, file_category, is_dir=False):
        if is_dir:
            path = filedialog.askdirectory(title="Seleziona una cartella")
        else:
            filetypes = [("Tutti i file", "*.*")]
            if file_category == "model":
                filetypes.insert(0, ("File Modello PyTorch", "*.pt"))
            elif file_category == "vocab":
                filetypes.insert(0, ("File JSON", "*.json"))
            elif file_category == "primer":
                filetypes.insert(0, ("File MIDI", "*.mid;*.midi"))
            path = filedialog.askopenfilename(title="Seleziona un file", filetypes=filetypes)
        if path:
            string_var.set(path)
        pass
            
    def browse_model(self, is_dir=False): self.browse_file(self.model_path, "model", is_dir)
    def browse_midi_vocab(self, is_dir=False): self.browse_file(self.midi_vocab_path, "vocab", is_dir)
    def browse_meta_vocab(self, is_dir=False): self.browse_file(self.meta_vocab_path, "vocab", is_dir)
    def browse_meta_freq_vocab(self, is_dir=False): self.browse_file(self.meta_freq_vocab_path, "vocab", is_dir)
    def browse_output_dir(self, is_dir=True): self.browse_file(self.output_dir, "directory", is_dir)
    def browse_primer_midi(self, is_dir=False): self.browse_file(self.primer_midi_path, "primer", is_dir)
    def browse_profiles(self, is_dir=False): self.browse_file(self.profiles_path, "vocab", is_dir)

    def load_all_data(self):
        self.load_and_populate_metadata_options() 
        self.load_profiles()

    def load_profiles(self):
        profiles_file = self.profiles_path.get()
        if not profiles_file or not Path(profiles_file).exists():
            messagebox.showwarning("Attenzione Profili", "File dei profili non selezionato o non trovato. La modalità guidata non sarà disponibile.")
            self.profile_combobox['values'] = ["File profili non caricato"]
            self.profile_combobox.set(self.profile_combobox['values'][0])
            self.profiles = []
            self.toggle_ui_mode()
            return
        try:
            with open(profiles_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.profiles = data.get("profiles", [])
            
            if not self.profiles:
                messagebox.showerror("Errore Profili", "Nessun profilo trovato nel file.")
                return

            profile_names = [p['profile_name'] for p in self.profiles]
            self.profile_combobox['values'] = profile_names
            if profile_names:
                self.profile_combobox.set(profile_names[0])
                self.on_profile_select()
            
            self.update_status(f"Caricati {len(self.profiles)} profili con successo.")
        except Exception as e:
            messagebox.showerror("Errore Caricamento Profili", f"Impossibile leggere il file:\n{e}")
            self.profiles = []
        finally:
            self.toggle_ui_mode()

    # --- Gestione della visibilità dei frame per le 3 modalità ---
    def toggle_ui_mode(self):
        mode = self.generation_mode.get()
        is_genre = mode == "Genere"
        is_guided = mode == "Guidata"
        is_manual = mode == "Manuale"

        def set_state(widget, state):
            try:
                if isinstance(widget, (ttk.Combobox, ttk.Entry, ttk.Button, ttk.Checkbutton, ttk.Radiobutton, Listbox)):
                    widget.config(state=state)
                for child in widget.winfo_children():
                    set_state(child, state)
            except tk.TclError:
                pass # Ignora errori se il widget non supporta lo stato

        set_state(self.genre_frame, tk.NORMAL if is_genre else tk.DISABLED)
        set_state(self.profile_frame, tk.NORMAL if is_guided else tk.DISABLED)
        set_state(self.manual_frame, tk.NORMAL if is_manual else tk.DISABLED)
        self.toggle_instrument_list_state()
    
    def on_profile_select(self, event=None):
        profile_name = self.selected_profile.get()
        profile_data = next((p for p in self.profiles if p['profile_name'] == profile_name), None)
        if not profile_data: return

        key_map = {
            "recommended_key": "Key", "recommended_timesig": "TimeSig",
            "recommended_tempo": "Tempo", "recommended_avg_vel": "AvgVel",
            "recommended_vel_range": "VelRange"
        }
        for p_key, v_key in key_map.items():
            value = profile_data.get(p_key)
            if value and v_key in self.control_vars and self.combobox_widgets[v_key]['values']:
                if value in self.combobox_widgets[v_key]['values']:
                    self.control_vars[v_key].set(value)
                else:
                    self.control_vars[v_key].set(self.combobox_widgets[v_key]['values'][0])
        
        if hasattr(self, 'instrument_listbox'):
            self.instrument_listbox.selection_clear(0, END)
            listbox_items = list(self.instrument_listbox.get(0, END))
            for inst_token in profile_data.get("instruments", []):
                try:
                    idx = listbox_items.index(inst_token)
                    self.instrument_listbox.selection_set(idx)
                except (ValueError, tk.TclError): pass
        try:
            self.update_progress(0)
            paths = [self.model_path.get(), self.midi_vocab_path.get(), self.meta_vocab_path.get(), self.output_dir.get()]
            if not all(paths): raise ValueError("Tutti i percorsi (modello, vocabolari, output) devono essere specificati.")

            prompt, num_inst_len = [], 0
            mode = self.generation_mode.get()

            if mode == "Genere":
                self.update_status("Modalità Genere: costruzione prompt...")
                genre_token = self.selected_genre.get()
                if not genre_token or "Caricare" in genre_token or "trovato" in genre_token:
                    raise ValueError("Selezionare un genere valido dalla lista.")
                
                prompt = [genre_token]
                try:
                    num_inst_len = int(self.num_instruments_genre_mode.get())
                    if num_inst_len <= 0: raise ValueError()
                except ValueError:
                    raise ValueError("Il numero di strumenti stimato deve essere un intero positivo.")

            elif mode == "Guidata":
                self.update_status("Modalità Guidata: costruzione prompt dal profilo selezionato...")
                profile_name = self.selected_profile.get()
                if not profile_name or "Caricare" in profile_name or "non caricato" in profile_name:
                    raise ValueError("Selezionare un profilo valido dalla lista.")
                
                selected_profile_data = next((p for p in self.profiles if p['profile_name'] == profile_name), None)
                if not selected_profile_data: raise ValueError(f"Dati per il profilo '{profile_name}' non trovati.")

                prompt.extend(selected_profile_data.get("instruments", []))
                recommended_tokens = [
                    selected_profile_data.get("recommended_key"), selected_profile_data.get("recommended_timesig"),
                    selected_profile_data.get("recommended_tempo"), selected_profile_data.get("recommended_avg_vel"),
                    selected_profile_data.get("recommended_vel_range"), selected_profile_data.get("recommended_num_inst")]
                prompt.extend([token for token in recommended_tokens if token])
                num_inst_len = len(selected_profile_data.get("instruments", []))
            
            else: # Modalità Manuale
                self.update_status("Modalità Manuale: costruzione prompt dai controlli...")
                prompt = [var.get() for var in self.control_vars.values() if var.get() and "Caricare" not in var.get() and "Nessuno" not in var.get()]
                
                selected_instruments = []
                if self.random_instruments_var.get():
                    all_instrument_options = self.metadata_options.get("Instrument", [])
                    if not all_instrument_options: raise ValueError("Nessuno strumento disponibile per la selezione casuale.")
                    num_to_select = random.randint(1, min(3, len(all_instrument_options)))
                    selected_instruments = random.sample(all_instrument_options, num_to_select)
                else:
                    selected_instruments = [self.instrument_listbox.get(i) for i in self.instrument_listbox.curselection()]
                
                if not selected_instruments: raise ValueError("Selezionare almeno uno strumento o spuntare 'Scegli casualmente'.")
                
                num_inst_len = len(selected_instruments)
                num_map = {1: "NumInst_Solo", 2: "NumInst_Duet"}
                num_token = num_map.get(num_inst_len)
                if not num_token:
                    if 2 < num_inst_len <= 4: num_token = "NumInst_SmallChamber"
                    elif 4 < num_inst_len <= 8: num_token = "NumInst_MediumEnsemble"
                    else: num_token = "NumInst_LargeEnsemble"
                if num_token and num_token in self.metadata_options.get("NumInst", []): prompt.append(num_token)
                prompt.extend(selected_instruments)

            if num_inst_len == 0: raise ValueError("Il prompt non contiene strumenti o una stima valida.")

            base_tokens = int(self.total_tokens_var.get())
            final_tokens = base_tokens * num_inst_len
            self.update_status(f"Budget token: {base_tokens} x {num_inst_len} strumenti = {final_tokens} token totali.")

            final_message = run_generation(
                model_path=self.model_path.get(), midi_vocab_path=self.midi_vocab_path.get(),
                meta_vocab_path=self.meta_vocab_path.get(), metadata_prompt=prompt,
                output_dir=self.output_dir.get(), total_tokens=final_tokens,
                temperature=float(self.temperature_var.get()), top_k=int(self.top_k_var.get()),
                max_rest_penalty=float(self.max_rest_penalty_var.get()), primer_midi_path=self.primer_midi_path.get(),
                update_status_callback=self.update_status, progress_callback=self.update_progress)
            
            messagebox.showinfo("Generazione Completata", final_message)
        except (ValueError, RuntimeError) as e:
            self.update_status(f"Errore: {e}")
            messagebox.showerror("Errore di Configurazione", str(e))
        except Exception as e:
            self.update_status(f"Errore imprevisto: {e}")
            logging.error("Errore durante la generazione", exc_info=True)
            messagebox.showerror("Errore Imprevisto", f"Si è verificato un errore:\n{e}")
        finally:
            self.update_progress(0)
            self.root.after(0, self.generate_button.config, {'state': 'normal'})
    
    def update_progress(self, value):
        self.root.after(0, self.progress_bar.config, {'value': value})

    # --- Caricamento e popolamento delle opzioni di generazione ---
    def load_and_populate_metadata_options(self):
        meta_vocab_file = self.meta_vocab_path.get()
        meta_freq_file = self.meta_freq_vocab_path.get()
        if not meta_vocab_file or not Path(meta_vocab_file).exists():
            messagebox.showerror("Errore", "Selezionare un file di vocabolario metadati valido.")
            return
        freq_counts = {}
        if meta_freq_file and Path(meta_freq_file).exists():
            try:
                with open(meta_freq_file, 'r', encoding='utf-8') as f: freq_counts = json.load(f).get('metadata_token_counts', {})
            except Exception as e: messagebox.showerror("Errore Frequenze", f"Impossibile leggere file frequenze:\n{e}")
        else:
            messagebox.showwarning("Attenzione", "File vocabolario frequenze non trovato.")
        try:
            with open(meta_vocab_file, 'r', encoding='utf-8') as f: token_to_id = json.load(f).get('token_to_id', {})
            all_tokens = list(token_to_id.keys())
            def sort_key(token): return freq_counts.get(token, -1)
            
            self.metadata_options = {
                "Key": sorted([t for t in all_tokens if t.startswith("Key=")], key=sort_key, reverse=True),
                "TimeSig": sorted([t for t in all_tokens if t.startswith("TimeSig=")], key=sort_key, reverse=True),
                "Tempo": sorted([t for t in all_tokens if t.startswith("Tempo_")], key=sort_key, reverse=True),
                "AvgVel": sorted([t for t in all_tokens if t.startswith("AvgVel_")], key=sort_key, reverse=True),
                "VelRange": sorted([t for t in all_tokens if t.startswith("VelRange_")], key=sort_key, reverse=True),
                "Instrument": sorted([t for t in all_tokens if t.startswith("Instrument=")], key=sort_key, reverse=True),
                "NumInst": sorted([t for t in all_tokens if t.startswith("NumInst_")], key=sort_key, reverse=True),
                # NUOVA CHIAVE PER I GENERI
                "Genre": sorted([t for t in all_tokens if t.startswith("Genre=")], key=sort_key, reverse=True)
            }
            
            for cat_key, combo in self.combobox_widgets.items():
                values = self.metadata_options.get(cat_key, ["Nessuno trovato"])
                combo['values'] = values if values else ["Nessuno trovato"]
                if values: combo.set(values[0])
            
            # Popola la combobox dei generi
            genre_values = self.metadata_options.get("Genre", [])
            self.genre_combobox['values'] = genre_values if genre_values else ["Nessun genere trovato"]
            if genre_values: self.genre_combobox.set(genre_values[0])

            self.instrument_listbox.delete(0, END)
            instrument_values = self.metadata_options.get("Instrument", [])
            if instrument_values: [self.instrument_listbox.insert(END, item) for item in instrument_values]
            else: self.instrument_listbox.insert(END, "Nessuno strumento trovato")
            
            self.update_status("Opzioni dei metadati caricate con successo.")
        except Exception as e: messagebox.showerror("Errore Vocabolario", f"Impossibile leggere file:\n{e}")

    def toggle_instrument_list_state(self):
        if self.random_instruments_var.get():
            self.instrument_listbox.config(state=tk.DISABLED)
            self.instrument_listbox.selection_clear(0, END)
        else:
            self.instrument_listbox.config(state=tk.NORMAL)

    def randomize_metadata(self):
        if self.generation_mode.get() != "Manuale":
            messagebox.showinfo("Info", "La selezione casuale è disponibile solo in Modalità Manuale.")
            return
        if not self.metadata_options:
            messagebox.showerror("Errore", "Per favore, carica prima un vocabolario di metadati.")
            return
        for cat_key, var in self.control_vars.items():
            options = self.metadata_options.get(cat_key, [])
            if options: var.set(random.choice(options))
        self.instrument_listbox.selection_clear(0, END)
        all_instruments = self.metadata_options.get("Instrument", [])
        if all_instruments:
            num_to_select = random.randint(1, min(5, len(all_instruments)))
            selected_instruments = random.sample(all_instruments, num_to_select)
            listbox_items = list(self.instrument_listbox.get(0, END))
            for name in selected_instruments:
                try: self.instrument_listbox.selection_set(listbox_items.index(name))
                except (ValueError, tk.TclError): pass
        self.update_status("Metadati e strumenti selezionati casualmente.")

    def start_generation_thread(self):
        self.generate_button.config(state=tk.DISABLED)
        self.update_progress(0)
        self.status_label.config(text="Avvio della generazione in un thread separato...")
        threading.Thread(target=self.generate_music, daemon=True).start()

    def update_status(self, message):
        self.root.after(0, self.status_label.config, {'text': message})

    def analyze_model(self):
        model_path = self.model_path.get()
        if not model_path:
            messagebox.showerror("Errore", "Per favore, prima seleziona un file modello (.pt).")
            return
        self.status_label.config(text="Analisi del modello in corso... Attendere.")
        self.root.update_idletasks()
        info_dict = get_model_info(model_path)
        if "error" in info_dict:
            messagebox.showerror("Errore Analisi Modello", info_dict["error"])
            self.model_info_label.config(text="Analisi fallita.")
            self.status_label.config(text="Errore.")
        else:
            info_text = "\n".join([f"{key}: {value}" for key, value in info_dict.items()])
            self.model_info_label.config(text=info_text)
            self.status_label.config(text="Informazioni modello caricate.")
    
    # --- Logica di GESTIONE CODA e GENERAZIONE ---
    def check_progress_queue(self):
        """Controlla la coda per aggiornamenti dai processi figli e aggiorna la GUI."""
        try:
            while not self.progress_queue.empty():
                msg_type, value, job_id = self.progress_queue.get_nowait()
                
                if msg_type == "status":
                    # Mostra sempre i messaggi di stato, indipendentemente dalla selezione
                    self.update_status(f"[{job_id}] {value}")
                
                elif msg_type == "progress":
                    # Trova l'indice del lavoro che ha inviato l'aggiornamento
                    listbox_index = -1
                    for i in range(self.jobs_listbox.size()):
                        if self.jobs_listbox.get(i).startswith(f"ID: {job_id}"):
                            listbox_index = i
                            break
                    
                    # Se il lavoro è visibile nella lista, procedi
                    if listbox_index != -1:
                        # Ottieni l'indice dell'elemento attualmente selezionato dall'utente
                        selected_indices = self.jobs_listbox.curselection()

                        # Aggiorna la barra di progresso SOLO SE
                        # 1. C'è un elemento selezionato (selected_indices non è vuoto)
                        # 2. L'indice dell'elemento selezionato è lo stesso del lavoro che ha inviato l'update
                        if selected_indices and selected_indices[0] == listbox_index:
                            self.update_progress(value)
        except Empty:
            pass # La coda è vuota, non fare nulla
        except Exception as e:
            # Aggiungiamo un log per vedere eventuali altri errori imprevisti
            logging.error(f"Errore in check_progress_queue: {e}")
        finally:
            # Richiama questa funzione dopo 100ms
            self.root.after(100, self.check_progress_queue)

    def add_job_to_queue(self):
        try:
            prompt, _ = self.build_prompt() # Ora usiamo solo il prompt da qui
            
            # *** NUOVA LOGICA INTELLIGENTE PER I VINCOLI SUGLI STRUMENTI ***
            allowed_programs = None # Default: nessun vincolo
            # Se la checkbox è attiva, applichiamo i vincoli
            if self.force_piano_only_var.get():
                allowed_programs = [] # Inizializza come lista vuota (significa: vincola a 0-7 di default)
                piano_family_range = range(0, 8)
                
                # Controlla se siamo in modalità manuale e se l'utente ha selezionato strumenti specifici
                if self.generation_mode.get() == "Manuale" and not self.random_instruments_var.get():
                    selected_instrument_tokens = [self.instrument_listbox.get(i) for i in self.instrument_listbox.curselection()]
                    
                    if selected_instrument_tokens:
                        # Se ci sono selezioni, popola la lista 'allowed_programs' solo con quelle
                        for token in selected_instrument_tokens:
                            try:
                                # Estrae il numero dall'instrument token (es. "Instrument=4")
                                prog_num = int(token.split('=')[-1])
                                # Aggiungilo solo se è un pianoforte, per sicurezza
                                if prog_num in piano_family_range:
                                    if prog_num not in allowed_programs:
                                        allowed_programs.append(prog_num)
                            except (ValueError, IndexError):
                                continue
            # A questo punto, 'allowed_programs' è:
            # - None: nessun vincolo
            # - []: vincolo alla famiglia 0-7
            # - [4, 6]: vincolo solo ai programmi 4 e 6

            # Ricalcola il numero di strumenti e i token totali
            num_inst_len = 0
            if allowed_programs is not None:
                # Se il vincolo è attivo, il numero di strumenti è quello dei programmi permessi
                # Se la lista è vuota, il modello può scegliere tra tutti gli 8 pianoforti
                num_inst_len = len(allowed_programs) if allowed_programs else random.randint(1, 3) 
            else:
                # Altrimenti, calcola dagli strumenti nel prompt
                num_inst_len = len([p for p in prompt if p.startswith("Instrument=")])
            
            if num_inst_len == 0: num_inst_len = 1 # Fallback a 1 per evitare token a 0

            base_tokens = int(self.total_tokens_var.get())
            final_tokens = base_tokens * num_inst_len

            self.job_counter += 1
            job_id = f"Job-{self.job_counter}"
            
            cleaning_options = { # ... (le opzioni di pulizia rimangono invariate) ...
                "remove_sustain": self.clean_remove_sustain_var.get(), "merge_tracks": self.clean_merge_tracks_var.get(),
                "trim_silence": self.clean_trim_silence_var.get(), "quantize": self.clean_quantize_var.get(),
                "quantize_grid": self.clean_quantize_grid_var.get(), "limit_polyphony": self.clean_limit_polyphony_var.get(),
                "polyphony_max": int(self.clean_polyphony_max_var.get()), "filter_pitch": self.clean_filter_pitch_var.get(),
                "pitch_min": int(self.clean_pitch_min_var.get()), "pitch_max": int(self.clean_pitch_max_var.get()),
                "filter_velocity": self.clean_filter_velocity_var.get(), "velocity_min": int(self.clean_velocity_min_var.get()),
                "velocity_max": int(self.clean_velocity_max_var.get()),
            }
            
            job_params = {
                "job_id": job_id,
                "model_path": self.model_path.get(),
                "midi_vocab_path": self.midi_vocab_path.get(),
                "meta_vocab_path": self.meta_vocab_path.get(),
                "metadata_prompt": prompt,
                "output_dir": self.output_dir.get(),
                "total_tokens": final_tokens,
                "temperature": float(self.temperature_var.get()),
                "top_k": int(self.top_k_var.get()),
                "max_rest_penalty": float(self.max_rest_penalty_var.get()),
                "rest_penalty_mode": self.rest_penalty_mode.get(),
                "primer_midi_path": self.primer_midi_path.get(),
                "cleaning_options": cleaning_options,
                "force_piano_only": self.force_piano_only_var.get(),
                "allowed_programs": allowed_programs,
                "status": "In attesa"
            }
            
            self.jobs_list.append(job_params)
            self.jobs_listbox.insert(END, f"ID: {job_id} | Prompt: {'_'.join(prompt)[:50]}... | Stato: In attesa")
            self.update_status(f"Lavoro '{job_id}' aggiunto alla coda.")

        except (ValueError, RuntimeError) as e:
            messagebox.showerror("Errore di Configurazione", str(e))
        except Exception as e:
            logging.error("Errore durante la creazione del lavoro", exc_info=True)
            messagebox.showerror("Errore Imprevisto", f"Si è verificato un errore durante la configurazione del lavoro:\n{e}")

    def start_processing_queue(self):
        """Avvia l'esecuzione dei lavori in coda, passando anche le opzioni di pulizia."""
        if not self.jobs_list:
            messagebox.showinfo("Info", "La coda di lavori è vuota.")
            return

        use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
        
        self.add_job_button.config(state=tk.DISABLED)
        self.start_queue_button.config(state=tk.DISABLED)

        if use_gpu:
            self.update_status("Rilevata GPU. Avvio esecuzione sequenziale (1 lavoro alla volta).")
            self.executor = ThreadPoolExecutor(max_workers=1)
        else:
            try:
                max_workers = int(self.num_workers_var.get())
                self.update_status(f"Rilevata CPU. Avvio esecuzione parallela con {max_workers} processi.")
                self.executor = ProcessPoolExecutor(max_workers=max_workers)
            except ValueError:
                messagebox.showerror("Errore", "Il numero di processi CPU deve essere un intero.")
                self.add_job_button.config(state=tk.NORMAL)
                self.start_queue_button.config(state=tk.NORMAL)
                return

        for i, job in enumerate(self.jobs_list):
            if job['status'] == 'In attesa':
                job['status'] = 'In esecuzione'
                self.jobs_listbox.delete(i)
                self.jobs_listbox.insert(i, f"ID: {job['job_id']} | Prompt: {'_'.join(job['metadata_prompt'])[:50]}... | Stato: In esecuzione")
                
                params_for_run = job.copy()
                params_for_run.pop('status', None) 
                params_for_run['progress_queue'] = self.progress_queue
                
                # LA CHIAMATA ORA INCLUDE LE OPZIONI DI PULIZIA
                future = self.executor.submit(run_generation, **params_for_run)
                self.active_futures[future] = job['job_id']
                future.add_done_callback(self.job_completed)
    
    def job_completed(self, future):
        """Callback eseguita quando un lavoro termina. ELIMINA il lavoro completato."""
        job_id = self.active_futures.pop(future)
        
        # --- NUOVA LOGICA DI RIMOZIONE ---
        listbox_index = -1
        job_to_remove = None

        # Trova l'indice nella Listbox
        for i in range(self.jobs_listbox.size()):
            if self.jobs_listbox.get(i).startswith(f"ID: {job_id}"):
                listbox_index = i
                break
        
        # Trova il dizionario nella lista dati
        for job in self.jobs_list:
            if job.get("job_id") == job_id:
                job_to_remove = job
                break

        # Rimuovi il lavoro prima dalla lista dati interna...
        if job_to_remove:
            self.jobs_list.remove(job_to_remove)
        
        # ...e poi dalla Listbox visibile
        if listbox_index != -1:
            self.jobs_listbox.delete(listbox_index)

        # Aggiorna lo stato per informare l'utente
        try:
            result_message, _ = future.result()
            self.update_status(f"Lavoro '{job_id}' completato e rimosso dalla coda.")
        except Exception as e:
            self.update_status(f"Lavoro '{job_id}' fallito e rimosso. Errore: {e}")
            logging.error(f"Errore nel future per il lavoro {job_id}", exc_info=True)
        
        # Se non ci sono più lavori in esecuzione, finalizza la sessione
        if not self.active_futures:
            self.root.after(100, self._finalize_queue_processing)

    def _finalize_queue_processing(self):
        """
        Esegue le operazioni di pulizia alla fine di tutti i lavori in esecuzione.
        """
        self.update_status("Tutti i lavori in coda sono stati eseguiti. Pronto per un nuovo avvio.")
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        # Assicurati che questa riga sia stata rimossa o commentata
        # self.jobs_list = [] 
        
        # Ri-abilita i pulsanti
        self.add_job_button.config(state=tk.NORMAL)
        self.start_queue_button.config(state=tk.NORMAL)
        self.update_progress(0)
    
    def build_prompt(self):
        """
        Costruisce il prompt dei metadati e calcola il numero di strumenti
        in base alla modalità selezionata nella GUI.
        Ritorna una tupla: (prompt, num_inst_len)
        """
        # Incolla qui il blocco che hai copiato prima
        prompt, num_inst_len = [], 0
        mode = self.generation_mode.get()

        if mode == "Genere":
            # ... tutta la logica per la modalità Genere
            self.update_status("Modalità Genere: costruzione prompt...")
            genre_token = self.selected_genre.get()
            if not genre_token or "Caricare" in genre_token or "trovato" in genre_token:
                raise ValueError("Selezionare un genere valido dalla lista.")
            
            prompt = [genre_token]
            try:
                num_inst_len = int(self.num_instruments_genre_mode.get())
                if num_inst_len <= 0: raise ValueError()
            except ValueError:
                raise ValueError("Il numero di strumenti stimato deve essere un intero positivo.")
            
        elif mode == "Guidata":
            # ... tutta la logica per la modalità Guidata
            self.update_status("Modalità Guidata: costruzione prompt dal profilo selezionato...")
            profile_name = self.selected_profile.get()
            if not profile_name or "Caricare" in profile_name or "non caricato" in profile_name:
                raise ValueError("Selezionare un profilo valido dalla lista.")
            
            selected_profile_data = next((p for p in self.profiles if p['profile_name'] == profile_name), None)
            if not selected_profile_data: raise ValueError(f"Dati per il profilo '{profile_name}' non trovati.")

            prompt.extend(selected_profile_data.get("instruments", []))
            recommended_tokens = [
                selected_profile_data.get("recommended_key"), selected_profile_data.get("recommended_timesig"),
                selected_profile_data.get("recommended_tempo"), selected_profile_data.get("recommended_avg_vel"),
                selected_profile_data.get("recommended_vel_range"), selected_profile_data.get("recommended_num_inst")]
            prompt.extend([token for token in recommended_tokens if token])
            num_inst_len = len(selected_profile_data.get("instruments", []))

        else: # Modalità Manuale
            # ... tutta la logica per la modalità Manuale
            self.update_status("Modalità Manuale: costruzione prompt dai controlli...")
            prompt = [var.get() for var in self.control_vars.values() if var.get() and "Caricare" not in var.get() and "Nessuno" not in var.get()]
            
            selected_instruments = []
            if self.random_instruments_var.get():
                all_instrument_options = self.metadata_options.get("Instrument", [])
                if not all_instrument_options: raise ValueError("Nessuno strumento disponibile per la selezione casuale.")
                num_to_select = random.randint(1, min(3, len(all_instrument_options)))
                selected_instruments = random.sample(all_instrument_options, num_to_select)
            else:
                selected_instruments = [self.instrument_listbox.get(i) for i in self.instrument_listbox.curselection()]
            
            if not selected_instruments: raise ValueError("Selezionare almeno uno strumento o spuntare 'Scegli casualmente'.")
            
            num_inst_len = len(selected_instruments)
            num_map = {1: "NumInst_Solo", 2: "NumInst_Duet"}
            num_token = num_map.get(num_inst_len)
            if not num_token:
                if 2 < num_inst_len <= 4: num_token = "NumInst_SmallChamber"
                elif 4 < num_inst_len <= 8: num_token = "NumInst_MediumEnsemble"
                else: num_token = "NumInst_LargeEnsemble"
            if num_token and num_token in self.metadata_options.get("NumInst", []): prompt.append(num_token)
            prompt.extend(selected_instruments)

        # Aggiungi il return alla fine
        return prompt, num_inst_len

    def update_progress(self, value):
        self.root.after(0, self.progress_bar.config, {'value': value})

    def update_status(self, message):
        self.root.after(0, self.status_label.config, {'text': message})

# --- Blocco di avvio ---
if __name__ == "__main__":
    # Per motivi di brevità, ho omesso il codice delle funzioni logiche che rimangono invariate.
    # Copia il contenuto di quelle funzioni dal tuo file originale qui.
    # Le funzioni da copiare sono:
    # - browse_file
    # - load_all_data
    # - analyze_model
    # - on_profile_select
    # - add_job_to_queue
    # - start_processing_queue
    # - job_completed
    # - check_progress_queue
    # - randomize_metadata
    # - toggle_instrument_list_state
    # - build_prompt
    
    root = tk.Tk()
    app = MusicGeneratorApp(root)
    root.mainloop()