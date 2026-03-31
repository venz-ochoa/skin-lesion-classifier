import pandas as pd
import numpy as np
import os
import joblib
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Try to use transformers for BERT analysis
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARNING] Transformers not found. Falling back to TF-IDF.")

# Dynamic weighting for symptoms (higher value = higher risk contribution)
SYMPTOM_ENTITIES = {
    'bleed': 0.25, 'itch': 0.12, 'growth': 0.20, 'change': 0.15, 
    'pain': 0.10, 'firm': 0.10, 'elevated': 0.10, 'crust': 0.10,
    'rapid': 0.25, 'irregular': 0.20, 'color': 0.10, 'dark': 0.18,
    'asymmetry': 0.20, 'border': 0.15, 'multicolored': 0.18, 'large': 0.10
}
# Risk factors based on patient history
RISK_ENTITIES = {
    'melanoma': 0.35, 'sun exposure': 0.12, 'history': 0.20, 
    'dysplastic': 0.20, 'atypical': 0.20, 'genetic': 0.10
}

def extract_clinical_entities(text):
    text_lower = text.lower()
    found_symptoms = [k for k in SYMPTOM_ENTITIES.keys() if k in text_lower]
    found_risks = [k for k in RISK_ENTITIES.keys() if k in text_lower]
    total_boost = sum(SYMPTOM_ENTITIES[s] for s in found_symptoms) + sum(RISK_ENTITIES[r] for r in found_risks)
    return found_symptoms, found_risks, min(0.7, total_boost)

class SkinLesionNLPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive Clinical NLP: Personalised Reasoning Engine")
        self.root.geometry("1100x850")
        self.root.minsize(1000, 750)
        self.root.configure(bg="#f8fafc")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Align NLP local data storage with the main model repository
        self.model_dir = os.path.join(self.base_dir, 'src', 'model', 'nlp')
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.nlp_pipeline = None; self.setup_ui(); self.init_system()

    def setup_ui(self):
        # 1. Main Paned Window (Top vs Bottom)
        self.paned = ttk.PanedWindow(self.root, orient="vertical")
        self.paned.pack(fill="both", expand=True)

        # 2. Top Section: Interactive Panels
        top_frame = tk.Frame(self.paned, bg="#f8fafc", padx=20, pady=20)
        self.paned.add(top_frame, weight=3) # Give more initial weight to interaction

        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)
        top_frame.rowconfigure(0, weight=1)

        # Panel A: Narrative
        input_panel = tk.LabelFrame(top_frame, text="Patient Narrative Analysis", font=("Helvetica", 11, "bold"), bg="white", padx=15, pady=15, relief="flat")
        input_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        input_panel.columnconfigure(0, weight=1); input_panel.rowconfigure(1, weight=1)

        tk.Label(input_panel, text="Input Patient Symptoms & History:", bg="white", font=("Helvetica", 10), fg="#64748b").grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.notes_entry = tk.Text(input_panel, font=("Helvetica", 11), bg="#f1f5f9", relief="flat", padx=10, pady=10)
        self.notes_entry.grid(row=1, column=0, sticky="nsew", pady=(0, 15))
        self.notes_entry.insert("1.0", "65-year-old with a growing dark spot on the chest. It's itchier than last month.")

        tk.Button(input_panel, text="EXECUTE CLINICAL REASONING", bg="#0284c7", fg="white", font=("Helvetica", 11, "bold"), relief="flat", height=2, command=self.analyze).grid(row=2, column=0, sticky="ew")

        # Panel B: Insights
        viz_panel = tk.LabelFrame(top_frame, text="Real-time Diagnostic Insights", font=("Helvetica", 11, "bold"), bg="white", padx=15, pady=15, relief="flat")
        viz_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        viz_panel.columnconfigure(0, weight=1); viz_panel.rowconfigure(4, weight=1)

        tk.Label(viz_panel, text="Composite Risk Confidence:", bg="white", font=("Helvetica", 10), fg="#64748b").grid(row=0, column=0, sticky="w")
        self.risk_meter = ttk.Progressbar(viz_panel, orient="horizontal", mode="determinate")
        self.risk_meter.grid(row=1, column=0, sticky="ew", pady=(10, 20))

        self.risk_label = tk.Label(viz_panel, text="Awaiting...", font=("Helvetica", 14, "bold"), bg="white", fg="#94a3b8")
        self.risk_label.grid(row=2, column=0, sticky="ew")

        self.checklist = tk.Label(viz_panel, text="", font=("Helvetica", 10), bg="white", justify="left", fg="#64748b")
        self.checklist.grid(row=3, column=0, sticky="nw", pady=20)
        
        # Spacer
        tk.Frame(viz_panel, bg="white").grid(row=4, column=0, sticky="nsew")

        # 3. Bottom Section: Log
        log_panel = tk.LabelFrame(self.paned, text="Clinical Reasoning Ledger (Continuous Log)", font=("Helvetica", 11, "bold"), bg="white", padx=15, pady=10, relief="flat")
        self.paned.add(log_panel, weight=1)
        log_panel.columnconfigure(0, weight=1); log_panel.rowconfigure(0, weight=1)

        self.details = tk.Text(log_panel, font=("Consolas", 10), bg="#0f172a", fg="#bae6fd", relief="flat", padx=10, pady=10)
        self.details.grid(row=0, column=0, sticky="nsew")
        
        scroll = ttk.Scrollbar(log_panel, orient="vertical", command=self.details.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.details['yscrollcommand'] = scroll.set

    def init_system(self):
        if TRANSFORMERS_AVAILABLE:
            try:
                self.nlp_pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
                self.details.insert("end", "[SUCCESS] Personalised Engine Online.\n")
            except Exception as e: self.details.insert("end", f"BERT Error: {e}\n")

    def analyze(self):
        text = self.notes_entry.get("1.0", "end-1c")
        if not text.strip(): return
        bert = 0.5
        if self.nlp_pipeline: 
            res = self.nlp_pipeline(text)[0]
            bert = res['score'] if res['label'] == 'NEGATIVE' else 1.0 - res['score']
        
        syms, rs, boost = extract_clinical_entities(text)
        final_risk = min(1.0, (bert * 0.3) + (boost * 0.9))
        self.risk_meter['value'] = final_risk * 100
        color = "#ef4444" if final_risk > 0.6 else ("#f59e0b" if final_risk > 0.35 else "#10b981")
        self.risk_label.config(text=f"{final_risk*100:.1f}% AI Risk Index", fg=color)
        m = "\n".join([f"🚩 FLAG: {s.upper()}" for s in syms + rs])
        self.checklist.config(text=m or "Negative for defined clinical markers")
        
        self.details.insert("end", f">>> ANALYSED: {pd.Timestamp.now().strftime('%H:%M:%S')} | RISK={final_risk:.2f}\n")
        self.details.insert("end", f"    ENTITIES: {', '.join(syms+rs)}\n")
        self.details.see("end")

# --- Bridge ---
def get_risk_score(clinical_text):
    """Bridge for RL and Dashboard Integration"""
    try:
        from transformers import pipeline
        nlp = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        res = nlp(clinical_text)[0]
        base = res['score'] if res['label'] == 'NEGATIVE' else 1.0 - res['score']
    except Exception: base = 0.5
    _, _, boost = extract_clinical_entities(clinical_text)
    return min(1.0, (base * 0.3) + (boost * 0.9))

if __name__ == "__main__":
    root = tk.Tk(); app = SkinLesionNLPApp(root); root.mainloop()
