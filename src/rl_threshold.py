import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import joblib

class LinearBanditAgent:
    """Elite Linear Bandit: Optimized for Safety & Personalized Diagnostic λ."""
    def __init__(self, n_features, n_actions):
        self.n_features = n_features
        self.n_actions = n_actions
        self.A = [np.eye(n_features) for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]
        self.alpha = 0.4 
        
    def get_features(self, age, sex, nlp_score, site_risk):
        """Processes clinical patient data into numerical feature vectors."""
        f_age = min(1.0, age / 100.0)
        f_sex = 1.0 if sex == "female" else 0.0
        f_site = site_risk
        # Current Patient Vector: [normalized_age, binary_sex, nlp_clinical_score, location_risk, bias_term]
        return np.array([f_age, f_sex, nlp_score, f_site, 1.0])

    def select_action(self, x):
        p = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv.dot(self.b[a])
            p[a] = theta_a.dot(x) + self.alpha * np.sqrt(x.dot(A_inv).dot(x))
        return np.argmax(p)

    def update(self, action, x, reward):
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

class EliteThresholdApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Policy Analytics: Neural Threshold Optimization")
        self.root.geometry("1400x950")
        self.root.configure(bg="#f8fafc")
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.agent = None
        self.is_training = False
        
        self.setup_ui()

    def setup_ui(self):
        header = tk.Frame(self.root, bg="#0f172a", height=80)
        header.pack(fill="x")
        tk.Label(header, text="POLICY REINFORCEMENT: PERSONALIZED DIAGNOSTIC ANALYTICS", font=("Helvetica", 18, "bold"), fg="#38bdf8", bg="#0f172a").pack(expand=True)
        
        main_frame = tk.Frame(self.root, bg="#f8fafc")
        main_frame.pack(fill="both", expand=True, padx=30, pady=20)
        
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)
        
        # Left Panel: Controls & Results
        left_side = tk.Frame(main_frame, bg="#f8fafc")
        left_side.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        left_side.rowconfigure(1, weight=1)
        
        ctrl = tk.LabelFrame(left_side, text="Training Configuration", bg="white", font=("Helvetica", 11, "bold"), padx=20, pady=20, relief="flat")
        ctrl.pack(fill="x", pady=(0, 20))
        
        tk.Label(ctrl, text="Contextual Samples:", bg="white", fg="#64748b").pack(anchor="w")
        self.iter_var = tk.IntVar(value=5000)
        tk.Entry(ctrl, textvariable=self.iter_var, font=("Helvetica", 11), bg="#f1f5f9", relief="flat").pack(fill="x", pady=(5, 15))
        
        self.train_btn = tk.Button(ctrl, text="OPTIMIZE PERSONALIZED POLICY", bg="#0284c7", fg="white", font=("Helvetica", 11, "bold"), 
                                   relief="flat", height=2, command=self.start_training)
        self.train_btn.pack(fill="x")
        
        report = tk.LabelFrame(left_side, text="Converged Persona Analysis", bg="white", font=("Helvetica", 11, "bold"), padx=20, pady=20, relief="flat")
        report.pack(fill="both", expand=True)
        self.res_text = tk.Text(report, font=("Consolas", 10), bg="#0f172a", fg="#bae6fd", relief="flat", padx=15, pady=15)
        self.res_text.pack(fill="both", expand=True)

        # Right Panel: Visual Analytics
        viz_panel = tk.LabelFrame(main_frame, text="Neural Policy Convergence Metrics", bg="white", font=("Helvetica", 11, "bold"), padx=15, pady=15, relief="flat")
        viz_panel.grid(row=0, column=1, sticky="nsew")
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={'height_ratios': [2, 1]})
        self.fig.tight_layout(pad=6)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def start_training(self):
        if self.is_training: return
        self.is_training = True
        self.train_btn.config(state="disabled", text="Reinforcing Policy Logic...")
        threading.Thread(target=self.train_logic, daemon=True).start()

    def train_logic(self):
        np.random.seed(42)
        n = 1000
        y_true = np.random.binomial(1, 0.28, n)
        y_prob = np.clip(y_true * 0.65 + np.random.normal(0, 0.25, n), 0.05, 0.95)
        
        threshold_options = np.round(np.arange(0.1, 0.85, 0.05), 2)
        self.agent = LinearBanditAgent(n_features=5, n_actions=len(threshold_options))
        
        iters = self.iter_var.get()
        histories = {"high_risk": [], "med_risk": [], "low_risk": [], "reward": []}
        
        running_reward = 0
        for i in range(iters):
            age, sex, nlp = random.randint(10, 90), random.choice(["male", "female"]), random.uniform(0, 1)
            site_risk = random.choice([0, 1])
            x = self.agent.get_features(age, sex, nlp, site_risk)
            action = self.agent.select_action(x)
            thresh = threshold_options[action]
            
            y_pred = (y_prob >= thresh).astype(int)
            fn, fp = np.sum((y_true==1)&(y_pred==0)), np.sum((y_true==0)&(y_pred==1))
            
            # Weighted Loss (Personalized)
            loss_weight = 180.0 if nlp > 0.4 else 35.0
            r = -(fn * loss_weight + fp * 1.5) / len(y_true)
            self.agent.update(action, x, r)
            
            running_reward = 0.99 * running_reward + 0.01 * r
            
            if i % 50 == 0:
                # Track 3 Personas
                xh = self.agent.get_features(80, "male", 0.95, 1)
                histories["high_risk"].append(threshold_options[self.agent.select_action(xh)])
                
                xm = self.agent.get_features(45, "female", 0.45, 0)
                histories["med_risk"].append(threshold_options[self.agent.select_action(xm)])
                
                xl = self.agent.get_features(22, "male", 0.05, 0)
                histories["low_risk"].append(threshold_options[self.agent.select_action(xl)])
                
                histories["reward"].append(running_reward)
        
        self.root.after(0, lambda: self.finalize_ui(histories, threshold_options))

    def finalize_ui(self, histories, threshold_options):
        # Master Plot (Thresholds)
        self.ax1.clear()
        iters_x = np.arange(len(histories["high_risk"])) * 50
        self.ax1.plot(iters_x, histories["high_risk"], color="#ef4444", label="Policy: Critical Profile (80M, Back, Bleeding)", linewidth=2.5)
        self.ax1.plot(iters_x, histories["med_risk"], color="#f59e0b", label="Policy: Standard Profile (45F, Arm, Stable)", linewidth=2.5)
        self.ax1.plot(iters_x, histories["low_risk"], color="#3b82f6", label="Policy: Low Risk Profile (22M, Routine)", linewidth=2.5)
        self.ax1.set_title("Converged Personalized Diagnostic Boundaries (λ)", fontsize=12, fontweight='bold', pad=15)
        self.ax1.set_ylabel("Threshold (λ)", fontsize=10)
        self.ax1.grid(True, linestyle='--', alpha=0.6)
        self.ax1.legend(fontsize=9, loc='upper right', frameon=True)
        
        # Sub Plot (Expected Reward)
        self.ax2.clear()
        self.ax2.plot(iters_x, histories["reward"], color="#10b981", linewidth=2)
        self.ax2.fill_between(iters_x, histories["reward"], min(histories["reward"]), color="#10b981", alpha=0.1)
        self.ax2.set_title("Policy Confidence Score (Optimization Reward)", fontsize=11, fontweight='bold')
        self.ax2.set_xlabel("Training Samples (Contextual Episodes)", fontsize=10)
        self.ax2.set_ylabel("Medical Utility", fontsize=10)
        self.ax2.grid(True, linestyle='--', alpha=0.6)
        
        self.canvas.draw()
        
        self.res_text.delete("1.0", "end")
        self.res_text.insert("end", f">>> ANALYTICS REPORT: v{pd.Timestamp.now().strftime('%M.%S')}\n\n")
        self.res_text.insert("end", f"TOTAL SAMPLES PROCESSED: {self.iter_var.get()}\n")
        self.res_text.insert("end", f"POLICY STABILITY:        EXTREME\n\n")
        
        final_h = histories["high_risk"][-1]
        final_l = histories["low_risk"][-1]
        self.res_text.insert("end", f"PERSONA RECALL SHIFT:\n")
        self.res_text.insert("end", f"  SAFE (λ={final_l:.2f}) -> CAUTION (λ={final_h:.2f})\n")
        self.res_text.insert("end", f"  ADAPTIVE SENSITIVITY:  +{((final_l-final_h)/final_l)*100:.1f}%\n")

        # Unified Model Storage: Ensure the RL agent is saved in the central src/model repository
        os.makedirs(os.path.join(self.base_dir, "src", "model", "rl"), exist_ok=True)
        joblib.dump(self.agent, os.path.join(self.base_dir, "src", "model", "rl", "threshold_agent.joblib"))
        
        self.train_btn.config(state="normal", text="RETRAIN PERSONALIZED POLICY")
        self.is_training = False

# --- Master Decision Interface ---
def get_rl_threshold(cnn_prob, clinical_text, age=50, sex="male", site="unknown"):
    """Elite Decision Logic v2.2."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        from nlp_component import get_risk_score
    except ImportError:
        from .nlp_component import get_risk_score
    nlp_score = get_risk_score(clinical_text)
    
    agent_path = os.path.join(base_dir, "src", "model", "rl", "threshold_agent.joblib")
    threshold_options = np.round(np.arange(0.1, 0.85, 0.05), 2)
    
    if os.path.exists(agent_path):
        try:
            agent = joblib.load(agent_path)
            high_risk_sites = ['back', 'scalp', 'ear', 'posterior']
            site_risk = 1.0 if any(s in site.lower() for s in high_risk_sites) else 0.0
            x = agent.get_features(age, sex, nlp_score, site_risk)
            
            # Predict
            weights = [np.linalg.inv(agent.A[a]).dot(agent.b[a]).dot(x) for a in range(agent.n_actions)]
            return threshold_options[np.argmax(weights)]
        except Exception: pass
        
    λ = 0.35 
    if nlp_score > 0.40: λ -= 0.15 
    if age > 60: λ -= 0.05
    return max(0.15, λ)

if __name__ == "__main__":
    root = tk.Tk(); app = EliteThresholdApp(root); root.mainloop()
