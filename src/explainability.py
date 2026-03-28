import torch
import torch.nn.functional as F
import timm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from torchvision import transforms
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from datetime import datetime
import sys

# Elite Clinical Components (v2.1 Safety)
try:
    from nlp_component import get_risk_score, extract_clinical_entities
    from rl_threshold import get_rl_threshold, LinearBanditAgent
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from nlp_component import get_risk_score, extract_clinical_entities
    from rl_threshold import get_rl_threshold, LinearBanditAgent

sys.modules['__main__'].LinearBanditAgent = LinearBanditAgent

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        if class_idx is None: class_idx = output.argmax(dim=1).item()
        score = output[:, class_idx]; score.backward()
        weights = np.mean(self.gradients.cpu().data.numpy()[0], axis=(1, 2))
        activations = self.activations.cpu().data.numpy()[0]
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights): cam += w * activations[i]
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0: cam = (cam - np.min(cam)) / np.max(cam)
        return cv2.resize(cam, (x.shape[3], x.shape[2])), class_idx, output

class EliteMultiModalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diagnostic Safety Dashboard v2.3 (Adaptive)")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 700)
        self.root.configure(bg="#f8fafc")

        self.output_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.grad_cam = None
        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'model', 
            'efficientnetb4_v2.pth'
        )

        self.setup_ui()
        self.init_model()

    def setup_ui(self):
        # --- Root uses a horizontal PanedWindow (Sidebar | Main) ---
        self.h_pane = ttk.PanedWindow(self.root, orient="horizontal")
        self.h_pane.pack(fill="both", expand=True)

        # ========== SIDEBAR (Left Pane) ==========
        sidebar_outer = tk.Frame(self.h_pane, bg="#0f172a")
        self.h_pane.add(sidebar_outer, weight=0)

        # Scrollable sidebar using Canvas
        self.sidebar_canvas = tk.Canvas(sidebar_outer, bg="#0f172a", highlightthickness=0, width=370)
        sidebar_scroll = ttk.Scrollbar(sidebar_outer, orient="vertical", command=self.sidebar_canvas.yview)
        self.sidebar = tk.Frame(self.sidebar_canvas, bg="#0f172a")

        # Keep a reference to the canvas window so we can resize it later
        self._sidebar_win_id = self.sidebar_canvas.create_window((0, 0), window=self.sidebar, anchor="nw")
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)

        # When the inner frame resizes, update the scroll region
        self.sidebar.bind("<Configure>", lambda e: self.sidebar_canvas.configure(
            scrollregion=self.sidebar_canvas.bbox("all")))

        # When the canvas itself resizes, stretch the inner frame to match its width
        self.sidebar_canvas.bind("<Configure>", self._on_sidebar_canvas_resize)

        # Enable mouse-wheel scrolling on the sidebar
        self.sidebar_canvas.bind("<Enter>", lambda e: self.sidebar_canvas.bind_all(
            "<MouseWheel>", lambda ev: self.sidebar_canvas.yview_scroll(int(-1 * (ev.delta / 120)), "units")))
        self.sidebar_canvas.bind("<Leave>", lambda e: self.sidebar_canvas.unbind_all("<MouseWheel>"))

        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        sidebar_scroll.pack(side="right", fill="y")

        # Sidebar Contents
        tk.Label(self.sidebar, text="CLINICAL PARAMETERS",
                 font=("Helvetica", 15, "bold"), fg="#10b981", bg="#0f172a").pack(pady=(30, 15), padx=20)

        # -- Image Upload Section --
        self._section(self.sidebar, "Image Input")
        v_frame = tk.Frame(self.sidebar, bg="#1e293b", padx=15, pady=12)
        v_frame.pack(fill="x", padx=15, pady=(0, 10))
        tk.Button(v_frame, text="Upload Lesion Image", bg="#3b82f6", fg="white",
                  font=("Helvetica", 10, "bold"), relief="flat", cursor="hand2",
                  command=self.upload_image).pack(fill="x")

        # -- Patient Metadata Section --
        self._section(self.sidebar, "Patient Metadata")
        m_frame = tk.Frame(self.sidebar, bg="#1e293b", padx=15, pady=12)
        m_frame.pack(fill="x", padx=15, pady=(0, 10))

        tk.Label(m_frame, text="Age:", font=("Helvetica", 9), fg="#94a3b8", bg="#1e293b").pack(anchor="w")
        self.age_var = tk.IntVar(value=45)
        tk.Scale(m_frame, from_=0, to_=100, orient="horizontal", variable=self.age_var,
                 bg="#1e293b", fg="white", highlightthickness=0, troughcolor="#334155").pack(fill="x")

        tk.Label(m_frame, text="Sex:", font=("Helvetica", 9), fg="#94a3b8", bg="#1e293b").pack(anchor="w", pady=(8, 0))
        self.sex_var = tk.StringVar(value="male")
        ttk.Combobox(m_frame, textvariable=self.sex_var, values=["male", "female"],
                     state="readonly").pack(fill="x", pady=(2, 0))

        tk.Label(m_frame, text="Body Site:", font=("Helvetica", 9), fg="#94a3b8", bg="#1e293b").pack(anchor="w", pady=(8, 0))
        self.site_var = tk.StringVar(value="arm")
        ttk.Combobox(m_frame, textvariable=self.site_var,
                     values=["arm", "leg", "back", "face", "scalp", "chest", "posterior"],
                     state="readonly").pack(fill="x", pady=(2, 0))

        # -- Clinical Notes Section --
        self._section(self.sidebar, "Clinical Notes")
        n_frame = tk.Frame(self.sidebar, bg="#1e293b", padx=15, pady=12)
        n_frame.pack(fill="x", padx=15, pady=(0, 10))
        self.notes_entry = tk.Text(n_frame, height=4, font=("Helvetica", 10),
                                   bg="#0f172a", fg="white", relief="flat", wrap="word")
        self.notes_entry.pack(fill="x")
        self.notes_entry.insert("1.0", "Suspicious mole. Irregular, large, and dark.")

        # -- Action Buttons --
        btn_frame = tk.Frame(self.sidebar, bg="#0f172a", padx=15)
        btn_frame.pack(fill="x", pady=15)
        tk.Button(btn_frame, text="DEPLOY SAFETY PIPELINE", bg="#10b981", fg="white",
                  font=("Helvetica", 11, "bold"), relief="flat", height=2, cursor="hand2",
                  command=self.run_full_pipeline).pack(fill="x", pady=(0, 8))

        self.export_btn = tk.Button(btn_frame, text="SAVE DIAGNOSTIC RESULT",
                                    bg="#64748b", fg="white", font=("Helvetica", 10),
                                    relief="flat", height=2, state="disabled", cursor="hand2",
                                    command=self.save_analysis)
        self.export_btn.pack(fill="x")

        self.status_label = tk.Label(self.sidebar, text="System Online",
                                     font=("Helvetica", 9), bg="#0f172a", fg="#64748b")
        self.status_label.pack(pady=(15, 25))

        # ========== MAIN CONTENT (Right Pane) ==========
        self.main_content = tk.Frame(self.h_pane, bg="#f8fafc")
        self.h_pane.add(self.main_content, weight=1)

        # Use grid for the right pane so everything scales
        self.main_content.columnconfigure(0, weight=1)
        self.main_content.rowconfigure(0, weight=0)  # header
        self.main_content.rowconfigure(1, weight=1)  # viz
        self.main_content.rowconfigure(2, weight=0)  # results bar

        # Header
        header = tk.Frame(self.main_content, bg="white")
        header.grid(row=0, column=0, sticky="ew")
        self.header_label = tk.Label(header, text="PERSONALIZED EXPLAINABILITY (v2.3 Adaptive)",
                 font=("Helvetica", 18, "bold"), fg="#1e293b", bg="white")
        self.header_label.pack(padx=30, pady=18, anchor="w")

        # Viz Area (Matplotlib)
        viz_frame = tk.Frame(self.main_content, bg="#f8fafc", padx=10, pady=5)
        viz_frame.grid(row=1, column=0, sticky="nsew")
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(9, 4))
        self.fig.patch.set_facecolor('#f8fafc')
        self.fig.tight_layout(pad=3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Results Bar
        self.res_frame = tk.Frame(self.main_content, bg="white", padx=30, pady=15,
                             highlightthickness=1, highlightbackground="#e2e8f0")
        self.res_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 15))
        self.res_frame.columnconfigure(0, weight=1)

        self.pred_label = tk.Label(self.res_frame, text="Awaiting Multi-Modal Input...",
                                   font=("Helvetica", 22, "bold"), bg="white", fg="#94a3b8",
                                   anchor="w", justify="left")
        self.pred_label.grid(row=0, column=0, sticky="ew")

        self.rationale_label = tk.Label(self.res_frame, text="", font=("Helvetica", 10),
                                        bg="white", fg="#64748b", justify="left",
                                        anchor="w")
        self.rationale_label.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        # Bind the results frame to dynamically update wraplength on resize
        self.res_frame.bind("<Configure>", self._on_res_frame_resize)

    def _on_sidebar_canvas_resize(self, event):
        """Stretch the sidebar inner frame to fill the canvas width."""
        self.sidebar_canvas.itemconfig(self._sidebar_win_id, width=event.width)

    def _on_res_frame_resize(self, event):
        """Dynamically set wraplength so text never overflows the results bar."""
        available = max(event.width - 60, 100)  # account for padding
        # Guard: only reconfigure when the value actually changed to prevent
        # a Configure-event feedback loop that causes layout thrashing.
        if getattr(self, '_last_wraplength', 0) == available:
            return
        self._last_wraplength = available
        self.pred_label.config(wraplength=available)
        self.rationale_label.config(wraplength=available)

    def _section(self, parent, title):
        """Utility: draw a small section header in the sidebar."""
        tk.Label(parent, text=title.upper(), font=("Helvetica", 9, "bold"),
                 fg="#475569", bg="#0f172a").pack(anchor="w", padx=20, pady=(12, 2))

    # ----------------------------------------------------------------
    def init_model(self):
        try:
            self.model = timm.create_model('efficientnet_b4',
                                           pretrained=not os.path.exists(self.model_path),
                                           num_classes=2)
            if os.path.exists(self.model_path):
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device, weights_only=False))
            self.model = self.model.to(self.device).eval()
            self.grad_cam = GradCAM(self.model, self.model.blocks[-2][-1])
        except Exception as e:
            print(f"Model init error: {e}")

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            self.status_label.config(text="Image Loaded ✓", fg="#3b82f6")

    def run_full_pipeline(self):
        if not hasattr(self, 'image_path') or not self.image_path:
            messagebox.showwarning("Missing Input", "Please upload a lesion image first.")
            return
        self.status_label.config(text="Processing...", fg="#f59e0b")
        threading.Thread(target=self._process, args=(
            self.image_path,
            self.notes_entry.get("1.0", "end-1c"),
            self.age_var.get(),
            self.sex_var.get(),
            self.site_var.get()), daemon=True).start()

    def _process(self, img_path, notes, age, sex, site):
        try:
            img = Image.open(img_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            cam, _, output = self.grad_cam(transform(img).unsqueeze(0).to(self.device))
            cnn_prob = F.softmax(output, dim=1).detach().cpu().numpy()[0][1]

            # --- Safety Lock Engine ---
            nlp_score = get_risk_score(notes)
            lam = get_rl_threshold(cnn_prob, notes, age, sex, site)
            entities, risks, _ = extract_clinical_entities(notes)

            safety_triggered = False
            if nlp_score > 0.40 or len(entities) >= 1:
                lam = min(lam, 0.15)
                safety_triggered = True

            pred = "Malignant" if cnn_prob >= lam else "Benign"

            # Prepare Visuals
            img_disp = np.array(img.resize((224, 224))) / 255

            cam = np.where(cam > 0.3, cam, 0)

            cam = cv2.GaussianBlur(cam, (5, 5), 0) # Smooth out pixelated edges
            if np.max(cam) > 0:
                cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            result = (heatmap[..., ::-1] / 255) * 0.35 + img_disp * 0.65

            self.root.after(0, lambda: self.update_ui(
                img_disp, result, pred, nlp_score, lam, cnn_prob,
                entities + risks, age, site, safety_triggered))
            self._log(age, sex, site, nlp_score, lam, cnn_prob, pred,
                      entities + risks, safety_triggered)

        except Exception as e:
            print(f"Pipeline Error: {e}")
            self.root.after(0, lambda: self.status_label.config(
                text="Error – see terminal", fg="#ef4444"))

    def _log(self, age, sex, site, nlp, lam, cnn, pred, ents, safety):
        border = "🛑" * 25 if safety else "🔥" * 25
        print(f"\n{border}")
        print(f"DIAGNOSTIC SAFETY REPORT - {datetime.now().strftime('%H:%M:%S')}")
        print(f"PATIENT: {age}yo {sex.upper()} | SITE: {site.upper()}")
        print(f"NLP RISK: {nlp:.4f} | λ: {lam:.2f} | CNN: {cnn:.4f}")
        if safety:
            print(f">>> SAFETY LOCK ACTIVE: Clinical markers ({', '.join(ents[:3])}) triggered.")
        print(f"FINAL DECISION: [ {pred.upper()} ]")
        print(f"{border}\n")

    def update_ui(self, orig, heat, pred, nlp, lam, cnn, ents, age, site, safety):
        self.ax1.clear()
        self.ax1.imshow(orig)
        self.ax1.axis('off')
        self.ax1.set_title("Clinical Photo", fontsize=11)

        self.ax2.clear()
        self.ax2.imshow(heat)
        self.ax2.axis('off')
        self.ax2.set_title("Grad-CAM Focus Region", fontsize=11)

        self.fig.tight_layout(pad=3)
        self.canvas.draw()

        color = "#ef4444" if pred == "Malignant" else "#10b981"
        self.pred_label.config(text=f"AI DECISION: {pred.upper()}", fg=color)

        warn = "\n⚠ HIGH RISK: Safety Lock overrode vision confidence." if safety else ""
        markers = ', '.join(ents[:4]) if ents else "None detected"
        self.rationale_label.config(
            text=f"Patient: {age}yr  |  Site: {site.upper()}\n"
                 f"Clinical Markers: {markers}\n"
                 f"Vision Prob: {cnn:.4f}  |  Active λ: {lam:.2f}{warn}")

        self.export_btn.config(state="normal")
        self.status_label.config(text="Analysis Complete ✓", fg="#10b981")

    def save_analysis(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"diagnostic_result_{stamp}.png")
        self.fig.savefig(path, bbox_inches='tight', dpi=200, facecolor='#f8fafc')
        messagebox.showinfo("Export Success", f"Diagnostic image saved to:\n{path}")

if __name__ == '__main__':
    root = tk.Tk()
    app = EliteMultiModalApp(root)
    root.mainloop()