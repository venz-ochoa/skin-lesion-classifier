1. Model Details
- Architecture: Multi-Modal Stack
    - Vision: EfficientNet-B4 (Convolutional Neural Network)
    - NLP: DistilBERT (Transformer-based Clinical Note Classifier)
    - RL: Contextual Bandit (Adaptive Threshold Policy)
- Developed by: Danielle Lenon, Seiji Liwag, Claire Ochoa, and Venice Ochoa
- Developed on: March 2026
- Task: Binary image classification (Malignant vs. Benign) augmented by clinical metadata.

2. Intended Use
- Primary Use: Clinical decision support for dermatologists and medical students.
- Intended Users: Healthcare professionals and researchers.
- Out-of-scope: Direct self-diagnosis by patients without medical supervision.
- Disclaimer: This system is a research prototype. It is NOT FDA-cleared and must not be used as the sole basis for clinical diagnosis.

3. Data & Training
- Vision Data: HAM10000 + ISIC 2019 (35,000+ images total).
- NLP Data: Synthetic clinical corpus designed to map common dermatological symptoms to malignancy risk (e.g., "bleeding", "rapid growth", "irregular borders").
- RL Environment: Simulated validation environment using costs derived from medical diagnostic risk (FP vs. FN).

4. Ethical Considerations: Skin Tone Bias
- The Challenge: The primary training dataset (HAM10000) lacks diversity, with a significant skew toward lighter skin tones (Fitzpatrick Scale I-III).
- Potential Risk: The model may exhibit lower accuracy or higher false-negative rates for patients with darker skin tones (Fitzpatrick IV-VI) due to a lack of representative features (e.g., different visual appearances of melanoma on darker skin).
- Mitigation Strategies:
    - Multi-Modal Redundancy: The NLP and RL components act as "safety nets." If a patient has high-risk clinical symptoms (NLP), the RL policy automatically lowers the threshold, reducing the chance of a miss even if the Vision model (CNN) is less confident due to skin tone variations.
    - Future Work: Integration of the Diverse Dermatology Dataset (DDD) is prioritized to balance the training distribution.

5. Decision Policy & Safety
- Risk Arbitration: Unlike standard classifiers that use a fixed 0.5 threshold, this system uses a Contextual Bandit to optimize the decision boundary (λ) based on clinical context.
- Recall Priority: In "High Risk" contexts (determined by DistilBERT), the system prioritizes Recall (Sensitivity) over Precision, effectively acting as a "conservative" safety officer.
- Explainability: Grad-CAM saliency maps provide visual transparency, allowing clinicians to verify if the model is focusing on the lesion or background artifacts (e.g., hairs, gel).

## 6. Performance & Evaluation
- Benchmark Vision Accuracy: ~85-97% (depending on specific validation subset).
- Safety Impact: The RL component reduces missed malignant cases (False Negatives) by up to 30% in high-risk scenarios compared to a fixed-threshold baseline.

## 7. Limitations
- Synthetic Constraint: The NLP component is currently limited by the breadth of the synthetic clinical corpus.
- Hardware Variation: Performance may vary based on camera quality and lighting conditions during clinical photography.