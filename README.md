# 🧭 Code Blue: Predictive Modeling of Depression from Linguistic Signals

**Code Blue** is a research-driven machine learning system designed to analyze linguistic and behavioral patterns associated with depression. The project benchmarks **classical, neural, and transformer-based models** to infer **diagnostic polarity** (positive, negative, recovery-oriented) and **underlying psychological root causes**, with a strong emphasis on **interpretability, efficiency, and real-time applicability**.

This work was developed as an academic research project under **Vellore Institute of Technology (VIT)** and is structured to support experimental reproducibility, academic evaluation, and future research extensions.

---

## 📌 Project Metadata

- **Project Title:** Code Blue – Predictive Modeling of Depression Using Machine Learning  
- **Institution:** Vellore Institute of Technology (VIT)  
- **Course:** Project–I  
- **Guide:** **Dr. Suganthini C**

### Team Members
- **Akki Aryan** (22BCE2475)  
- **Om Kumar** (22BCE2663)  
- **Shatakshi Singh** (22BCE3079)

---

## 🧠 Motivation & Research Context

Depression is not a binary condition. It manifests as a spectrum of psychological states shaped by early life experiences, personality traits, trauma exposure, and substance-related coping mechanisms. While multimodal systems incorporating vision, audio, and physiological signals promise higher accuracy, they introduce heavy computational, privacy, and deployment constraints.

**Code Blue** addresses this gap by demonstrating that **language alone—when analyzed carefully—encodes rich behavioral and emotional signals** sufficient for meaningful inference. Rather than treating depression modeling as a simple classification task, this project frames it as an **interpretive process** that balances prediction with explanation.

---

## 🧪 Problem Formulation

Given a user-generated textual input (e.g., social media post, blog entry, reflective narrative), the system aims to:

1. **Predict diagnostic polarity**
   - Positive (Recovery / Resilience)
   - Negative (Active distress)
   - Reflective / Transitional states

2. **Infer dominant psychological root cause(s)**
   - Trauma and Stress  
   - Early Life Factors  
   - Personality-driven Vulnerabilities  
   - Drug and Alcohol-related Coping  

3. **Explain predictions** using interpretable linguistic cues rather than opaque probability scores.

---

## 🏗️ System Architecture

The system follows a lightweight, modular pipeline:

1. Text ingestion and preprocessing  
2. Feature extraction  
   - TF–IDF n-grams  
   - Recovery-aware lexical indicators  
   - Behavioral polarity signals  
3. Model inference  
   - Classical ML models  
   - Neural architectures  
   - Transformer-based encoders  
4. Explainability layer  
   - Trigger-word contribution analysis  
   - Confidence-aware abstention  
5. Interactive visualization  
   - Streamlit-based research interface  

The architecture intentionally avoids heavy multimodal dependencies to ensure deployability on limited hardware.

---

## 🧰 Technology Stack

### Programming & Machine Learning
- Python 3.10+
- scikit-learn
- PyTorch
- HuggingFace Transformers
- NumPy, Pandas

### Models Benchmarked
- Logistic Regression  
- Naïve Bayes  
- Random Forest  
- Linear Support Vector Machine (SVM)  
- Convolutional Neural Network (CNN)  
- BiLSTM  
- **DistilBERT (final deployment model)**  

### Explainability & Visualization
- SHAP
- Custom TF–IDF contribution analysis
- Plotly

### Deployment & Interface
- Streamlit
- Cloudflare Tunnel (for live demonstrations)

---

## 📊 Experimental Results (Diagnosis Task)

All models were trained and evaluated on a unified labeled dataset of depression-related narratives.

| Model                | Accuracy | Precision | Recall | F1-Score | Compute Cost |
|---------------------|----------|-----------|--------|----------|--------------|
| Logistic Regression | ~78%     | High      | Moderate | Balanced | Very Low     |
| Naïve Bayes         | ~74%     | Moderate  | High     | Moderate | Very Low     |
| Random Forest       | ~81%     | High      | High     | High     | Medium       |
| Linear SVM          | ~83%     | High      | High     | High     | Medium       |
| CNN                 | ~84%     | High      | High     | High     | High         |
| BiLSTM              | ~86%     | High      | High     | High     | High         |
| **DistilBERT**      | **~89–91%** | **Very High** | **Very High** | **Very High** | Optimized |

**Key Observation:**  
DistilBERT achieves near state-of-the-art performance while maintaining lower inference latency than larger transformer models, making it suitable for real-time analysis.

---

## 🔍 Interpretability Highlights

Unlike black-box classifiers, Code Blue provides:
- Trigger words and phrases influencing predictions  
- Directional contribution (positive vs negative)  
- Root-cause probability distributions  

This supports **human-in-the-loop validation** and aligns with ethical and explainable AI principles.

---

## 🧪 Streamlit Research Interface

The Streamlit interface supports:
- Live text analysis  
- Per-chunk prediction breakdown  
- Confidence-aware abstention  
- Highlighted linguistic triggers  
- Model comparison experimentation  

The interface is designed for **academic demonstrations, vivas, and exploratory research**, not clinical diagnosis.

---

## ⏱️ Project Timeline (5 Months)

| Month | Work Conducted |
|------|----------------|
| Month 1 | Literature review, dataset analysis, problem formulation |
| Month 2 | Classical ML model implementation and benchmarking |
| Month 3 | Neural models (CNN, BiLSTM) and performance analysis |
| Month 4 | Transformer integration (DistilBERT), interpretability layer |
| Month 5 | Streamlit interface, result consolidation, documentation |

---

## 🚀 Why Code Blue Stands Out

- Treats language as behavioral signal, not mere sentiment  
- Balances performance with interpretability  
- Avoids unnecessary multimodal overhead  
- Designed for real-time, low-compute environments  
- Easily extensible for future research and clinical collaboration  

---

## 🔮 Future Scope

- Multilingual depression modeling  
- Temporal progression tracking  
- Lightweight multimodal fusion (text + metadata)  
- Federated and privacy-preserving learning  
- Clinical collaboration for validation studies  

---

## ⚠️ Disclaimer

This project is a **research prototype**.  
It is **not a medical or clinical diagnostic tool** and must not be used for diagnosis or treatment decisions.

---

## 📚 References

References consolidated from reviewed literature and associated project documentation.  
See the final project report for full citations.

---
