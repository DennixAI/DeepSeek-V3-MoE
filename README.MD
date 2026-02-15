# DeepSeek-V3 TinyStories Pipeline 

A high-performance, memory-efficient implementation of the **DeepSeek-V3** architecture, scaled down (~82M parameters) for pre-training on the **TinyStories** dataset using consumer hardware.

## Key Features

### 1. Multi-Head Latent Attention (MLA)
Utilizes low-rank compression to reduce the KV Cache footprint. This allows the model to handle larger contexts and batch sizes during generation by squeezing Key and Value heads into a tiny latent bottleneck.

### 2. DeepSeekMoE (Shared + Routed Experts)
Implements the hybrid Mixture-of-Experts design. Featuring 1 Shared Expert (always active for grammar/syntax) and 8 Routed Experts (specialists). Uses Top-2 routing with a squared-probability auxiliary loss for stable load balancing.

### 3. Muon Optimizer
Incorporates the state-of-the-art **Muon (Momentum Orthogonal)** optimizer for 2D weight matrices, enabling faster convergence than standard AdamW by maintaining orthogonal weight updates.

### 4. Qualitative & Quantitative Monitoring
*   **Periodic Validation:** Tracks loss on unseen data to prevent overfitting.
*   **Generative Sampling:** Automatically writes stories every 500 steps to a Markdown log to monitor linguistic evolution.
*   **Cloud Sync:** Automated uploads to Hugging Face Hub for checkpoints, CSV metrics, and sample logs.

## 🛠️ Technical Specifications
*   **Parameters:** ~82 Million
*   **Architecture:** 6 Layers, 384 Dim, 6 Heads
*   **Tokenizer:** GPT-2 (50,257 Vocab)
*   **Optimization:** BF16 Mixed Precision + Flash Attention
*   **Training Hardware:** Target 14GB VRAM (NVIDIA T4)

## Setup and run
1. Update `config.py` with your Hugging Face Token.
2. Run `python train.py`.
3. Monitor your progress via `training_log.csv` or the `generated_samples.md` file on Hugging Face.