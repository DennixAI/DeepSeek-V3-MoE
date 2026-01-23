***

# DeepSeek-V3 Efficient Training Pipeline 🚀

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Hub-yellow)](https://huggingface.co/)

A rigorous, from-scratch implementation of the **DeepSeek-V3** architecture, engineered for efficient pre-training on consumer GPUs (**14GB VRAM** target).

This repository is not just a model definition; it is a complete **distributed training framework** featuring state-of-the-art optimizations like **Multi-Head Latent Attention (MLA)**, **DeepSeekMoE**, and the **Muon optimizer**.

---

## ⚡ Key Features

### 🧠 Advanced Architecture (DeepSeek-V3)
*   **Multi-Head Latent Attention (MLA):** Implements Low-Rank Key-Value Joint Compression to drastically reduce KV Cache memory usage during inference and training.
*   **DeepSeekMoE (Mixture-of-Experts):** Features the specific **Shared + Routed Experts** design. Unlike standard MoEs, this architecture separates "general knowledge" (shared experts) from "specialized knowledge" (routed experts) for superior performance.

### 🛠️ System Optimizations
*   **Muon Optimizer:** Implements the momentum-orthogonal optimizer for efficient training of hidden layers, combined with AdamW for embeddings.
*   **Liger Kernel Support:** Uses `LigerFusedLinearCrossEntropyLoss` to fuse the final projection and loss calculation, saving gigabytes of VRAM.
*   **Vectorized Routing:** Optimized expert routing logic that avoids slow Python loops, maximizing GPU throughput.
*   **Gradient Checkpointing:** Custom implementation to fit deep MoE models into limited VRAM.

### ⚙️ Robust Infrastructure
*   **Automatic Resume:** Intelligently detects crashes and resumes training from the latest checkpoint (Local or Hugging Face).
*   **Distributed Training (DDP):** Built for multi-GPU setups with `no_sync` gradient accumulation to minimize network latency.
*   **Hugging Face Integration:** Automatically pushes checkpoints to the Hub during training.

---

## 📉 VRAM Requirements & Scaling

This project is configured by default to run on a **single NVIDIA T4 (16GB)** or **RTX 3060/4070 (12GB+)**.

| Configuration | Parameters | VRAM Usage | Hardware Target |
| :--- | :--- | :--- | :--- |
| **Default (Mini)** | **~550M** | **~10-12 GB** | **Colab T4, RTX 3060** |

---

##  Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/DenniXai/DeepSeek-V3-MOE.git
cd DeepSeek-V3-MOE
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file in the root directory to handle Hugging Face authentication:
```env
HF_TOKEN=hf_your_token_here_xxxx
```

### 3. Start Training
To start training on the **FineWeb-Edu** dataset (streaming mode):
```bash
python train.py
```

---

## 📂 Project Structure

*   **`model.py`**: The core architecture. Contains the `MLA` attention mechanism, `DeepSeekMoE` routing logic, and `RMSNorm`.
*   **`train.py`**: The training engine. Handles the training loop, DDP setup, Muon optimizer grouping, and checkpoint management.
*   **`config.py`**: Configuration center. Controls model dimensions, learning rates, and VRAM constraints.
*   **`data.py`**: Data pipeline. Streams the `HuggingFaceFW/fineweb-edu` dataset efficiently.

---

## 🔬 Technical Deep Dive

### Why MLA?
Standard Multi-Head Attention stores a massive KV cache, which creates a memory bottleneck. MLA projects Keys and Values into a low-rank latent vector (`kv_lora_rank`), significantly compressing the memory footprint without sacrificing performance.

### Why Shared Experts?
In standard MoE, all experts are routed. DeepSeek-V3 introduces "Shared Experts" that *always* activate. This allows the model to capture common linguistic patterns (grammar, syntax) in the shared experts, leaving the routed experts free to specialize in complex reasoning tasks.


### 🚀 Optimization Strategy: Muon-AdamW Hybrid
This implementation utilizes a dual-optimizer approach to maximize training efficiency:
- **`Muon (Momentum Orthogonal Optimizer):`** Applied to all 2D internal weight matrices (Attention & MoE Experts). It de-correlates features via orthogonal updates, typically leading to 1.5x-2x faster early-stage convergence compared to standard AdamW.
- **`AdamW:`** Used for 1D parameters (RMSNorm, Biases) and Embeddings, where orthogonal updates are not mathematically applicable.
- **`Auto-Fallback:`** The script intelligently detects the environment; if the muon library is missing, it seamlessly reverts to an optimized pure AdamW parameter grouping to ensure training remains functional.


---

## 📜 Credits

This implementation is based on the research presented in:
*   *DeepSeek-V3 Technical Report* (DeepSeek AI)
*   *Muon: Momentum Orthogonal Optimizer* (Keller Jordan et al.)

---

**License:** MIT