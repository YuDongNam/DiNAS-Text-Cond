# Multi-conditioned Graph Diffusion for Neural Architecture Search (Editing Baseline)
Rohan Asthana, Joschua Conrad, Youssef Dawoud, Maurits Ortmanns, Vasileios Belagiannis

> **Note:** This repository is a modified branch designed to demonstrate a **Graph Editing Neural Architecture Search (NAS)** baseline. Instead of generating architectures from scratch, it conditions the generation on a `Parent Graph` and a `Text` prompt to yield a modified `Child Graph`. This uses the `NAD_triplet_dataset.jsonl`.

This repository contains the code for the paper titled "Multi-conditioned Graph Diffusion for Neural Architecture Search" [\[link\]](https://openreview.net/forum?id=5VotySkajV).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-conditioned-graph-diffusion-for-neural/neural-architecture-search-on-nas-bench-101)](https://paperswithcode.com/sota/neural-architecture-search-on-nas-bench-101?p=multi-conditioned-graph-diffusion-for-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-conditioned-graph-diffusion-for-neural/neural-architecture-search-on-nas-bench-201-1)](https://paperswithcode.com/sota/neural-architecture-search-on-nas-bench-201-1?p=multi-conditioned-graph-diffusion-for-neural)

![alt text](https://github.com/rohanasthana/DiNAS/blob/master/dinas.png)

## Abstract
 Neural architecture search automates the design of neural network architectures usually by exploring a large and thus complex architecture search space. To advance the architecture search, we present a graph diffusion-based NAS approach that uses discrete conditional graph diffusion processes to generate high-performing neural network architectures. We then propose a multi-conditioned classifier-free guidance approach applied to graph diffusion networks to jointly impose constraints such as high accuracy and low hardware latency. 

## Graph Editing Pipeline (How it works)

This baseline modifies the standard generation process into a **conditional editing process** using the following multi-condition strategy:

1. **Data Parsing:** The target `Child Graph` is loaded as a sparse tensor for the diffusion target, while the `Parent Graph` is loaded as a fixed dense tensor padded up to 110 nodes. The `Text Prompt` is embedded into a 768-D continuous condition.
2. **Forward Diffusion (Noise):** Discrete noise is iteratively added *only* to the `Child Graph` ($z_t$). The `Parent Graph` is kept pristine to serve as the structural compass.
3. **Condition Injection:** At every denoising step, the noisy `Child Graph` features are concatenated with the clean `Parent Graph` features along the channel dimension (`X_input = concat([X_t, X_parent], dim=-1)`). The text embedding is also provided conditionally.
4. **Generation / Inference:** The pipeline starts from pure noise $z_T \sim N(0, I)$, and denoises step-by-step $T \rightarrow 0$. At each step, it continuously references the concatenated `Parent Graph` and the `Text Prompt` to construct the final edited `Child Graph`.
5. **Classifier-Free Guidance (CFG):** During the inference denoising steps, CFG is applied to the **Text condition** (`pred = (1+W)*pred_cond - W*pred_uncond`) to control editing intensity, while the `Parent Graph` is always preserved (never dropped) to ensure structural adherence.

> **⚠️ Important Note on Text Embeddings:** The text condition requires a pre-computed 768-D embedding cache. If the `embeddings_file` parameter in your configuration is empty or not provided, the pipeline will fallback to a **deterministic Dummy Embedding (zeros/ones)**. Ensure you connect a valid embedding mapping to achieve actual language-guided node/edge additions.

## Getting Started

To get started with the DiNAS editing baseline project, follow these steps:

1. Clone the repository: `git clone https://github.com/YuDongNam/DiNAS-Text-Cond.git`
2. Load the base conda environment `environment.yml` and install additional data-handling dependencies (e.g., `rdkit`, `pandas`, `seaborn`, `tensorflow-cpu`).
3. **Run the training process on the NAD Triplet Editing Dataset:** 
   ```bash
   python main_reg_free.py dataset=nad
   ```
   This command starts the diffusion model training using the multi-conditioned editing configurations.

4. **Evaluation Metrics:** We provide a dedicated evaluation suite for the editing task.
   ```bash
   python evaluate.py
   ```
   The suite calculates the four key continuous evaluation metrics:
   - **Validity:** Proportion of graphs passing semantic constraints.
   - **Uniqueness:** Diversity among the generated samples.
   - **Novelty:** Proportion of generated graphs NOT present in the training set.
   - **Latency (Simulated Proxy):** A simulated latency proxy metric based on logical graph depth and node counts. **Note:** This is an algorithmic surrogate for baseline demonstration, not actual hardware metrics.

5. **Graph Editing Task Testing (Dry-run):** Run the verified dry-run script to verify local functionality without full training.
   ```bash
   python test_dryrun.py
   ```
   This script executes a full CPU-compatible forward and backward pass, testing the concatenation of sparse child graphs with dense parent structures.


## Cite this paper
```
@article{
asthana2024multiconditioned,
title={Multi-conditioned Graph Diffusion for Neural Architecture Search},
author={Rohan Asthana and Joschua Conrad and Youssef Dawoud and Maurits Ortmanns and Vasileios Belagiannis},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=5VotySkajV},
note={}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
