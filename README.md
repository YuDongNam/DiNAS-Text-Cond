# Multi-conditioned Graph Diffusion for Neural Architecture Search (Editing Baseline)
Rohan Asthana, Joschua Conrad, Youssef Dawoud, Maurits Ortmanns, Vasileios Belagiannis

> **Note:** This repository is a modified branch designed to demonstrate a **Graph Editing Neural Architecture Search (NAS)** baseline. Instead of generating architectures from scratch, it conditions the generation on a `Parent Graph` and a `Text` prompt to yield a modified `Child Graph`. This uses the `NAD_triplet_dataset.jsonl`.

This repository contains the code for the paper titled "Multi-conditioned Graph Diffusion for Neural Architecture Search" [\[link\]](https://openreview.net/forum?id=5VotySkajV).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-conditioned-graph-diffusion-for-neural/neural-architecture-search-on-nas-bench-101)](https://paperswithcode.com/sota/neural-architecture-search-on-nas-bench-101?p=multi-conditioned-graph-diffusion-for-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-conditioned-graph-diffusion-for-neural/neural-architecture-search-on-nas-bench-201-1)](https://paperswithcode.com/sota/neural-architecture-search-on-nas-bench-201-1?p=multi-conditioned-graph-diffusion-for-neural)

![alt text](https://github.com/rohanasthana/DiNAS/blob/master/dinas.png)

## Abstract
 Neural architecture search automates the design of neural network architectures usually by exploring a large and thus complex architecture search space. To advance the architecture search, we present a graph diffusion-based NAS approach that uses discrete conditional graph diffusion processes to generate high-performing neural network architectures. We then propose a multi-conditioned classifier-free guidance approach applied to graph diffusion networks to jointly impose constraints such as high accuracy and low hardware latency. 

## Getting Started

To get started with the DiNAS editing baseline project, follow these steps:

1. Clone the repository: `git clone https://github.com/rohanasthana/DiNAS.git`
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
   - **Latency (Simulated):** A simulated latency proxy metric based on graph depth/nodes.

5. **Graph Editing Task Testing (Dry-run):** Run the verified dry-run script to verify local functionality without full training.
   ```bash
   python test_dryrun.py
   ```
   This script executes a full CPU-compatible forward and backward pass, testing the concatenation of sparse child graphs with dense parent structures.
- `nasbench101`: for the NAS-Bench-101 benchmark
- `nasbench201`: for the NAS-Bench-201 benchmark

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
