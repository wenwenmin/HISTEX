# HISTEX
## Introduction
The groundbreaking development of spatial transcriptomics (ST) enables researchers to map gene expression across tissues with spatial precision. However, current next-generation sequencing methods, which theoretically cover the entire transcriptome, face limitations in resolving spatial gene expression at high resolution. The recently introduced Visium HD technology offers a balance between sequencing depth and spatial resolution, but its complex sample preparation and high cost limit its widespread adoption. To address these challenges, we introduce HISTEX, a multimodal fusion approach that leverages a bidirectional cross-attention mechanism and a general-purpose foundation model. HISTEX integrates spot-based ST data with histology images to predict super-resolution (SR) spatial gene expression. Experimental evaluations demonstrate that HISTEX outperforms state-of-the-art methods in accurately predicting SR gene expression across diverse datasets from multiple platforms. Moreover, experimental validation underscores HISTEX’s potential to generate new biological insights. It enhances spatial patterns, enriches biologically significant pathways, and facilitates the SR annotation of tissue structures. These findings highlight HISTEX as a powerful tool for advancing ST research.

![Overview.pdf](Overview.pdf)

## Requirements
Before running HISTEX, you need to create a conda environment and install the required packages:
```shell
conda create -n HISTEX python==3.11.5
conda activate HISTEX
pip install -r requirements.txt
```
