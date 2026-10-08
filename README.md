# DVCF

**Beyond Edge Aggregation: Dual-View Evidence Fusion for Causal Discovery**

Official repository for the DVCF paper.

[Overview](#overview) | [Datasets](#datasets) | [Code](#code-availability) | [Contact](#contact)

## Overview

Dual-View Causal Fusion (DVCF) combines statistical and semantic evidence for causal structure discovery. The framework supports directional assessment and candidate-graph decisions through two complementary variants: DVCF-PF (Precision-Focused) and DVCF-RF (Recall-Focused).

## Datasets

We use the same five statistical benchmark datasets and public download sources listed in [MATMCD](https://github.com/D2I-Group/matmcd). The original datasets and Bayesian network definitions are available from the following sources:

| Dataset | Download source |
|---|---|
| Auto_MPG | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/9/auto+mpg) |
| DWD_climate | [Tübingen Cause-Effect Pairs](https://webdav.tuebingen.mpg.de/cause-effect/) |
| Sachs | [bnlearn: Sachs](https://www.bnlearn.com/bnrepository/discrete-small.html#sachs) |
| Asia | [bnlearn: Asia](https://www.bnlearn.com/bnrepository/discrete-small.html#asia) |
| Child | [bnlearn: Child](https://www.bnlearn.com/bnrepository/discrete-medium.html#child) |

Following the dataset instructions in MATMCD, CSV versions of **Auto_MPG, DWD_climate, and Sachs** can be downloaded from the [release accompanying Takayama et al.](https://github.com/mas-takayama/LLM-and-SCD). **Asia and Child** are sampled from their Bayesian network definitions; MATMCD provides the `data/SampleFromBIF.py` script for sampling and CSV conversion. Our experiments use 1,000 samples for each of these two discrete benchmarks.

### Benchmark configurations

The table below describes the dataset versions used in our paper:

| Dataset | Samples | Variables | Reference edges | Type |
|---|---:|---:|---:|---|
| Auto_MPG | 392 | 5 | 5 | Continuous |
| DWD_climate | 349 | 6 | 6 | Continuous |
| Sachs | 7,466 | 11 | 19 | Continuous |
| Asia | 1,000 | 8 | 8 | Discrete |
| Child | 1,000 | 20 | 25 | Discrete |

The continuous benchmarks use the processed data and evaluation reference graphs from the Takayama release. In particular, the Sachs evaluation uses its 19-edge reference graph; the bnlearn link above identifies the original benchmark source.

### Data and metadata release

We additionally extracted and organized metadata associated with these statistical datasets for our experiments. The statistical data files used in the paper and the accompanying metadata will be uploaded to this repository.

**Semantic data will not be uploaded or included in the public release.**

We thank the original dataset providers and the authors of MATMCD and the Takayama release for making these benchmark resources available. Please also credit the corresponding original sources when using the datasets.

## Code Availability

**Code coming soon.**

The DVCF implementation and experiment scripts will be released separately from the statistical data and metadata.

## Contact

For questions about this repository, please [open an issue](https://github.com/ikaqiu-Lemon/DVCF/issues).
