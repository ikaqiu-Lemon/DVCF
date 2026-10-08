# DVCF

**Beyond Edge Aggregation: Dual-View Evidence Fusion for Causal Discovery**

Official repository for the DVCF paper.

> **Data release:** In preparation. The initial release will contain the statistical benchmark datasets used in the paper.
>
> **Code coming soon.**

## Overview

Dual-View Causal Fusion (DVCF) combines statistical and semantic evidence for causal structure discovery. The framework supports directional assessment and candidate-graph decisions through two complementary variants: DVCF-PF (Precision-Focused) and DVCF-RF (Recall-Focused).

## Statistical Benchmarks

The paper evaluates DVCF on the following five benchmarks:

| Dataset | Samples | Variables | Reference edges | Type |
|---|---:|---:|---:|---|
| Auto_MPG | 392 | 5 | 5 | Continuous |
| DWD_climate | 349 | 6 | 6 | Continuous |
| Sachs | 7,466 | 11 | 19 | Continuous |
| Asia | 1,000 | 8 | 8 | Discrete |
| Child | 1,000 | 20 | 25 | Discrete |

**Data files have not yet been uploaded.** This repository is being prepared for the statistical benchmark data release.

## Data Sources

- **Auto_MPG, DWD_climate, and Sachs:** the processed data and evaluation reference graphs follow the [release accompanying Takayama et al.](https://github.com/mas-takayama/LLM-and-SCD).
- **Asia and Child:** the paper uses 1,000-sample discrete datasets with reference networks documented in the [bnlearn Bayesian Network Repository](https://www.bnlearn.com/bnrepository/).
- **Auto_MPG original source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/9/auto+mpg).

The Sachs evaluation uses the 19-edge reference graph from the Takayama release. Dataset dimensions and reference-edge counts above correspond to the versions used in this paper.

The benchmarks originate from third-party providers. Source attribution and applicable terms will accompany the data release; this repository does not grant additional rights to third-party materials.

## Code Availability

**Code coming soon.**

The DVCF implementation and experiment scripts are not included in the current repository. Release updates will be posted here.

## Contact

For questions about this repository, please [open an issue](https://github.com/ikaqiu-Lemon/DVCF/issues).
