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
| [Auto_MPG](data/Auto_MPG/) | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/9/auto+mpg) |
| [DWD_climate](data/DWD_climate/) | [Tübingen Cause-Effect Pairs](https://webdav.tuebingen.mpg.de/cause-effect/) |
| [Sachs](data/Sachs/) | [bnlearn: Sachs](https://www.bnlearn.com/bnrepository/discrete-small.html#sachs) |
| [Asia](data/asia/) | [bnlearn: Asia](https://www.bnlearn.com/bnrepository/discrete-small.html#asia) |
| [Child](data/child/) | [bnlearn: Child](https://www.bnlearn.com/bnrepository/discrete-medium.html#child) |

Following the dataset instructions in MATMCD, CSV versions of **Auto_MPG, DWD_climate, and Sachs** can be downloaded from the [release accompanying Takayama et al.](https://github.com/mas-takayama/LLM-and-SCD). **Asia and Child** are sampled from their Bayesian network definitions; MATMCD provides the `data/SampleFromBIF.py` script for sampling and CSV conversion. Our experiments use 1,000 samples for each of these two discrete benchmarks.

### Benchmark configurations

The table below describes the dataset versions used in our paper:

| Dataset | Samples | Variables | Reference edges | Type |
|---|---:|---:|---:|---|
| [Auto_MPG](data/Auto_MPG/) | 392 | 5 | 5 | Continuous |
| [DWD_climate](data/DWD_climate/) | 349 | 6 | 6 | Continuous |
| [Sachs](data/Sachs/) | 7,466 | 11 | 19 | Continuous |
| [Asia](data/asia/) | 1,000 | 8 | 8 | Discrete |
| [Child](data/child/) | 1,000 | 20 | 25 | Discrete |

The continuous benchmarks use the processed data and evaluation reference graphs from the Takayama release. In particular, the Sachs evaluation uses its 19-edge reference graph; the bnlearn link above identifies the original benchmark source.

### Data and metadata release

The statistical data files used in the paper, evaluation reference graphs, and variable metadata are available in [`data/`](data/). The release includes all five benchmarks and descriptions for all 50 variables. The original data CSVs and reference matrices are preserved without changing their contents.

Each dataset folder contains:

| File | Contents |
|---|---|
| `<dataset>_data.csv` | Statistical data, with variable names in the first row and one sample per subsequent row. |
| `<dataset>_GTmatrix.csv` | Reference adjacency matrix used for evaluation, without a header or index column. |
| `reference_edges.csv` | The same reference graph as a labeled `source,target` edge list. |
| `metadata.json` | Variable names, full names, meaning descriptions, and synonyms, in CSV column order. |
| `manifest.json` | File paths, data dimensions, column order, formats, source links, and original-file checksums. |
| `variable_statistics.csv` | Observed ranges, missing-value counts, distinct-value counts, and observed codes for discrete variables. |

The variable descriptions in `metadata.json` are extracted from the existing DVCF variable cards. They explain what the dataset columns represent. The separate `manifest.json` describes the dataset files and their organization. No additional mapping between numeric codes and human-readable categories has been inferred during packaging.

For the reference matrices, `A[i,j] = 1` denotes the edge from variable `i` to variable `j`; both axes follow the original CSV column order. The `column_index_1based` field in the variable statistics starts at 1. Discrete numeric codes are categorical labels, not an assertion of ordinal meaning.

See [`datasets.csv`](datasets.csv) for the dataset index, [`DATA_SOURCES.md`](DATA_SOURCES.md) for source and version details, and [`checksums.sha256`](checksums.sha256) for file integrity checks.

**Semantic retrieval corpora and retrieved passages are not included in the public release.** Variable descriptions are included as dataset documentation. Runtime configurations, prompts, execution logs, and model outputs are not included.

We thank the original dataset providers and the authors of MATMCD and the Takayama release for making these benchmark resources available. Please also credit the corresponding original sources when using the datasets.

## Code Availability

**Code coming soon.**

The DVCF implementation and experiment scripts will be released separately from the statistical data and metadata.

## Contact

For questions about this repository, please [open an issue](https://github.com/ikaqiu-Lemon/DVCF/issues).
