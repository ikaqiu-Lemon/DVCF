# Dataset sources

The five benchmarks use the same public download sources listed in [MATMCD](https://github.com/D2I-Group/matmcd). The URLs below identify the original benchmark or network sources; the packaged CSVs are the specific versions used in the DVCF paper.

| Dataset | Original benchmark or network source | Packaged data/reference version |
|---|---|---|
| Auto_MPG | [UCI Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg) | 392 rows and five variables; processed data and reference graph from the [Takayama release](https://github.com/mas-takayama/LLM-and-SCD). |
| DWD_climate | [Tübingen Cause-Effect Pairs](https://webdav.tuebingen.mpg.de/cause-effect/) | 349 rows and six variables; processed data and reference graph from the Takayama release. |
| Sachs | [bnlearn Sachs](https://www.bnlearn.com/bnrepository/discrete-small.html#sachs) | 7,466 rows and 11 variables; processed data and the 19-edge evaluation reference from the Takayama release. |
| Asia | [bnlearn Asia](https://www.bnlearn.com/bnrepository/discrete-small.html#asia) | The fixed 1,000-row discrete sample used in the paper, with the eight-edge reference network. |
| Child | [bnlearn Child](https://www.bnlearn.com/bnrepository/discrete-medium.html#child) | The fixed 1,000-row discrete sample used in the paper, with the 25-edge reference network. |

MATMCD links to the Takayama release for CSV versions of Auto_MPG, DWD_climate, and Sachs, and describes `data/SampleFromBIF.py` for sampling Asia and Child networks and exporting CSV. This package preserves the existing sample files used in the paper; no data were downloaded again or resampled during packaging.

## Variable metadata

The variable descriptions, full names, and synonyms in `metadata.json` come from the existing DVCF variable cards. They are supplied as dataset documentation. File manifests and observed-value statistics were prepared from the packaged CSV files.

## Attribution

Please credit the original dataset providers and relevant source publications when using these benchmarks. The data and reference graphs retain their applicable upstream terms; this package does not assign a new blanket license to third-party materials.

Auto MPG dataset citation: Quinlan, R. (1993). *Auto MPG* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5859H.

Additional provenance and original references are available on the linked source pages and in the DVCF paper.
