# Dynamic Stock Recommendation

An extension of the code and method published with [A Practical Machine Learning Approach for Dynamic Stock Recommendation](https://doi.org/10.1109/TrustCom/BigDataSE/ISPA.2018). The original code repository is [AI4Finance-Foundation/Dynamic-Stock-Recommendation-Machine_Learning-Published-Paper-IEEE](https://github.com/AI4Finance-Foundation/Dynamic-Stock-Recommendation-Machine_Learning-Published-Paper-IEEE). This project is not an official release of, or affiliated with, the original authors. The original paper authors are Hongyang Yang, Xiao-Yang Liu, and Qingwei W. [SSRN version](https://ssrn.com/abstract=3302088).

When referring to the research method, cite the original paper:

> Hongyang Yang, Xiao-Yang Liu, and Qingwei W., “A Practical Machine Learning Approach for Dynamic Stock Recommendation,” in *2018 IEEE International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom)*, 2018. [doi:10.1109/TrustCom/BigDataSE/ISPA.2018.00184](https://doi.org/10.1109/TrustCom/BigDataSE/ISPA.2018.00184).

When referring to the upstream implementation, also cite and link to the [original GitHub repository](https://github.com/AI4Finance-Foundation/Dynamic-Stock-Recommendation-Machine_Learning-Published-Paper-IEEE). This attribution does not replace the need to verify permission and license terms for code, data, and figures before redistribution.

## Purpose and status

The project selects S&P 500 stocks each quarter using fundamental indicators. It uses sector-specific models, rolling training windows, and five regression methods:

- Linear regression
- Ridge regression
- AIC-based stepwise regression
- Random forest
- Gradient boosting

The models estimate the return for the following quarter. Stocks are then ranked and used for portfolio and backtesting analyses.

The current code contains the historical WRDS/Compustat workflow and a free-data extension from 2017-09 using SEC Company Facts, Yahoo Finance, and historical S&P 500 membership snapshots. This extension is an auditable approximation, not an exact 1:1 reproduction of the paper: SEC indicators are formula proxies, SEC data is not fully point-in-time, and the original portfolio balance file and parts of the original environment are unavailable.

The results are research and reproduction material, not investment advice or a trading recommendation.

## Data and external services

The current local development version uses the following sources:

- Historical fundamental tables from WRDS/Compustat, 1990-2017. Redistribution and usage rights must be clarified before publication.
- SEC EDGAR Company Facts for the free-data extension. No API key is required, but SEC requests require a descriptive User-Agent with a contact address.
- Yahoo Finance through `yfinance` for price data. The terms of use and redistribution rights of retrieved data must be reviewed before publication.
- Wikipedia for the current S&P 500 constituent list.
- The public `fja05680/sp500` repository for historical S&P 500 component snapshots. That source documents gaps in its early history.

The upstream repository already tracks historical data files under `Data/` and figures under `figs/`; because this is a fork, those files and their history remain part of the repository. The ignore rules do not untrack inherited files. This extension does not add another copy of those datasets. Newly downloaded caches and generated results are ignored. The code expects historical input files under `Data/1-focasting_data/`; without those files or a future legally suitable alternative, the complete hybrid reproduction cannot run.

## Requirements

- Python 3.12 was used for the current development state.
- Network access to SEC EDGAR, Wikipedia, and Yahoo Finance for dataset construction.
- Sufficient disk space for local price and SEC caches.
- Exact package versions are listed in [pipeline/requirements.txt](pipeline/requirements.txt). Other Python versions have not been formally tested.

## Installation

```shell
git clone https://github.com/Lezrock1/Dynamic-Stock-Recommendation-Machine_Learning-Published-Paper-IEEE.git
cd Dynamic-Stock-Recommendation-Machine_Learning-Published-Paper-IEEE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r pipeline/requirements.txt
```

Set a descriptive User-Agent before fetching SEC or Wikipedia data. The example intentionally contains only a placeholder:

```shell
export DYNAMIC_STOCK_USER_AGENT='dynamic-stock-recommendation-research/1.0 (your-email@example.com)'
```

The [.env.example](.env.example) file can also be used as a reference. It is not loaded automatically, and no API credentials are required.

## Quick start

The complete pipeline downloads data, builds tables, trains the models, and generates reports:

```shell
python pipeline/run_all.py
```

The individual steps can be run separately for debugging:

```shell
python pipeline/build_dataset.py
python pipeline/run_model.py
python pipeline/report.py
python pipeline/paper_performance.py
```

For example, one sector or only feature coverage can be inspected with:

```shell
python pipeline/run_model.py --sector sector45 --features-only
```

Model fitting is computationally expensive and can take a substantial amount of time depending on the dataset and parallelization. Check that no other training process is running before starting another run.

## Results and project structure

```text
pipeline/                  data retrieval, feature construction, models, reports
code/ml_model.py           model core and result serialization
Data/                       local input data and generated tables
cache/                      local API and universe caches
results/                    local predictions, picks, tables, and figures
figs/                       historical figures from the original project
fundamental_*.ipynb         legacy portfolio and backtest notebooks
```

Important local outputs include `results/latest_picks.csv`, sector-specific files under `results/sectorXX/`, and paper comparisons under `results/paper_extension/`. Newly generated outputs and caches are excluded by `.gitignore`. Files already tracked by the upstream repository, including its original data and figures, remain tracked in this fork unless explicitly removed in a commit; ignore rules do not remove them from Git history.

## Survivorship bias and limitations

According to the original documentation, the historical WRDS workflow contains 1,193 historical S&P 500 component stocks. For the free-data extension, membership is checked quarterly using dated snapshots, and some former constituents are added through SEC CIK mappings and historical GICS assignments. This reduces survivorship bias but does not eliminate it: early snapshots are incomplete, some delisted companies have no usable free prices or fundamentals, and SEC restatements are not historical point-in-time data.

Other known limitations:

- SEC ratios are approximations of the original Compustat formulas.
- Not all 20 indicators are available or trainable in every sector; `feature_manifest.csv` documents the selection.
- The original R and portfolio implementations, the exact original backtest balance matrix, and parts of the original data are unavailable.
- The model validation and portfolio assumptions have not been audited as an investment product.

## Planned extensions

A planned research step is to add the STOXX Europe 600. The same workflow should then run over a combined nominal universe of up to 1,100 index constituents (S&P 500 plus STOXX Europe 600). Overlapping companies must be deduplicated, and historical membership, currencies, trading calendars, sector classifications, delistings, missing data, and transaction costs must be handled consistently. Comparable fundamental data for both regions is also required before evaluating a combined model.

## Tests and verification

The pipeline currently has no complete automated test suite. A minimal Python syntax check is:

```shell
python -m py_compile code/ml_model.py fundamental_run_model.py pipeline/*.py
```

After a run, inspect the generated tables and feature manifests for quarterly coverage, missing values, and data-source provenance. Results can change when upstream data is revised.

## License and rights

This repository currently has no selected open-source license. The upstream GitHub repository does not expose a `LICENSE` file (checked 2026-10-03). GitHub's [Terms of Service, section D.5](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#5-license-grant-to-other-users) grant users a limited right to view and reproduce public repository content by forking it through GitHub. This platform-specific fork permission is not a general-purpose open-source license for redistribution outside GitHub. GitHub's [licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository) explains that, without a license, default copyright rules otherwise apply. Rights to third-party material such as WRDS/Compustat data and figures must still be checked before including them in a public fork.

Before publication, also review rights for WRDS/Compustat data, Yahoo Finance data, paper figures, and historical membership data. Yahoo's [Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) restrict automated collection and reuse absent permission; SEC EDGAR's [fair-access guidance](https://www.sec.gov/os/accessing-edgar-data) requires a descriptive User-Agent and reasonable request rates. These service terms do not themselves grant redistribution rights to downloaded data.

A license for original code can only be selected after those rights are clarified. No license is added here deliberately, and a license must not purport to grant rights to third-party material. The paper authors and data sources should be credited appropriately; this is not a substitute for legal advice.

## Publication status

This repository is a GitHub fork of the upstream project. Updates should be added as commits on this fork so the upstream relationship and original history remain intact. The fork inherits the upstream's files and history; a new commit must not add local datasets, caches, generated results, notebook outputs, or personal paths. Review the rights for inherited WRDS/Compustat data and figures before reusing them outside GitHub or redistributing them separately.
