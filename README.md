# Dynamic Stock Recommendation

An extension of the code and method published with [A Practical Machine Learning Approach for Dynamic Stock Recommendation](https://doi.org/10.1109/TrustCom/BigDataSE/ISPA.2018). The original code repository is [AI4Finance-Foundation/Dynamic-Stock-Recommendation-Machine_Learning-Published-Paper-IEEE](https://github.com/AI4Finance-Foundation/Dynamic-Stock-Recommendation-Machine_Learning-Published-Paper-IEEE). This project is not an official release of, or affiliated with, the original authors. The original paper authors are Hongyang Yang, Xiao-Yang Liu, and Qingwei W. [SSRN version](https://ssrn.com/abstract=3302088).

When referring to the research method, cite the original paper:

> Hongyang Yang, Xiao-Yang Liu, and Qingwei W., “A Practical Machine Learning Approach for Dynamic Stock Recommendation,” in *2018 IEEE International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom)*, 2018. [doi:10.1109/TrustCom/BigDataSE/ISPA.2018.00184](https://doi.org/10.1109/TrustCom/BigDataSE/ISPA.2018.00184).

When referring to the upstream implementation, also cite and link to the [original GitHub repository](https://github.com/AI4Finance-Foundation/Dynamic-Stock-Recommendation-Machine_Learning-Published-Paper-IEEE). This attribution does not replace the need to verify permission and license terms for code, data, and figures before redistribution.

## Purpose and status

### Validation performance without model changes

Historical and signal-panel validation indexes membership once per call instead of scanning
the entire membership table for every row. Period windows and label-end times are reused only
within that validation call; no persistent cache can conceal changed inputs. Every nonmissing
feature component still receives its source, publication and cutoff checks. Estimators,
features, thresholds, forecasts and optimizer arithmetic are unchanged.

A local five-repeat comparison on the same synthetic 2,100-row / 42,000-component panel
measured median validation time of 0.793 s before versus 0.136 s after (about 5.8x).
This measures this validation stage, not the full model runtime or predictive accuracy.
Valid inputs and ten intentionally invalid variants produced identical accept/reject results
and error messages. Hardware and input sizes affect timing.

### Portable quick start (no private cache required)

From the cloned checkout on macOS or Linux, using Python 3.12 (reference: 3.12.2).
On Windows, use WSL2/Linux: the durable process locks use POSIX `fcntl`, so native
Windows execution is not currently supported.

```bash
python3 -m venv .venv
# macOS / Linux:
source .venv/bin/activate
python -m pip install -r pipeline/requirements.txt
python reproduce.py check
python reproduce.py test
python reproduce.py demo
python reproduce.py dashboard
```

Open `http://127.0.0.1:8765`. The demo is explicitly synthetic, works offline, creates
`state/reproduction/synthetic.sqlite3` and never starts a scheduler or broker orders.
It refuses to overwrite an existing demo database. Preserve that database for viewing;
to repeat from scratch, use the existing CLI with a new relative database filename:

```bash
python -m pipeline.paper_portfolio --env synthetic --db state/reproduction/second.sqlite3 demo
```

The portable launcher anchors execution to its own checkout, not the caller's working
directory. It can also be called from elsewhere with a relative path to `reproduce.py`.
There are no user-specific home-directory paths in its commands. Runtime code necessarily
resolves local filesystem locations internally; relative configuration paths are not
replaced with another user's machine paths.

This reproduces the software tests and synthetic depot flow, **not the private Q2 research
snapshot**. Real runs additionally require SEC identification, acquired source data and
reviewed manifests. Local `state/`, caches, databases and backups are intentionally not
distributed. Set `DYNAMIC_STOCK_USER_AGENT` in your own environment before SEC acquisition;
the placeholder in `.env.example` is not a real contact and that file is not auto-loaded.
Never commit the actual contact or credentials. Original dataset redistribution rights
must be checked independently. Research commands naming dated local snapshots are audit
records, not fresh-clone quick-start commands.

Direct dependencies are pinned and checked. This is not yet a cross-platform lock of
all transitive dependencies or a guarantee of bit-identical scientific results across
Python, BLAS and OS versions. Missing local real-source fixtures are explicitly skipped
in offline tests rather than represented as historical evidence.
GitHub Actions runs the dependency check, offline tests and demo on Linux and macOS;
these remote checks only execute once the workflow is published.

### Manual paper workflow: current result, 2026-10-16

**Update after PSKY recovery:** the same-class history is now recovered via the source's current
SKYD alias, with original company/exchange continuity evidence, not predecessor concatenation.
The exact 253-session interval is 2025-08-28–2026-08-31. A new fully copied input/code snapshot
`state/research/diagnostics/20261016-q2-psky-full-snapshot/` completed both offline passes.
**All three numerical optimizations now succeed**: 76 positive equal weights, 39 minimum-variance
and 37 Max-Sharpe positions above `1e-9`. Each result is fully invested, long-only and capped at 5%.
All 362 forecasts, the 76-stock selection and retained features match the previous diagnostic exactly;
only PSKY's missing covariance input was intentionally repaired. Independent exhaustive split-basis
verification, causal feature/label qualification and the real ISIN map are still open.
The three downloadable `*_DIAGNOSTIC_UNVERIFIED.csv` files are **not handoff portfolios**.
The covariance errors described in the earlier bullets below are the preserved earlier run, not
the current numerical result.

**Original-reference policy:** all eleven local original sector workbooks are byte-identical to
upstream commit `0f29f6027fe4061a1484fb7529402706b3903aca` (commit timestamp 2018-06-06).
That timestamp alone does not independently prove when the bytes became public.
The user's explicit `original-repository-reference-assumption-v1` accepts these identified
files as reference inputs, **not as PIT-certified historical observations**. The diagnostic's
`--original-reference-assumption` option freezes all eleven files and compares every retained
original training row, all 20 features and its target to the original workbook. Only numerical
CSV round-trip noise is tolerated. Missing, ambiguous or changed source rows fail explicitly.
The option never applies to SEC rows, identity, membership or unknown target/action definitions.
The dashboard exposes the assumption and the row-level source comparison separately.

The signal-only executor also supports the explicit definition
`training_vintage_policy=published-version-at-training-v1`: verified training versions must
exist by the target model's training cutoff, not necessarily by their original feature quarter.
The four validation decisions and current prediction retain their own cutoffs. Labels still
need to be completed by training. Omission preserves `own-decision-vintage-v1`, including the
37-quarter executor's strict historical default. Neither policy certifies source versions or
promotes the original-reference assumption into `VERIFIED`. In particular, 2026 reference
eligibility is not evidence that the same bytes were available for a 2017 decision.

```bash
.venv/bin/python -m pipeline.q2_numeric_diagnostic \
  --identities state/research/acquisition/20261016-q2-primary-cases-final-c/security_identities.json \
  --psky-covariance state/research/acquisition/20261016-psky-253-recovered/PSKY_253_covariance_levels.csv \
  --original-reference-assumption --output state/research/diagnostics/NEW_UNUSED_RUN
```

The current process now captures calculation times; provenance status strings or a nonempty evidence
list no longer qualify a current handoff. Late-downloaded, exact original SEC releases can be checked
against actual source bytes, original acceptance, context/entity, value and retrieval receipt.
The checker was executed on the real MMM original, not just synthetic fixtures. Its result is still
not a certificate for the entire feature dataset.

`pipeline/paper_manual_run.py` wires reviewed source acquisition, checks, snapshot, signal/replay,
export and independent import without a manually authored signal JSON. **It is not yet a complete
raw SEC-to-qualified-training-panel builder** and no real fully qualified source plan exists.
The exact outstanding row inventory is:
16,892 training rows (5,769 original historical / 11,123 SEC), 1,701 SEC validation rows and
362 SEC prediction rows, requiring 202,822 retained feature cells. 4,863 cached nominal trade dates
do not match the required actual execution session; labels need source-based recalculation/proof.
These rows are enumerated in `state/research/acquisition/20261016-q2-required-inputs-exact/`.
Training labels are checked at model-training cutoff, not incorrectly at their earlier feature cutoff.

The new `portfolio_consumer.ConsumerLedger` adds account/strategy-bound durable progress, reconciliation,
partial execution, crash recovery and explicit replacement semantics. Its 19 simulated-consumer tests
are technical evidence only; no broker adapter or external exactly-once guarantee is claimed.

The earlier dated live-readiness descriptions below are not the manual historical paper rule.
The manual path never requires all 500 issuers to be READY: evidenced nonpublication may exclude
an issuer; a missing cached published report or unresolved identity requires acquisition/review,
not an invented exclusion. No scheduler, broker connection, production order or automatic rebalance
has been enabled.

* A **real-data numerical Q2 diagnostic completed twice offline**: 362 forecasts, 76 selected
  securities, 11 sectors. Every sector actually used 40 training quarters (2015Q1–2024Q4) and four
  validation quarters (2025Q1–2025Q4). Inputs include original historical **and newer SEC hybrid
  rows**, not only the original files ending in 2017. The frozen run exports the exact rows/features.
  Both numerical passes match. This is today's cached Q2-period analysis, **not a historically
  admissible August-31 signal** and not historical return validation.
* Equal weighting completed numerically. Both covariance-dependent strategies explicitly failed:
  selected PSKY lacks 253 complete pre-cutoff daily levels. No substituted weights or forced
  removal of PSKY. Diagnostic: `state/research/diagnostics/20261016-q2-numeric-final-identities/`.
* Original cover/class matches increased to **502 securities / 499 issuers** within 503/500.
  C, MMM, NDAQ and RVTY are resolved. SMCI has an original August-11 release with 44 exact
  observations; it is **not** automatically excluded as unpublished. The release does not disclose
  period-end shares outstanding; weighted-average shares are not a replacement.
* Genuine primary SPY NAV/distribution TR now contains **5,755 NYSE observations and 93
  distributions, 2003-12-01–2026-10-15**, as a separate complete series without splicing.
  The acquired 1993-onward provider market history is not yet a certified 1995-onward TR comparison.
* **The genuine historical Q2 signal has not started**, and the 37-quarter continuation and
  certified continuous 1995-onward strategy valuation are not complete. Historical validation
  stays `NOT_VALIDATED`. Additional HTML comparison alone is not a blanket signal blocker;
  exact required feature/label versions and causal training inputs are.

Signal calculation now runs independently of the valuation cash ledger. Missing dividend pay
dates or a full 37-quarter valuation do not themselves prevent a source-qualified Q2 signal.
The reviewed SIGNAL bundle still needs causal training/validation, definitions, identities and
covariance evidence; it is not created by relabelling the diagnostic caches.

Manual commands, from this repository directory:

```bash
# Show the real future paper window; does not acquire data or start a scheduler.
.venv/bin/python -m pipeline.paper_quarter schedule --cycle 2026Q3

# Run an isolated, source-reviewed SIGNAL bundle twice offline (no portfolio DB).
.venv/bin/python -m pipeline.paper_signal_execution \
  --bundle REVIEWED_SIGNAL_BUNDLE --output NEW_SIGNAL_RUN --cycle 2026Q2
.venv/bin/python -m pipeline.paper_quarter from-signal \
  --execution-dir NEW_SIGNAL_RUN --output NEW_SIGNAL_RESULT.json
.venv/bin/python -m pipeline.paper_quarter research \
  --signal-result NEW_SIGNAL_RESULT.json --instrument-map VERIFIED_INSTRUMENT_MAP.json \
  --output state/research/target_portfolios
.venv/bin/python -m pipeline.paper_quarter validate --package EXPORTED_PACKAGE
.venv/bin/python -m pipeline.paper_quarter list

# Run the existing test suite, including synthetic end-to-end signal/package/import tests.
.venv/bin/python -m unittest discover -s tests -v
```

Uppercase paths denote required reviewed inputs, **not files already supplied**. No verified real
ISIN map or historically qualified SIGNAL bundle currently exists. The `regular` CLI accepts an
already computed, qualified signal and enforces its actual window; it does **not yet implement an
automatic acquire-all-data-to-qualified-signal command**. A historical or stale result cannot become
a current handoff by changing its filename. Full contract: [EXECUTION_INTERFACE.md](EXECUTION_INTERFACE.md).
Atomic packages include three CSVs only when all three strategies and identities succeed;
otherwise a failed strategy has an explicit `.FAILED.json`, not an empty or fabricated CSV.
Dashboard downloads verify integrity and never claim/process a portfolio.

The optimizer is explicitly versioned `annual-log-max-sharpe-v1`, labelled
**Rendite/Risiko – Max-Sharpe**, with an explicit 1.5% annual risk-free assumption. Forecast means,
covariance and risk-free rate share annual log units; the weighted-log objective is an approximation,
not an exact MATLAB reproduction or a conventional simple-return Sharpe claim. A zero-risk
denominator is an error, never a manufactured successful result.

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
```

`run_all.py` deliberately does not run the old price-based performance report. The report now requires raw prices, corporate actions, fixed target-weight inputs, and an explicitly verified total-return benchmark. The inherited adjusted-price and `^GSPC` archives do not establish those inputs.

For example, one sector or only feature coverage can be inspected with:

```shell
python pipeline/run_model.py --sector sector45 --features-only
```

Model fitting is computationally expensive and can take a substantial amount of time depending on the dataset and parallelization. Check that no other training process is running before starting another run.

## Backtest and paper-ledger tools

The performance engine accepts raw unadjusted close prices and explicit corporate actions. Run its deterministic synthetic demonstration with:

```shell
python pipeline/paper_performance.py --demo
```

This produces synthetic output only; it is neither an S&P 500 comparison nor a historical validation. A real input run requires:

```shell
python pipeline/paper_performance.py \
  --prices path/to/raw_prices.csv \
  --targets path/to/target_weights.csv \
  --actions path/to/corporate_actions.csv \
  --benchmark-csv path/to/benchmark.csv \
  --benchmark-metadata path/to/benchmark.json
```

Required CSV columns are `date,ticker,close` (one raw close per NYSE session and ticker), `strategy,date,ticker,target_weight`, `action_id,ticker,ex_date,kind,value,pay_date`, and `date,total_return_index`. The JSON benchmark manifest must identify either `sp500_gross_total_return` or the explicitly labeled `spy_total_return_proxy`, and provide source, identity, currency, dividend treatment, covered dates, retrieval time, verification method, and evidence. A manifest is a provenance assertion supplied by the operator; the program cannot independently establish the truth of external evidence.

Unverified metadata skeletons are in [examples/benchmark_sp500_gross_total_return.template.json](examples/benchmark_sp500_gross_total_return.template.json) and [examples/benchmark_spy_total_return_proxy.template.json](examples/benchmark_spy_total_return_proxy.template.json). Both intentionally set `verified` to `false` and contain placeholders; replace them only with provider-backed evidence before running.

The SQLite paper ledger can be initialized and inspected locally:

```shell
python pipeline/paper_portfolio.py --db state/paper_portfolio.sqlite3 init
python pipeline/paper_portfolio.py --db state/paper_portfolio.sqlite3 status
python pipeline/paper_portfolio.py --db state/paper_portfolio.sqlite3 backup \
  --output backups/paper_portfolio.sqlite3
python pipeline/paper_portfolio.py --db state/paper_portfolio.sqlite3 dashboard
```

Initialization creates three independent USD paper portfolios (Equal Weight, Mean-Var, Min-Var), each with its own starting cash and strategy ID. Repeating initialization with the same configuration is safe; a conflicting pre-existing database is not reset. The dashboard binds to `127.0.0.1` by default and is read-only.

For manual filing-readiness tracking, `create-cycle` takes an immutable expected-filing CSV with columns `filing_key,company_id,security_id,expected_period,filing_type,source`. Every expected company-period must occur once; missing filings do not reduce the denominator. `update-filing` records the receipt/facts validation outcome, and `set-model-status` records model readiness separately. These commands maintain auditable state, but do **not** fetch EDGAR updates, construct a historically valid S&P universe, generate model signals, select positions, execute due orders, or mark daily NAV.

No continuously running scheduler, automatic retry/lock service, daily pricing job, actual `predict_as_of` signal pipeline, or activated server process is included. Do not treat this ledger as a running forward test until those jobs and their external data feeds have been implemented, operated, and verified. In particular, the present README commands are not a substitute for the missing daily valuation and trading workflow.

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

The deterministic local tests can be run without Internet access:

```shell
python -m unittest discover -s tests -v
python -m py_compile code/ml_model.py fundamental_run_model.py pipeline/*.py tests/*.py
```

The tests cover sample accounting, benchmark validation, filing-readiness state, append-only ledger records, SQLite persistence, and selected time-series/PIT helpers. They are not a proof that all requested operating workflows are complete. The synthetic performance demo is run separately with `python pipeline/paper_performance.py --demo`; it explicitly does not validate any real historical period.

No real historical S&P 500 comparison is currently validated. The locally available adjusted-price/SPX archives lack the raw prices, complete corporate actions, a verified gross-total-return benchmark, and proven historical point-in-time membership/filing availability needed for a strict run. The 2017-to-present free-data extension also does not yet rebuild every historical feature set from SEC versions available as of each signal date. Do not infer a validated date range or excess return from the synthetic demo or the older report outputs.

## License and rights

This repository currently has no selected open-source license. The upstream GitHub repository does not expose a `LICENSE` file (checked 2026-10-03). GitHub's [Terms of Service, section D.5](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#5-license-grant-to-other-users) grant users a limited right to view and reproduce public repository content by forking it through GitHub. This platform-specific fork permission is not a general-purpose open-source license for redistribution outside GitHub. GitHub's [licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository) explains that, without a license, default copyright rules otherwise apply. Rights to third-party material such as WRDS/Compustat data and figures must still be checked before including them in a public fork.

Before publication, also review rights for WRDS/Compustat data, Yahoo Finance data, paper figures, and historical membership data. Yahoo's [Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) restrict automated collection and reuse absent permission; SEC EDGAR's [fair-access guidance](https://www.sec.gov/os/accessing-edgar-data) requires a descriptive User-Agent and reasonable request rates. These service terms do not themselves grant redistribution rights to downloaded data.

A license for original code can only be selected after those rights are clarified. No license is added here deliberately, and a license must not purport to grant rights to third-party material. The paper authors and data sources should be credited appropriately; this is not a substitute for legal advice.

## Publication status

This repository is a GitHub fork of the upstream project. Updates should be added as commits on this fork so the upstream relationship and original history remain intact. The fork inherits the upstream's files and history; a new commit must not add local datasets, caches, generated results, notebook outputs, or personal paths. Review the rights for inherited WRDS/Compustat data and figures before reusing them outside GitHub or redistributing them separately.

## Automatic virtual portfolio (reports → signal → orders → daily marks)

Modules: `pipeline/paper_live.py` (real adapters: SEC filings, sector models, Yahoo prices),
`pipeline/paper_runner.py` (orchestration, locking, retries), `pipeline/paper_store.py` (durable
ledger, schema v3, migration), `pipeline/portfolio_weights.py` (three weighting variants),
`pipeline/universe_asof.py` (membership at a past reference date), `pipeline/paper_dashboard.py`
(read-only status), `pipeline/paper_synthetic.py` (**synthetic** adapters; tests/demo only).
Configuration: `config/real_paper.json` (real). The synthetic demo has no configuration and its own
database.

### Strict separation
| | real paper tracking | synthetic demo / tests |
|---|---|---|
| CLI | `--env real` (default) | `--env synthetic` |
| database | `state/real_paper.sqlite3` | `state/synthetic_demo.sqlite3` |
| marker in DB | `store_meta.environment = real` | `synthetic` / `test` |
| adapters | `paper_live:real_adapters` | `paper_synthetic` |
| dashboard banner | `REAL PAPER TRACKING` | `SYNTHETIC DEMO / TEST - NOT REAL DATA` |

The runner refuses real adapters on a non-real database and vice versa; the CLI refuses an
`--env` that differs from the database marker.

The dashboard's live target-weight comparison is sourced only from a persisted depot signal.
Separately, it displays the hash-verified archived result from run
`20261014T195516Z-071b27` (50% feature-coverage threshold, eligibility gate enabled) as historical
model output. Those archived weights are not imported into the depot ledger, are not an active
signal, and did not create orders; historical validation remains `NOT_VALIDATED`.

The dashboard also offers a read-only research view for complete, hash-verified archived runs,
including their available feature-coverage thresholds, diagnostic/enforced modes, forecasts, and
portfolio weights. It keeps missing or infeasible optimizer weights empty and labels the latest
attempt separately from the latest complete run. Search, sector filtering, paginated stock rows,
and CSV export affect only the browser presentation; they do not write to the depot. A live
depot panel continues to show only persisted target weights and actual valuations. The research
view never promotes an archive to a live signal or historical validation result.

### Flow per cycle
1. `generate-cycle` derives the expected reports: universe members at `universe_as_of` (default: the
   quarter end, frozen in `state/expected_filings/<cycle>.json`; a different date is refused),
   reconstructed from the current list by reversing dated S&P changes in reverse chronological
   order (including later removals and reinstatements). One report per CIK; period and
   form come from the company's own filed 10-Q/10-K calendar (52/53-week calendars normalised,
   off-grid calendars via the prior-year anniversary). Reviewed exceptions live in
   `config/real_paper.json`: `period_overrides` (cited source) and `cik_successors` (documented
   legal successor; both filing histories are merged and deduplicated by accession, and financial
   data continuity is checked against overlapping values and the 8-K12B). If any period cannot be
   derived, **no cycle is created**, the cases are written to `unresolved_<quarter>.json`, and nothing
   is rebalanced. Nothing is estimated or excluded silently.
2. `check-filings` / `tick` record, per expected report, publication (SEC acceptance time),
   first-seen time and whether all required XBRL facts exist (`READY`).
3. Only when **100 %** of expected reports are accounted for as `READY` or individually evidenced
   `CORPORATE_ACTION_RESOLVED`, exactly one `predict_as_of(snapshot)` runs. A resolved corporate
   action is never labelled READY and never becomes a buy candidate. The snapshot uses only
   facts accepted by the snapshot timestamp (SEC acceptance times; same-day facts need proven
   acceptance) and prices up to the last closed NYSE session. Each cycle stores publication,
   first-seen, snapshot and model-completion timestamps (`cycle_timeline`, immutable).
4. The three portfolios use their own weights: `equal_weight` (capped 1/N), `mean_var`
   (max-Sharpe SLSQP) and `min_var` (minimum variance), all long-only with a 5 % cap, one-year
   covariance. Optimisation failures fail the step; there is no silent fallback to equal weights.
5. Orders are planned from actual holdings/cash and executed at the close of the first NYSE session
   opening after model completion; then daily corporate actions, fills and NAV/benchmark marks.

### Commands (real)
```bash
export DYNAMIC_STOCK_USER_AGENT="Your Name you@example.org"   # required by SEC; never commit
PY=.venv/bin/python
$PY pipeline/paper_portfolio.py init                         # once; DB state/real_paper.sqlite3
$PY pipeline/paper_portfolio.py migrate                      # older DB: verified backup, then upgrade
$PY pipeline/paper_portfolio.py generate-cycle --cycle-key 2026Q3 --quarter-end 2026-09-30
$PY pipeline/paper_portfolio.py check-filings --cycle-key 2026Q3
$PY pipeline/paper_portfolio.py tick                         # one locked pass (cron/launchd)
$PY pipeline/paper_portfolio.py scheduler --interval-seconds 900   # permanent operation (not enabled by default)
$PY pipeline/paper_portfolio.py dashboard                    # http://127.0.0.1:8765
$PY pipeline/paper_portfolio.py status
# synthetic demo (separate DB, never real data)
$PY pipeline/paper_portfolio.py --env synthetic demo
```
A `<db>.lock` file prevents concurrent schedulers; restarts never reset the depot or rebook
(deterministic ids + store uniqueness). `migrate` copies the database through the SQLite backup API,
verifies its integrity and only then upgrades in place.

### Historical validation vs. live depot
The dashboard shows `historical_backtest_validation` separately from `live_paper_depot`. A JSON status
field never validates anything: `VALIDATED` is recomputed from a historical-backtest run manifest
(`validation_manifest` in the config) that must carry all input hashes and, for each of the five
evidence categories, `verified: true` plus a local evidence file whose SHA-256 matches. No such
manifest is configured, so the status is `NOT_VALIDATED`.

### Documented assumptions (decisions that remain the user's)
* Benchmark: labelled **SPY-Vergleich** – SPY adjusted close (reinvested dividends), USD, base = session before the
  actual portfolio start; it is a proxy, not the S&P 500 total-return index. Holdings are valued at the raw close
  and dividends enter only as receivable/cash, so no dividend is counted twice.
* Dividends: ordinary cash dividends are booked as a receivable on the ex-date and become cash only on the
  documented pay date (Nasdaq dividend calendar, unofficial and without SLA). Unknown or inconsistent pay date
  → nothing is booked and the check is retried on later days (no early buying power). Remaining gaps: a pay date
  appears only after the announcement; a holding that changes between ex-date and a late-resolved pay date is
  skipped instead of guessed.
* Compustat history for model training: signal time = close before the trade date, label known at the
  next trade date close (conservative derivation in `_add_historical_timing`).
* The model's pending slot is the next quarterly rebalancing date; the virtual execution is the first
  session after model completion, which can differ from the label's start date.
* Restored index members (removed after the reference date) get their sector from Yahoo mapped to GICS
  codes when Compustat has none (approximation, marked in `data_source`).
* Spin-offs are never given their parent's financials; a new entity without history has no model
  history and is reported as such.

Tests: `.venv/bin/python -m unittest discover -s tests`. The real-data test in
`tests/test_paper_real.py` runs only if `DYNAMIC_STOCK_USER_AGENT` is set.

### Documented run with current data (2026-10-13)
Status: **real-data integration verified up to the readiness gate; no rebalance; continuous operation NOT enabled.**

| Step | Result |
|---|---|
| `generate-cycle --cycle-key 2026Q3 --quarter-end 2026-09-30` | 500 expected reports, 3 cycles (equal_weight, mean_var, min_var); manifest frozen in `state/expected_filings/2026Q3.json` (universe as of 2026-09-30) |
| `check-filings --cycle-key 2026Q3` | READY 15, WAITING_FACTS 1 (LEN), WAITING_FILING 484 → readiness 15/500, **no signal, no orders** |
| HONA / XOM / VYLR | HONA: user-supplied fixed 10-Q period 2026-09-26 (source not verifiable by the tool, HTTP 403); XOM: CIK 34088 → 2115436 with continuity check (9 overlapping values, 0 deviations); VYLR: not a member on 2026-09-30 |
| Still expected, not yet filed | HONA, XOM (new CIK), WBD and CTVA (both left the index on 2026-10-06; a filing may never come → user decision) |
| Historical backtest validation | `NOT_VALIDATED` (no `validation_manifest` with hashed evidence configured) |
| Diagnostic model dry run (`ModelSignalProvider.predict_as_of`, not stored, not a signal) | Completed in about 32 min after fixing `StepwiseAICRegressor` (scikit-learn tags); 5 sector models ran, 18 tickers skipped for missing history/required facts (e.g. ABNB, GS, HONA, V, VYLR). It used only the 15 ready reports and is therefore **not** a valid signal. |

The orchestrator only starts the model after 100 % readiness; the dry run above bypassed that gate on
purpose and wrote nothing to the database. The scheduler (`scheduler` command) is documented but has not
been started as a permanent service.

The [.env.example](.env.example) file can also be used as a reference. It is not loaded automatically, and no API credentials are required.

### Model-coverage changes (2026-10-14, readiness rule v2)
Diagnosis: `python pipeline/model_coverage.py diagnose|readiness|model-run --cycle-key 2026Q3`
(read-only, no signal, no database writes; report in `state/diagnostics/coverage_<key>.json`).

Causes found and fixed (with regression tests in `tests/test_model_coverage.py`):
* **Mapping**: revenue concepts `RevenueFromContractWithCustomerIncludingAssessedTax` and
  `RevenuesNetOfInterestExpense`, EPS `EarningsPerShareBasicAndDiluted` were not mapped.
* **Universal industrial required fields**: only NetIncome, Assets and StockholdersEquity are required; Revenue and
  shares are optional (missing → NaN, never 0). Readiness rule v2 (`readiness_rule` in the config): sector 40
  (financials) does not require Revenue; Visa (CIK 1403161) has its own override with the full list. Banks'
  total revenue is not derived from interest income.
* **Fiscal-quarter derivation bug** (`sec_edgar._discrete_quarters`): `fy`/`fp` describe the filing, so
  comparative facts overwrote and suppressed Q4 values (e.g. DUK Q4 2023 −478 M instead of 1 005 M), which broke all
  TTM and growth features for about 56 companies. Quarters are now derived from the period dates.
* **Price cache bug** (`prices.get_price_history`): an unstable sort kept stale NaN raw closes; caches with
  <90 % raw close are fully refreshed and fresh rows win.
* **Model change**: the feature set is chosen per sector from the historical training panel
  (`cache/sector_feature_sets.json`) and is identical for training and live rows; a live row needs
  `min_feature_coverage` (0.5) of those features, otherwise it is skipped as `LOW_FEATURE_COVERAGE`. Company
  history is no longer required in full; the pooled training window is independent of it.
* **VYLR**: the model run used the snapshot-date universe; it now uses the frozen cycle universe (2026-09-30).
* **WBD**: `CORPORATE_ACTION_RESOLVED` (SEC Form 25 and 8-K, 2026-10-06) is reported separately from received
  and open reports, is never READY and never a buy candidate; the gate still requires READY for every expected
  report. CTVA stays expected (index change only).
* CTVA/VYLR spin-off: the cached CTVA series shows no price discontinuity on 2026-10-01 and VYLR starts at 68.26;
  consolidated pre-spin financials of CTVA are not adjusted. Features of CTVA for post-spin signal dates are an
  **unresolved risk** and need a decision (exclude CTVA until its first post-spin 10-Q).

Historical backtest validation remains `NOT_VALIDATED` (no hashed evidence manifest).

The complete pipeline downloads data, builds tables, trains the models, and generates reports:

### Implemented state (2026-10-14, schema v5, readiness rule v3)
Status: **real-data diagnosis and integration verified up to the readiness gate; no signal, no orders (0), scheduler and automatic rebalancing NOT enabled; historical validation `NOT_VALIDATED`.**

* **Readiness (rule v3)**: expected reports (500 issuers for 503 securities; FOX/FOXA, GOOG/GOOGL, NWS/NWSA share a CIK)
  are shown as three separate groups in the dashboard: READY, evidenced special cases (WBD `CORPORATE_ACTION_RESOLVED`,
  never READY) and open. Timeouts, missing metrics or ordinary index exits never create an exception. The WBD
  resolution (Form 25 + 8-K, effective 2026-10-05/06) is stored idempotently in `filing_resolutions`.
* **Forecast eligibility**: every forecast is stored as `ELIGIBLE` or `DIAGNOSTIC_ONLY` (`forecast_records`).
  `paper_runner._validated_payload` rejects a signal that is not eligibility-enforced or has target weights for
  ineligible tickers (test `test_diagnostic_signal_cannot_create_weights`).
* **Timestamp handling**: SEC availability timestamps are parsed as UTC before calendar-session scheduling;
  timezone-aware and date-only observations use the same conservative UTC-date comparison.
* **iXBRL fallback** (`pipeline/ixbrl_fallback.py`, rule `ixbrl-fallback-v1`, config block `ixbrl_fallback`): values from original
  filings (accession, tag, context, unit, period, acceptance time, document SHA-256) fill only periods missing in
  companyfacts and only after acceptance. Traced end to end in `state/diagnostics/ixbrl_fallback_trace_2026Q3.json`:
  BKR (0.35→0.55), BRK-B (0.33→0.56) and ERIE (0.44→0.67) become model-usable (≥ 0.5); **GD stays 0.43** (revenue, net
  income, assets from the original 10-Q, but only diluted EPS is published – no basic EPS is guessed – and cost of goods is not
  adopted); APA, ARES (ambiguous/dimensioned revenue), SKYD, FDXF (no 10-Q) remain unsupported. Class assignment
  (ERIE/BKR = Class A, BRK-B = equivalent Class B) is a documented user decision.
* **Q4 derivation** `q-derive-v2`: annual minus nine-month only for additive items with comparable periods/units; never
  for EPS, share counts, ratios or balances. Later comparatives do not improve earlier data states.
* **Diagnosis suite** (`python pipeline/model_coverage.py suite --cycle-key 2026Q3`): first copies the selected
  database, manifests, SEC/price caches, feature panels, model source (`code/ml_model.py`), pipeline, tests and
  configuration into `state/diagnostics/runs/<run_id>/snapshot/`. It validates file hashes and the SQLite
  logical contents before running 50 % (diagnostic and enforced), 75 % and 100 % with network access blocked.
  A diagnostic-only optimization failure is recorded per weighting method; no equal-weight substitute is
  silently inserted for a failed mean/minimum-variance optimizer. The production provider still blocks the
  whole signal on optimizer failure.
  Every run exports the actual sector features, source/snapshot manifests, before/after hashes, thresholds,
  forecasts, readiness and a common artifact manifest (run ID, evaluation time, code hash and SHA-256 per output).
  A failed or changed-input run remains explicitly incomplete; it is not relabelled reproducible.
  Threshold comparison shows only coverage, sector distribution, forecasts and weights – **no statement on return quality**; the
  final test window must not be used to pick a threshold. The 50 % threshold is provisional.
* **CTVA** (`state/diagnostics/ctva_investigation_2026Q3.json`): remains in the expectation, **purchase block** continues.
  Separation effective 2026-10-01 (record date 2026-09-24); Vylor is the accounting successor, CTVA statements for the remaining
  company exist only as combined carve-out financials in an 8-K. The cache holds no raw price/adjustment factor; the history
  looks back-adjusted (pre-spin closes not on whole cents, post-spin all on whole cents) – a missing price jump proves nothing.
  Release needs (a) documented price/adjustment treatment, (b) documented scope of the fundamentals, (c) comparable features of
  the remaining company. The next 10-Q alone does not release.
* **Dividends**: receivable at the ex-date when entitlement and amount are known, cash only on the evidenced pay date;
  late-added pay dates do not double-book; a source outage never deletes known entitlements.
* **SPY**: `spy_benchmark` is a parallel buy-and-hold model depot (same capital, start fill, currency, costs only on actual fills,
  no rebalancing fills). SPY `adj_close` is an additionally labelled total-return approximation.
* **Dashboard**: disjoint report groups (`READY`, `WAITING_FACTS`, `CORPORATE_ACTION_RESOLVED`,
  `WAITING_FILING`, `INVALID_DATA`); stock weight comparison (benchmark / ML target / actual, benchmark weight descending)
  and sector bar comparison (full GICS names), data states, sources, cash, and a diagnostic-only banner with no targets shown.
  Actual stock weights use per-symbol quantity, raw close and market value persisted at exactly the same timestamp as NAV;
  legacy marks without that valuation evidence remain blank rather than being reconstructed from stale fill prices.
  In the all-stocks table, company names and tickers are green when their weight exceeds `1e-9` in the selected archived
  ML run, analysis variant, and strategy. This highlight also appears in the live view, but does not indicate a held position
  or an executable signal; zero-weight and forecast-only stocks are not highlighted.
  The **Veröffentlichungen** tab adds weekly publication-delay bars relative to official calendar quarter-end.
  It separates SEC 10-Q/10-K acceptance from verified first earnings releases, with company counts and sourced
  quarter-end issuer market-cap totals. The quarter selector also lists locally analysed historical quarters.
  The historical view pools up to twelve preceding quarters with their own membership and counts issuer-quarter
  observations, not distinct companies. Negative delays and delays of 91+ days are retained.
  Unknown dates/capitalizations are shown as missing, never as zero; recent quarters remain right-censored.
  Historical membership comes from cached snapshots; historical CIK mappings are not fully time-versioned.
  Neither 8-K event dates nor SPY holdings weights substitute for first release dates or market capitalization.
  Generate/update the read-only chart dataset explicitly after updating source caches:

  ```bash
  .venv/bin/python -m pipeline.publication_timing --history-quarters 12
  ```

  This command uses local inputs only, writes `state/diagnostics/publication_timing.json` with source hashes,
  and never changes portfolio databases, model archives or scheduler state. Opening the dashboard does not fetch
  data or regenerate this report. The generation timestamp and missing coverage remain visible.
  Optional independently verified sources can be supplied in `state/sources/publication_timing_evidence.json`:

  ```json
  {
    "schema_version": 1,
    "earnings": [{
      "cycle_key": "2026Q2", "company_id": "123", "period_end": "2026-06-30",
      "published_on": "2026-07-30", "first_publication_verified": true,
      "source_url": "https://issuer.example/verified-release",
      "observed_at": "2026-10-16T00:00:00Z"
    }],
    "market_caps": [{
      "cycle_key": "2026Q2", "company_id": "123", "as_of": "2026-06-30",
      "scope": "issuer_all_share_classes", "value_usd": 1000000000,
      "source_url": "https://provider.example/verified-quarter-end-cap",
      "observed_at": "2026-10-16T00:00:00Z"
    }]
  }
  ```

  This is a schema example, not real data. Earnings dates must be actual first releases, not estimates, and refer
  to the matched fiscal report period. Publication dates use the US Eastern calendar. Market caps must cover all
  issuer share classes, in USD at official quarter-end; current cap, public float and adjusted-close/share products
  do not satisfy this requirement. Invalid or duplicate evidence fails explicitly.
  `BRK.B`/`BF.B` provider symbols map explicitly to `BRK-B`/`BF-B` model symbols. Benchmark weights: official SPY holdings file
  (State Street, 2026-10-13, 506 positions, SHA-256
  `aa260f9d…741d7c`, archived in `state/sources/`) – an ETF holding on that date, not the index, not necessarily the portfolio start.
  Cash and non-equity SPY instruments are shown separately from equity weights; raw holdings totals are not normalized.
  The read-only page now also shows company name, benchmark-relative actual deviation, top-stock bars, portfolio NAV/cash/receivables,
  a NAV comparison on timestamps shared by ML and SPY portfolios from their common first daily mark, the additional explicitly
  labelled SPY `adj_close` proxy, and a 60-second display refresh. Explicit real `tick`/`scheduler` runs refresh archived SPY
  holdings at most daily, with persistent retry status; merely opening the dashboard does not make a network request. The
  scheduler remains off unless its command is explicitly started.
* **HONA**: 8-K 0002089271-26-000020 (filed 2026-08-05, SHA-256 `04635e01…4b85f6d`, archived in `cache/filings/`) proves the quarter end
  2026-09-26, not that the Q3 10-Q exists.
* **Validation manifest** (`python pipeline/validation_manifest.py build` → `state/validation/validation_manifest.json`): records code
  hash, configuration, input hashes, periods and limitations; all five evidence categories are `verified: false` (raw price basis,
  corporate actions, historical membership, point-in-time labels, benchmark provenance) and no walk-forward result exists. Result:
  `NOT_VALIDATED`. Configure it via `validation_manifest` in the config to show it in the dashboard.
* **Schema v6 migration**: daily marks retain immutable per-symbol quantity, raw-close price and market value for
  same-time dashboard weights; `facts_ready_at` records the first time required facts became usable, independently of
  filing publication and first observation. Upgrade an existing v1–v5 store explicitly with
  `python pipeline/paper_portfolio.py --env real --db state/real_paper.sqlite3 migrate --backup-dir state/backups`.
  The migration verifies a SQLite backup before the additive schema change; prior depot ledgers and marks are retained.

Remaining external prerequisites: outstanding Q3 reports as enumerated in the latest `readiness.json`, an authoritative
dividend-pay-date source, CTVA release evidence, any unresolved issuer/class-specific filing ambiguity, a real walk-forward
validation, and an explicit decision on enabling the scheduler. Readiness counts are time- and run-specific; do not treat this
static list as a current report count.

### Consistency correction and reproducible run (2026-10-14)
* **Cycle period rule correction**: a `2026Q3` cycle denotes the calendar quarter July–September, not a fiscal year-end quarter. The old expectation rule (`window-pm45d-v1`, ±45 days) could assign period ends after the cycle date (e.g. 2026-10-31 for cycle key 2026Q3). The implemented rule is now **`calendar-quarter-v2`**: actual period end in `(previous calendar-quarter end + 7 days, current quarter end + 7 days]`; the seven-day tolerance is only for 52/53-week or off-grid fiscal calendars. The issuer's reported 10-Q/10-K period end and its later publication/availability date remain distinct.
* **Applied correction**: `python pipeline/paper_portfolio.py --env real correct-periods --cycle-key 2026Q3 --quarter-end 2026-09-30 --apply`.
  The original frozen manifest remains archived as `state/expected_filings/2026Q3.window-pm45d-v1.json`; the active manifest supersedes it and the full delta is in `state/expected_filings/2026Q3.period_correction.json`.
  A, COO and MDT changed from expected period `2026-10-31` to `2026-07-31`.
* **Data acquisition separated from diagnosis**: acquisition logs are retained separately; a `suite` consumes only the frozen input copy and blocks network access. Hashes are made before the run, not retroactively presented as hashes from an earlier run.
* **Prior run classification**: `20261014T135944Z-fdf707` is preserved as a restricted diagnostic, not a reproducible offline run: its input manifest omits historical sector panels, `code/ml_model.py` and the persistent database. Its `forecast_comparison_35_vs_32.json` filename is also inconsistent with its contents, which record 35 diagnostic and 35 enforced forecasts (35 in both; no differing or one-sided tickers). Do not cite that run as evidence for 32 forecasts.
  The separate earlier run `20261014T121703Z-a8a9ce` did record 35 diagnostic and 32 enforced forecasts: A, COO and MDT were diagnostic-only because the superseded period rule expected 2026-10-31 while the rows used 2026-07-31. The later `20261014T180910Z-2b1fd6` has 58 diagnostic records.
  `state/diagnostics/clarifications/forecast_run_reconciliation_20261014.json` compares the 35-record run with the 58-record run per security, including CIK, periods, current accession/availability, prior forecast/readiness state, and code-hash differences. For all 24 newly forecast securities the old captured filing status was `READY`, but they were not predicted because there was no row in the pending trade-date slot; the later run has each as `READY` with its accession and publication timestamp. The current `_align_cycle_reports` path moves each available expected-period report into a shared next slot, which directly addresses that exclusion reason. However, the old run did not archive its source tree or a consistent DB snapshot, so an exact paired replay cannot prove this was the sole cause; the per-ticker reconciliation keeps unavailable old accession times null rather than guessing. SMCI is the one removed record and remains excluded from the current cycle because its available period is not the expected period. The old run folders remain unchanged.
* **Quarter-cycle preparation**: a real `tick` or explicitly started `scheduler` prepares the current cycle when no predecessor exists, and only prepares a successor after all strategy portfolios in the prior cycle are `EXECUTED`. The frozen manifest and unique cycle keys make retries/restarts idempotent. This does not start the scheduler or submit broker orders.
  `20261014T173710Z-6375c2` additionally exposed that a 100 % feature threshold leaves too few names for the capped full-investment optimizers; it is retained as an incomplete diagnostic, not a completed threshold comparison.
* **Completed immutable suite**: `20261014T180910Z-2b1fd6` contains the frozen 2,334-file input snapshot and verified SQLite backup, no network attempts, unchanged before/after manifests, actual per-threshold feature sets, and a complete artifact SHA-256 manifest. It reconciles 503 securities to 500 issuers and reports 59 READY, 1 WAITING_FACTS, 1 evidenced `CORPORATE_ACTION_RESOLVED`, and 439 WAITING_FILING. The 50 %, 75 %, and 100 % runs produce 58, 48, and 14 forecasts; at 100 %, mean/minimum variance are explicitly unavailable because only six names meet the optimizer's 20-name minimum. All run weights remain diagnostic only because the cycle readiness gate is not satisfied. The 35-to-58 record-level comparison identifies 34 shared symbols, SMCI only in the old run, and 24 only in the new run.
* **Delivery**: the 2026-10-14 baseline report and archive describe run `20261014T180910Z-2b1fd6`; they are historical artifacts and are not overwritten by later verification. The latest full regression result and any later reproducibility archive are recorded separately. Scheduler remains disabled and historical validation remains `NOT_VALIDATED`.
* **Status groups (disjoint, sum=500)**: READY, WAITING_FACTS, CORPORATE_ACTION_RESOLVED, WAITING_FILING are exported and shown consistently in diagnosis and dashboard.
* **Historical validation** remains `NOT_VALIDATED`.

### Signal-cohort and resolution timeline corrections (2026-10-14)
* A `CORPORATE_ACTION_RESOLVED` row contributes its immutable `filing_resolutions.recorded_at` to the common
  information cut-off, but never supplies a fabricated filing publication timestamp. `published_at` remains the
  maximum actual publication time among received filings. The cut-off is the latest actual first-seen time or
  resolution-record time.
* For a live signal, the provider selects each issuer's latest expected-period row that was available by the
  shared snapshot. These rows are aligned into one common future prediction slot; a past historical quarterly
  slot can no longer cause an already-published current-cycle report to disappear from that cycle's signal.
  The actual virtual execution is still scheduled by the runner after model completion.
* Regression coverage now includes special-case evidence through signal persistence and a two-issuer case in
  which an earlier report's historical trade slot has passed while a later report remains current. This does not
  resolve the other audit findings about dashboard, historical input freezing, or run-all cycle automation.
  Scheduler remains disabled; no real signal was generated by these tests.
* **Step 3 follow-ups**: the optimized strategies now require at least 20 usable names under the 5% per-name cap;
  otherwise optimization fails explicitly rather than quietly returning a near-zero equity allocation. Equal
  weighting retains its documented residual cash behavior. The SPY benchmark creates only its initial buy order;
  later cycles do not rebalance or re-buy it, including after process restart. Pending dividend claims are
  resolved and paid independently of current holdings, so a sale before the pay date does not strand a claim.
  These are regression-tested with a <20-name optimization case, two SPY cycles plus restart, and dividend payment
  after the final position is sold. They are synthetic/function tests, not return validation.
* **Point-in-time / READY follow-up**: SEC `acceptanceDateTime` is now checked against the full UTC snapshot
  cutoff for every filing, including filings whose `filingDate` is the previous UTC calendar day. The cutoff is
  no longer used to backdate a fact's availability. If the accession has no acceptance timestamp, availability
  is conservatively set to the close of the next NYSE session and the fact is excluded until that timestamp.
  `READY` now requires every configured required fact to have a numeric finite value, a currency unit, the
  matched filing accession, and the matched report period end. The original-filing iXBRL fallback must satisfy
  the same checks using its recorded currency unit, context ID, and period; a null fact or comparative from
  another period cannot satisfy the gate. Regression tests cover an overnight UTC acceptance after the cutoff,
  a null current fact, an out-of-period comparison, and valid versus invalid fallback facts. These changes do not alter the 100%
  report-completeness gate or turn diagnostic forecasts into production signals.
* **Late dividend discovery and correction recovery follow-up**: a late-discovered cash dividend now queries ledger
  holdings at the prior NYSE close before its ex-date, even if the position has since been fully sold. The recorded
  entitlement is paid once on its evidenced pay date; this is tested with a sale before discovery and a repeat call.
  Expected-period correction takes the same database lock used by the scheduler, retains a pending recovery record,
  rechecks the aggregate filing status before reopening, and only reports "already corrected" when the database,
  active manifest, archived predecessor and audit artifact agree. Tests inject failures before DB commit and after DB
  commit/before manifest publication, then resume. No signal/order was generated by these tests.
* **Offline input coverage follow-up**: the immutable input manifest now includes all of `Data/`, including the
  historical sector training panels and root-level fundamental tables. Offline universe, historical-membership,
  SEC-ticker and S&P-change readers use the frozen cache contents irrespective of file age; missing cache files raise
  explicit errors. A regression test uses stale mtimes and confirms zero attempted network requests. This does not
  retroactively complete the older 135944 run or prove a new two-run model replay.
* **Final offline verification (2026-10-14)**: two independent runs on the same immutable 2,344-file snapshot
  completed (`20261014T195510Z-3449b1`, `20261014T195516Z-071b27`). Both produced 58 eligible forecast records;
  the enforced 50% run contains no diagnostic-only names in its target weights. Inputs and code fingerprints match,
  as do all substantive threshold/model outputs; only evaluation/generated timestamps and run IDs differ in the
  coverage/reconciliation metadata. This is an offline diagnostic, not a live signal or a historical return
  validation. The regression suite reports 110 tests passed and one network integration test skipped because no
  SEC User-Agent/contact was configured. Scheduler and orders remain disabled, and historical validation remains
  `NOT_VALIDATED`. See [the final audit report](state/diagnostics/FINAL_REPORT_20261014T195510Z.md) and the
  separately delivered audit ZIP for run outputs, logs, dependencies, and the frozen snapshot.

### Historical continuation and total-return comparison (`paper_fixed_lag-v1`)

This is a separate **research** variant, not a relaxation of the live 100% readiness gate.
No research command submits orders, enables the scheduler, migrates the production database,
or turns the October Q2 diagnostic into a historical September signal. Status remains
`NOT_VALIDATED`. A successful synthetic test or reproducible evidence audit is not a successful
real-data historical model run.

From the project directory:

```bash
# Inventory available hybrid panels and source evidence, freeze inputs, audit twice offline:
.venv/bin/python -m pipeline.fixed_lag_research

# Read-only dashboard using the existing temporary serving copy, not the production database:
.venv/bin/python -m pipeline.paper_dashboard \
  --db /tmp/paper-dashboard-8767.sqlite3 --host 127.0.0.1 --port 8767
```

Open `http://127.0.0.1:8767/?cycle_key=2026Q2#performance`. The existing archive quarter
selector and publications tab are preserved. The performance tab offers common-period rebasing,
dates/presets, index-100/cumulative-return/drawdown modes, cursor-centered wheel zoom, dragging,
joint hover, visible-series switches and CSV export. Instrument search uses only the reviewed
local registry: ambiguous ISIN/WKN/ticker/name/index matches stay separate. Unknown identifiers
are not heuristically converted or replaced by price series.

Historical stock/sector views have a separate **Gewichtsquelle / Historisches Datum** control.
With a reviewed simulation selected, both views use its exact dated holdings and historical
sectors; they never substitute today's benchmark or the old diagnostic weights. Unavailable
dated benchmark weights remain unknown, not zero or renormalized. Live/depot information
remains separate. No live execution means **noch nicht investiert**.

#### Causal research conventions and deliberate deviations

* Q1/Q2/Q3/Q4 execute on June/September/December/March 1, moved to the next NYSE session.
  Information closes at the preceding NYSE session close; execution uses the following fixed
  session's close. Intraday timing is an explicit assumption, not an established paper fact.
* Q2 2026 therefore uses **2026-08-31 20:00 UTC**, execution **2026-09-01 20:00 UTC**.
  The October-cutoff archive is retained only as a later diagnostic.
* The original archived last model date is June 1, 2017; the new continuation starts
  September 1, 2017: 37 decisions through Q2 2026. Paper end values are never spliced into
  the new NAV. The paper NAV handoff/overlap remains unverified.
* Historical features must be available at their own decision cutoff. Labels must mature by
  the current cutoff. This requires a decision-quarter maturity gap before four complete
  validation quarters and 16–40 contiguous training quarters, checked again after cleaning.
* Training-only factor removal above 5% missingness, complete cases, no imputation; scaling
  is fitted on training only. The paper's unspecified stock-deletion heuristic is not claimed
  identical. Five Python model families use sector validation MSE; their deterministic
  sklearn settings are **not** the original R/caret hyperparameter search.
* Selection is `ceil(20% of eligible names)`, descending forecast then security ID. Ties and
  the old quantile-`>=` selection count are exported. Intercept-only/regularized models can
  genuinely yield equal predictions; ranking is not evidence of predictive information.
* Expected quarterly log returns are annualized by four; covariance uses 252 daily log-return
  observations from the last 253 actual NYSE levels through cutoff. Risk-free return is
  `log1p(0.015)` annually, an explicit fixed-rate assumption, not a historical yield curve.
  Target/covariance return definitions must match. Gross-TR targets are supported only as an
  explicit deviation from the paper's price target, not silently as an exact replication.
* Long only, full investment, maximum 5% per name, hence at least 20 selected names.
  Optimizer failures have explicit errors and no substitute weights or silently skipped
  continuous-performance quarters.
* The existing share/cash simulator carries holdings, ex-date dividend receivables,
  pay-date cash, verified actions/delistings and actual turnover across quarters.
  Gross and net (10bps per trade side) are separate. The prior-close cash baseline preserves
  initial purchase costs; the portfolio is not reset each quarter. Fractional shares and
  zero cash interest are explicit research assumptions.

#### Execute a reviewed historical bundle

The existing cache is not automatically certified as a historical source bundle. First supply
and review the missing source evidence. Then:

```bash
.venv/bin/python -m pipeline.paper_history_execution \
  --bundle /absolute/path/to/reviewed-bundle \
  --output state/research/executions/unique-run-id --start 2017Q2 --end 2026Q2
# For a standalone Q2 decision, set both --start and --end to 2026Q2.
# Add --publish to import the replayed research curves/weights into the dashboard registry.
```

The bundle has exactly these hashed input files, declared in `bundle_manifest.json`
(`schema_version: 1`, `review_status: SOURCE_REVIEWED`, `files: {filename: sha256}`):

| File | Required content |
|---|---|
| `panel.csv` | `security_id,cycle_key,gsector,feature_available_at,reconstruction_status,y_return,label_known_at,label_source_ref,reconstructed_at,period_end,expected_period,period_rule,filing_form,facts_ready_at,acquired_at` plus all 20 `PAPER_FEATURES` columns |
| `membership.csv` | `security_id,cycle_key,gsector`; omitted prediction securities additionally need reviewed `exclusion_status,exclusion_reason` |
| `lineage.json` | Keys `cycle:security`; each nonmissing feature maps to its component list, each with `accession,value_version,source_url,published_at` |
| `raw_prices.csv` | Ascending unique `date` index and security-ID columns, raw unadjusted as-traded prices; no dividend adjustments |
| `covariance_levels.csv` | Same market-panel shape; source-reviewed levels matching the target return definition; never filled |
| `corporate_actions.csv` | `action_id,ticker,ex_date,kind,value,pay_date`; empty only with reviewed complete no-action coverage |
| `definitions.json` | Reviewed definitions and HTTPS `sources`, as specified below |

Required definitions: `currency=USD`, `raw_price_basis=unadjusted_as_traded`,
`action_coverage=splits_dividends_delistings_verified`, `historical_membership=quarter_end_verified`,
`historical_sector=point_in_time_verified`, `identity_scope=security_share_class_time_versioned`,
`source_transition=factor_definitions_overlap_reviewed`,
`simulation_dividends=receivable_ex_date_cash_pay_date_until_rebalance`.
Use either `target_definition=forward_execution_split_adjusted_price_log_return` with
`covariance_definition=split_adjusted_price_log`, or the explicitly different gross-TR pair
`forward_execution_gross_total_return_log` / `gross_total_return_ex_date_log`.

`calendar-quarter-v2` and the exact issuer-specific `expected_period` are mandatory; 10-K
versus 10-Q is explicit. Timestamps must have timezones. Acquisition/reconstruction times
are actual nonfuture times. Unknown operational `facts_ready_at` stays empty. Prediction rows
need no future return; training/validation rows do need completed, sourced labels.
Missing historical sectors, unexplained membership omissions, later publications, insufficient
windows and missing held prices fail explicitly.

The executor freezes inputs/code/configuration, runs independently into `primary/` and
`replay/`, compares output hashes, verifies source inputs and blocks network connections.
Exports include actual training/validation rows, feature sets, MSE/forecasts/ties/exclusions,
target weights, daily NAV/shares/cash/receivables/actual weights/trades, and importer-ready
comparison CSV/metadata. `execution_summary.json` distinguishes failed and completed runs,
including optimizer errors. Snapshot hashes and model replay are not historical validation.

#### Reviewed total-return CSV import

```bash
.venv/bin/python -m pipeline.return_comparison --csv /path/levels.csv --metadata /path/metadata.json
.venv/bin/python -m pipeline.return_comparison --search US0378331005
```

CSV columns are exactly `date,value,valued_at`: positive finite levels, unique ascending
completed valuations, explicit timestamp timezones and no missing declared sessions. For
`NYSE_close_UTC`, actual session closes including early closes are checked. No forward fill,
pre-inception values, implicit FX conversion or guessed dividends.

Metadata requires `instrument_id,name,instrument_type,identifiers,share_class,exchange,currency,
return_type,dividend_tax_basis,reinvestment_rule,cost_basis,source,source_url,valuation_convention,
calendar,review_status,evidence_files,observed_at`. Types are `stock,etf,index,strategy,paper_reference`;
identifiers are explicit lists keyed by `ticker,isin,wkn,index_id,strategy_id`. ISIN checksums and
WKN syntax are validated, but syntax is not issuer identity verification.
`evidence_files` lists source-definition files relative to the metadata (`path,sha256`).
Only `USD`, `before_personal_tax`, `SOURCE_REVIEWED`, and documented
`gross_total_return,etf_total_return,strategy_total_return` are accepted.
`calendar=EXPLICIT` additionally needs an exact `expected_sessions` list.

Reinvestment is explicitly `ex_date`, `pay_date` or `cash_until_rebalance`; incompatible rules
cannot be strictly compared. Portfolio receivables/pay-date cash are not silently treated as
analytical ex-date reinvestment. ETFs must document embedded fees/internal taxes in `cost_basis`;
the importer does not deduct them again. Optional `weight_history_file,weight_history_sha256`
add exact dated holdings; every nonmissing benchmark weight needs a dated `benchmark_source`.
The immutable imported JSON and sidecar hash live under `state/research/comparison/`.
EUR stays unavailable until sourced historical FX exists.

#### Public-source follow-up, 2026-10-16

The initial audit above is an immutable earlier assessment, not the latest acquisition status.
SEC access was successfully tested with the user's real, session-only HTTP User-Agent.
It is neither stored nor supplied to other providers. No licensed provider is assumed necessary.
Network acquisition and offline evidence review remain separate from model execution.

Actual acquired/reviewed evidence:

* 500 issuer value catalogs and 503 Q2 securities; 498 original cover-page ticker/entity matches.
  C, MMM, NDAQ and RVTY still require historical class/cover review. SMCI's expected original
  filing was accepted at 22:21:23 UTC on August 31, after the 20:00 UTC information cutoff.
* 14,318 current-period or complete-TTM field-component versions were checked. 8,428 match
  original numeric facts **and context entity IDs**; 5,890 require additional original documents.
  The earlier 17,543-component inventory included incomplete-TTM candidates and is not the
  admissible feature-dependency count. 1,055 incomplete TTM chains are exported separately;
  their stale candidates are not used as current metrics. These are not 498 certified feature rows.
* Dated public sector snapshots cover 36/37 decision quarters; the 2023Q1 file labels
  subindustries as `Sector` and is rejected. Q2's HONA/CAG discrepancy was reconciled with
  the original S&P effective-date announcement: HONA joins June 29; CAG leaves June 30.
  This yields 503 dated Q2 sector observations, **not** verified historical intervals for 37 quarters.
* 503 stock chart sources and 29,884 provider action events were acquired. This does not
  certify raw as-traded prices or complete corporate actions. AVB, EA, EQR and WBD sources
  currently contain only one daily observation each; 43 null closes occur in two sources.
  Apple demonstrates a real unit-basis difference: the issuer's May 2017 dividend is
  $0.63, versus $0.1575 in the provider series (the subsequent 4:1 split basis).
  Missing pay dates, spinoffs and delisting settlements are not invented.
* The dashboard registry now contains a genuine State Street **SPY NAV total-return proxy**:
  2,293 completed daily observations, August 31, 2017 through October 15, 2026, and
  37 explicit issuer distributions. It analytically reinvests at ex-date NAV; it is neither
  exchange execution, a pay-date cash ledger, the official S&P total-return index nor a strategy.
  An independent provider check finds zero SPY split events and all 37 distribution dates,
  but rounded/different amounts (maximum difference $0.003431); primary issuer amounts remain authoritative.

The admissible component cross-check and its independent offline replay agree. No historical
August-cutoff model run has started, no strategy weights/NAV have been fabricated, and
historical validation remains `NOT_VALIDATED`. Historical own-cutoff training/validation,
source-definition overlap, security intervals and full market/action coverage remain open.
The old October diagnostic is not reclassified as a September decision. Production databases,
existing Q2/Q3 manifests, scheduler and orders remain unchanged.

Missing filing evidence is **not** an all-500-READY gate for historical research.
Each security is classified separately: an evidenced publication after cutoff needs a reviewed
paper-rule exclusion (including checking earlier original releases); a report published before
cutoff but not acquired requires acquisition; unresolved identity/value provenance requires
targeted review. Absence from a cache or incomplete submissions history never proves
nonpublication. `pipeline.filing_case_review` exports these cases without creating automatic
exclusions. A filing identity match also does not certify all model features.
The historical executor already permits reviewed `NOT_PUBLIC_BY_CUTOFF` exclusions and
rejects acquisition/review gaps disguised as exclusions. A synthetic integration test verifies
that seven reviewed exclusions still allow the remaining universe to execute and replay.
Sector coverage, training requirements and optimizer constraints remain unchanged; live
production readiness is not modified.

Evidence outputs are under `state/research/acquisition/`; the latest hash-verified progress
pointer is `latest.json`. The dashboard displays it separately from the frozen old audit.
All commands below run from this project directory and require a new output directory:

```bash
# Offline original identity review: no SEC contact or network needed.
.venv/bin/python -m pipeline.filing_identity_evidence \
  --review-originals state/research/acquisition/20261016-q2-originals-01 \
  --output state/research/acquisition/NEW-IDENTITY-REVIEW

# Offline current/complete-TTM original-value and issuer-context proof.
.venv/bin/python -m pipeline.reconcile_historical_evidence \
  --values state/research/acquisition/20261016-q2-values-01 \
  --originals state/research/acquisition/20261016-q2-originals-01 \
  --identities state/research/acquisition/20261016-q2-identity-review-final \
  --output state/research/acquisition/NEW-COMPONENT-REVIEW

# Optional new primary SPY acquisition (network); --publish explicitly imports the proxy.
.venv/bin/python -m pipeline.spy_primary_comparison \
  --output state/research/acquisition/NEW-SPY-REVIEW --publish

# Read-only dashboard; this is not permanent scheduler operation.
.venv/bin/python -m pipeline.paper_dashboard \
  --db /tmp/paper-dashboard-8767.sqlite3 --host 127.0.0.1 --port 8767
```
