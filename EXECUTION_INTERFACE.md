# Execution interface: paper_fixed_lag target portfolios

Status: interface v1 (`paper-fixed-lag-handoff-v1`). The model is **NOT_VALIDATED**. Packages are
complete target portfolios, **not orders**. No broker/wikifolio API is assumed or implemented.

Code: `pipeline/target_portfolio.py` (export, strict import, legacy claim ledger),
`pipeline/paper_quarter.py` (manual CLI), tests: `tests/test_target_portfolio.py`.
The resumable consumer contract is implemented separately in `pipeline/portfolio_consumer.py`;
an old claim by itself is not a complete external execution lifecycle.

## Schedule (America/New_York, NYSE calendar, DST-aware)

| Cycle | Nominal execution | Information cutoff | Simulated execution / valid until |
|---|---|---|---|
| Q1 | 1 June | close of the previous NYSE session | close of first NYSE session on/after nominal date |
| Q2 | 1 September | " | " |
| Q3 | 1 December | " | " |
| Q4 | 1 March (next year) | " | " |

2026Q2: cutoff 2026-08-31T20:00Z (16:00 EDT), execution 2026-09-01T20:00Z.
2026Q3: cutoff 2026-11-30T21:00Z (16:00 EST), execution 2026-12-01T21:00Z.

**When to start the manual run:** after the information cutoff (`start_not_before`) and no later than
`recommended_start_by` (= execution − 4 h). Show the exact times with
`python -m pipeline.paper_quarter schedule [--cycle 2026Q3]` (UTC, New York, Berlin). A run started
before cutoff is refused; a run after execution is refused (or, with `--allow-stale-diagnostic`,
published as `STALE`). Missed runs are never backdated. Entering outside the paper schedule needs a
separate, explicitly named strategy variant.

## CLI (exit 0 = ok/handoff allowed, 2 = invalid/refused, 3 = intact but not handoff-eligible)

```
.venv/bin/python -m pipeline.paper_quarter schedule [--cycle 2026Q3]
.venv/bin/python -m pipeline.paper_quarter research --signal-result S.json --instrument-map M.json --output DIR
.venv/bin/python -m pipeline.paper_quarter regular  --signal-result S.json --instrument-map M.json --output DIR [--allow-stale-diagnostic]
.venv/bin/python -m pipeline.paper_quarter validate --package PKG [--strategy NAME] [--ledger L] [--accept-unvalidated-model]
.venv/bin/python -m pipeline.paper_quarter claim    --package PKG --strategy NAME --ledger L --consumer ID [--accept-unvalidated-model]
.venv/bin/python -m pipeline.paper_quarter from-history --execution-dir RUN --cycle 2026Q2 --output S.json
.venv/bin/python -m pipeline.paper_quarter from-signal  --execution-dir RUN --output S.json
.venv/bin/python -m pipeline.paper_quarter list --root state/research/target_portfolios [--ledger L]
```
`regular` always uses the real clock; the signal's `cycle_key` must be the cycle whose window
contains now. `from-history` adapts a `paper_history_execution` output and is always `historical_only`.

## Inputs

**Signal result** `paper-signal-result-v1` (JSON, produced by the signal executor; valuation is not required; `model_rules` required, see Model labelling):
`format_version, cycle_key, rule_version (paper_fixed_lag-v1), information_cutoff_utc,
scheduled_execution_at_utc` (must equal the schedule), `signal_computed_at_utc` (≥ cutoff),
`code_id, data_id, config_id, data_check_status ("PASSED" for handoff), historical_validation_status,
base_currency ("USD"), synthetic_fixture (bool), historical_only (bool), restrictions [str]`, and
`strategies` with exactly `equal_weight, min_variance, mean_variance`, each
`{"status":"COMPLETE","weights":{security_id: float fraction},"error":null}` or
`{"status":"FAILED","weights":null,"error":"reason"}`.
Internal names map `min_var → min_variance`, `mean_var → mean_variance`.

**Signal executor adapter** (`signal_result_from_signal_execution`, CLI `from-signal`) takes the sealed
run root of `paper_signal_execution.execute_signal` and trusts no flag or file on its own:
- `RUN/artifact_manifest.json` is required. The set of files under `RUN` (excluding the manifest) must equal its keys
  exactly; every SHA-256 must match. Symlinks, absolute paths, `..` and non-hex hashes are rejected. Any file
  (signal, `rule_config.json`, `execution_summary.json`, snapshot, code) changed or added after sealing fails.
- `primary/` and `replay/` must contain the same relative files with identical hashes (incl. `signal.json`);
  `model_replay_identical` must be a bool equal to that actual comparison. Any replay difference is rejected
  (no exportable mismatch state).
- `execution_summary.json`: `status` `COMPLETED_NOT_VALIDATED|PARTIAL_STRATEGY_ERRORS`, same `cycle_key`;
  `data_manifest` must equal the hashes of `snapshot/`; `snapshot/bundle_manifest.json` must be
  `SOURCE_REVIEWED`, scope `SIGNAL`, and match the snapshot; `code_manifest.json` must equal the `code/` hashes and
  the summary `code_manifest`.
- `primary/signal.json`: `cycle_key, information_cutoff, execution_at` checked against the schedule,
  `execution_orders` must be `false`, `strategy_results` exactly the three canonical names with `SUCCESS|ERROR`.
  `ERROR` becomes `FAILED`, and any weights in it are discarded.
- Ids are derived, not taken from the summary: `code_id` = SHA-256 of `code_manifest.json`, `data_id` = SHA-256 of
  canonical `data_manifest`, `config_id` = SHA-256 of `rule_config.json` (embedded as `model_rules`).
- `signal_computed_at_utc`: `computed_at_utc` from the signal if present (`signal_computed_at_source:
  "signal.json"`). New executors capture this in the calculation process and seal it in the summary;
  actual primary/replay completion times are separate. Legacy files may retain a labelled mtime for
  historical inspection only; mtime never qualifies a new current handoff.
- Neither `OBSERVED_LIVE_BEFORE_CUTOFF` nor a nonempty evidence list establishes admissibility.
  `publication_evidence.json`, raw sources, acquisition receipts and original publication records
  must pass the checks described below. A later download of a proven original is allowed; the
  current execution window is still enforced, so October reconstruction never becomes a September
  current handoff. A real non-historical export rechecks its sealed source executor and rejects
  manually declared qualification.

### Publication evidence and actual data qualification

`pipeline/publication_provenance.py` implements `publication-provenance-v1`.
`OBSERVED_BEFORE_CUTOFF` requires the hashed original response and a matching acquisition receipt
(source URL, response hash, HTTP 200, actual acquisition time before cutoff).
`ORIGINAL_PUBLICATION_RETRIEVED_LATER` currently supports original SEC iXBRL filings: the SEC
archive URL must bind the CIK/accession; a separately hashed SEC submissions response must contain
that accession and the exact acceptance time; each claimed tag/context/period/unit/value must
re-parse exactly from the original, undimensioned, correct-entity context. A current Companyfacts
response cannot pass as this original. Recorded publication and retrieval times stay separate.

All evidence file paths and hashes are checked; symlinks and traversal fail. Optional
`bundle_manifest.evidence_files` hashes every extra proof file in the SIGNAL snapshot. Unhashed
extra files are rejected. A publication check alone does **not** certify TTM formulas, historical
membership, source overlap or return labels. Feature lineage must bind its exact checked components
at each row's own information cutoff. Training labels may mature later than those features, but
must be known by the actual target model's training cutoff.

Current availability coverage also requires byte-bound pre-cutoff captures for membership and
covariance input tables. Late-retrieved market-provider histories and original press releases are
**not yet supported proof adapters**; they must not be relabelled as observed captures. This is a
concrete remaining limitation, not a finding that the source data did not exist then.

### Manual source-plan orchestration (not an automatic raw-fundamental builder)

```
.venv/bin/python -m pipeline.paper_manual_run --cycle 2026Q2 --historical \
  --source-plan REVIEWED_SOURCE_PLAN.json --source-root . --output NEW_WORKSPACE
```

The command acquires all configured cache/HTTPS sources, checks their reviewed hashes, builds a
separate SIGNAL snapshot, verifies publication evidence, calculates/replays the model, exports
packages and runs independent import checks. It consumes no signal JSON, never enables an
unvalidated-model acceptance flag, and records failed acquisition/qualification stages in
`workflow.json`. Each strategy's actual status is preserved.

Plan format: `manual-paper-source-plan-v1`, exact `cycle_key`, and `files` mapping destination
names to `{"cache_path": "relative/path", "sha256": "..."}` or
`{"url": "https://...", "sha256": "..."}`. Required destinations are all five SIGNAL tables,
`publication_evidence.json`, `instrument_map.json`, and all raw proof files referenced by the
evidence. Cache paths are relative to `--source-root`; destinations are confined to the new
workspace. The clock cannot be supplied/overridden on this CLI.

**Still incomplete:** this source-plan runner does not transform arbitrary Companyfacts into a
historically qualified full training panel. No reviewed real source plan meeting the requirements
has yet been supplied/generated. It is a tested orchestration component, not a claim that the
requested all-in-one raw-data-to-qualified-Q2 system is finished.

**History executor adapter** (`signal_result_from_history_execution(run_root, cycle)`, CLI `from-history`) takes
the run ROOT of `paper_history_execution.execute` (not `primary/`). The summary status must be
`COMPLETED_NOT_VALIDATED|COMPLETED_WITH_OPTIMIZER_ERRORS`, `model_replay_identical` true, and
`network_attempts`/`inputs_changed` empty. The non-snapshot files of both `primary/` and `replay/` must equal
`artifact_hashes` exactly. Both `snapshot/` trees must be identical and match their `bundle_manifest.json`
(`SOURCE_REVIEWED`), and `code_manifest.json` must match `code/`. The summary itself has no external anchor;
integrity rests on two independently written copies agreeing with it. The result is always `historical_only`,
`data_check_status: HISTORY_EXECUTOR_REPLAY_VERIFIED`, `signal_computed_at_source: "file_mtime"`.

**Instrument map** `paper-instrument-map-v1` (frozen, verified; tickers are never resolved by guess):
`{"format_version","frozen_at_utc","instruments":[{"security_id","isin" (ISO 6166 check digit),
"wkn" ("" allowed),"ticker","name","asset_type","share_class" ("" allowed),"sector_code",
"sector_name","currency":"USD","verification_status":"VERIFIED","evidence":[{"source","reference",
"retrieved_at_utc","sha256"}]}]}`. Duplicate security IDs/ISINs are rejected. A selected security
without an entry makes that strategy `BLOCKED_IDENTITY` (no CSV).
Mapping source requirements: every row is keyed by the executor's time-versioned `security_id` (share
class level, valid on the cutoff date), not by ticker; ISIN/WKN/name/share class must come from a
primary or reviewed reference (e.g. issuer filing, exchange/ISIN agency record, broker instrument
master export) captured with `retrieved_at_utc` before `frozen_at_utc` and a content `sha256`;
`frozen_at_utc` must not be in the future. Ticker-to-ISIN lookups without evidence are invalid.
No such map exists in the repository yet, so every real export is `BLOCKED_IDENTITY`.

## Dashboard

`list_packages(root=DEFAULT_PACKAGE_ROOT ("state/research/target_portfolios"), now=, ledger=)` (CLI `list`) is
read-only: one entry per published directory (newest first, `.tmp-*` skipped) with
`inspect_package(...).as_dict()` plus `generated_at_utc, mode, historical_only, data_check_status,
synthetic_fixture, model_validation_status, paper_replication`; damaged packages appear as
`integrity: "INVALID"` with `error` and `handoff_allowed: false`. Show `package_kind` and
`handoff_blockers` prominently; never present HISTORICAL_REPLAY/STALE/DIAGNOSTIC as current.
Export with `--output state/research/target_portfolios` so packages are direct children of the root.
Downloads: `manifest.json`, `zusammenfassung.md` and, per strategy, `<strategy>.csv` only when its
status is `COMPLETE`; otherwise `<strategy>.FAILED.json` exists instead (never offer a CSV for it).
Take file names from `manifest.strategies[<name>].file` / `.failure_file`, not from patterns.

## Package

Directory `<cycle>_<kind>_<alloc16>_<run_id>/`, published atomically (hidden `.tmp-*` + rename):
`equal_weight.csv, min_variance.csv, mean_variance.csv` (only for `COMPLETE` strategies; otherwise
`<strategy>.FAILED.json`), `manifest.json, zusammenfassung.md, instrument_map.json, signal_result.json`.

- `allocation_id` = hash of the business content (cycle, timings, ids, instrument-map hash, per-strategy
  rows/status). Identical inputs → identical `allocation_id`; re-export into the same directory returns
  the existing package. `run_id` identifies each issuance.
- `package_kind`: `CURRENT_HANDOFF` (regular run inside window, data PASSED, not historical-only),
  `HISTORICAL_REPLAY`, `STALE`, `DIAGNOSTIC`. Only `CURRENT_HANDOFF` has `handoff_eligible: true`.
- `valid_until_utc` = scheduled execution time.

CSV `paper-target-portfolio-csv-v1`: UTF-8 without BOM, `,` delimiter, `"` quoting, `.` decimal,
`\n` line ends, ISO-8601 UTC. Columns:
`schema_version, run_id, allocation_id, package_kind, strategy, cycle_key, signal_as_of_utc,
scheduled_execution_at_utc, row_type (SECURITY|CASH), security_id, isin, wkn, ticker, name,
asset_type, share_class, sector_code, sector_name, currency, target_weight`.
`target_weight` has 12 decimals; `0.025` = 2.5 %. Security weights ≤ 0.05 (+1e-8). Exactly one
`CASH` row (`CASH:USD`); all rows sum to 1 within 1e-6. No renormalization, no substitutes.

## Model labelling

Every manifest has `model` with `paper_replication: NOT_EXACT_PAPER_REPLICATION`,
`validation_status: NOT_VALIDATED`, the rule manifest, per-strategy definitions and deviations:
mean_variance is the max-Sharpe (tangency) point. The paper explicitly mentions maximization of
the Sharpe ratio; its exact original parameters are not established. The original notebook also
uses `EfficientFrontier.max_sharpe`. The fixed 1.5 % risk-free value remains our explicit assumption,
with annual log units;
min_variance minimizes variance directly (the notebook's `max_sharpe([0]*n)` is a bug);
deterministic scikit-learn instead of R caret. `zusammenfassung.md` repeats this.

Strategy definitions are versioned; each manifest strategy entry carries `label` and
`definition_version`, which also enter the `allocation_id`, and import rejects unknown versions.
Historical and new packages use the same definitions:

| strategy | label | definition_version |
|---|---|---|
| equal_weight | Gleichgewichtet | `equal_weight-1_over_n-v1` |
| min_variance | Minimale Varianz | `min_variance-annual-log-max-sharpe-v1` |
| mean_variance | Rendite/Risiko – Max-Sharpe | `mean_variance-annual-log-max-sharpe-v1` |

Max-Sharpe units (all annual log): expected return = 4 x predicted quarterly log return;
covariance = 252 x covariance of daily log returns over the last 253 NYSE sessions through cutoff;
risk-free = log1p(0.015). Not an exact reproduction of the original MATLAB/notebook optimizer.
`risk_free_annual = 0.015` and `weight_cap = 0.05` are frozen: a signal whose `model_rules` differ is
refused (a new definition version would be required; no tuning on later data).

The optimization identity is pinned to `paper_fixed_lag.OPTIMIZATION_VERSION`
(`annual-log-max-sharpe-v1`): every signal result must carry the producing run's `model_rules`
(rule manifest; the adapters read `RUN/rule_config.json` and refuse if absent) with that
`optimization_version` and an `optimization_definition` exactly equal to
`optimization_definition()`. There is no fallback to the current code's rules, so older or unversioned
optimizer results (e.g. earlier curves) are never joined. The manifest `model` section embeds
`optimization_version`, the full `optimization_definition` (units, objective, approximation,
assumptions, `matlab_equivalence: NOT_PROVEN`) and `performance_metric_note`: a conventional
simple-return Sharpe reported for realized performance is distinct from the annual-log objective.
The version and the definition hash are part of the `allocation_id`.

## Target-portfolio semantics

- Each CSV is the complete portfolio of one strategy. In an account dedicated to that strategy,
  holdings missing from the CSV have target weight 0.
- Never apply that to positions of other strategies or unrelated holdings.
- The three strategies are three separate portfolios, not three consecutive rebalancings.
- A failed strategy has no CSV; never use another strategy's weights instead.

## Import (`load_target_portfolio`) — read-only

`inspect_package(path, now=..., ledger=..., accept_unvalidated_model=False)` raises
`TargetPortfolioError` on any integrity failure: schema/protocol versions, exact file set, safe
member names (no traversal, no symlinks, size limit), SHA-256 + size of every file, instrument-map
hash, exact CSV header/field count, constant columns matching the manifest, timings recomputed from
the schedule, finite nonnegative 12-decimal weights, cap, unique security IDs/ISINs, identity equal
to the frozen map, one cash row, sum, row count/cash vs manifest, and recomputed `allocation_id`.
It then lists handoff blockers: non-current kind, not regular, synthetic, data check not PASSED,
model not validated (unless explicitly accepted), current time before generation or at/after
`valid_until_utc`, and allocation/strategy already claimed.
`load_target_portfolio(path, strategy, ...)` returns `(report, StrategyPortfolio)` only if no blocker
applies to that strategy. Validation never marks anything as processed.

`claim_target_portfolio(path, strategy, ProcessedLedger(dir), consumer=...)` validates and then
creates `<allocation_id>__<strategy>.claim.json` with `O_EXCL` + fsync. A second claim (also from a
re-issued package with the same allocation) fails with `AlreadyClaimed`. Claim before placing orders.
This legacy operation is only an exclusive reservation; it does not itself implement crash recovery
or guarantee one external execution.

## Resumable execution consumer (simulated, no broker adapter)

`portfolio_consumer.ConsumerLedger` uses a separate SQLite ledger and append-only audit, bound to
the **allocation, strategy and target account**. Public APIs: `accept_package`, `get`, `audit`,
`process`; external contracts: `ExecutionProvider` plus snapshots, intents, reports and fills.
Acceptance, active processing, partial fills and completion remain distinct persisted states.

Every resume reconciles actual positions, open orders and confirmed fills before any submission.
An ambiguous external acceptance blocks new submission unless an authoritative query or the
provider's same-key idempotency resolves it. A replacement allocation needs its predecessor ID and
explicit reason; residual orders must be reconciled/cancelled, not blindly duplicated.
Research simulation is explicit opt-in. No historical CSV price is converted into order quantities.

The simulated consumer tests include crash after external acceptance but before local acknowledgement,
partial fills, resume, duplicate processing, account/strategy isolation and explicit replacement.
External broker/provider implementation and live planning are still outside this scope.
**Local hashes, SQLite transactions and claims do not guarantee external exactly-once execution.**

## Responsibilities of the later execution program (not implemented here)

1. Read the target portfolio via `claim_target_portfolio` (one strategy per dedicated account).
2. Fetch actual positions, cash and open orders.
3. Map ISINs to the platform's actual instruments.
4. Consider tradability, account currency (USD research vs. a possibly EUR account), live prices,
   lot sizes and costs. Never derive share counts from historical reference prices.
5. Compute buy/sell orders from the difference to the target portfolio.
6. Handle partial fills, rejections and retries.
7. Report actual executions and remaining deviations.
Untradeable securities must not be silently replaced or dropped; a platform-adapted portfolio is a
separately documented strategy variant.
