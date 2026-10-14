# Traffic Alert Lab

**Missing traffic observations, PyTorch forecasting, and event-level congestion alerts.**

[中文](README.zh-CN.md) · [Interactive results](demo/index.html) · [Reproduction guide](docs/pems_bay_pipeline_reproduction.zh-CN.md) · [PyTorch lesson](docs/undergraduate_pytorch_learning.zh-CN.md)

Explore how risk supervision and alert thresholds trade earlier notifications against missed events and false alarms. This repository includes a CPU training pipeline, frozen evaluation records, and a small synthetic tensor lesson.

**The predefined joint primary test did not pass.** The onset MLP has fewer false alerts but substantially lower event detection. This project does not establish a new neural architecture or overall service superiority.

## Online deployment

The repository includes a GitHub Pages workflow for the same aggregate explorer. Its deployment status appears under Actions. Enable **Settings → Pages → Source: GitHub Actions** once before the first deployment.

## Start without installing anything

Download the repository and open `demo/index.html` in your browser. Select an input scenario and compare all seven methods across early warning, detection and false alerts. The page is self-contained and uses saved aggregate results. It contains no per-station observation series and makes no network requests.

To check the result tables with Python 3.12 and the standard library:

```bash
python verify_release.py
python build_thesis_experiment_draft.py --check
```

The first command also restores three large frozen JSON records from compressed files after checking their exact SHA-256 and original byte length. It refuses to replace changed existing records. These commands check saved result consistency, without reconstructing traffic events from raw data or retraining models.

## What the experiment found

| Fine-grid method | ≥10-minute proxy warning recall | Detection including late alerts | False alerts / valid station-day |
| --- | ---: | ---: | ---: |
| Global calibration | 8.92% | 80.39% | 0.187396 |
| Availability-group calibration | 8.47% | 80.14% | 0.179023 |
| Onset logistic regression | 10.85% | 36.80% | 0.096126 |
| Onset MLP | 11.89% | 28.18% | 0.086581 |

Rates are equal averages over three input scenarios. The evaluated subset contains 24 stations and 673 complete low-speed proxy events over 83 days. Replaying the same events under three input scenarios does not triple the independent event count.

The onset MLP's R10 differences versus global and group calibration are +2.97 and +3.42 percentage points. Their 97.5% paired-date bootstrap intervals are [−0.20, 5.89] and [0.36, 6.29]. The predefined requirement was that both lower bounds exceed zero, so the joint test failed. Detection falls by 52.20 percentage points versus global calibration. A favorable early-warning point estimate is insufficient to establish overall benefit.

The “10 minutes” metric follows the study's proxy alignment, `L = 5(s − a + 1)`. It is not independently verified lead time against real congestion onset. See the [full tables](docs/thesis_result_tables.zh-CN.md) and [discussion](docs/thesis_experiment_discussion.zh-CN.md).

## Learn the PyTorch implementation

The forecasting front end uses a masked graph TCN with 19,523 parameters per model. It consumes 12 historical time steps and predicts three future speeds. The risk MLP has dimensions `51 → 32 → 16 → 1` and 2,209 parameters per model. The networks train separately.

After installing the research dependencies:

```bash
python undergraduate_torch_demo.py --output runs/torch-lesson.json
```

The synthetic lesson uses the actual model classes with randomly initialized weights. It checks tensor shapes, mask invariance, finite all-missing outputs, and BCE gradients. It performs **zero optimizer updates**, reads no traffic data and loads no trained weights. Its loss is a teaching value, not research performance.

## Run the full CPU pipeline

The recorded environment is Windows, Python 3.12 and CPU PyTorch 2.8.0. Create a virtual environment and obtain PEMS-BAY independently from its upstream access instructions.

```bash
python -m venv .venv
# Activate the environment using your shell's normal command.
python -m pip install -r requirements-pilot.txt --extra-index-url https://download.pytorch.org/whl/cpu
python restore_reference_records.py
python run_pems_bay_pipeline.py --workspace runs/my-first-run --data /path/to/pems-bay.h5
```

The runner requires a fresh `runs/` subdirectory, checks the input SHA and executes 18 stages. It trains the forecasting and risk models, selects operating points on calibration data, replays 42 fixed evaluation settings and performs the independent event audit. Use `--resume` after interruption or `--verify-only` to check a completed run.

A previous complete run in the recorded environment matched the saved numerical results. See [verification](verification/report.zh-CN.md). Re-running an already analyzed dataset is engineering reproduction, not fresh independent confirmation. Other hardware and library versions have not been verified for identical numbers.

## Repository map

| Location | Purpose |
| --- | --- |
| `demo/index.html` | Offline aggregate result explorer |
| `run_pems_bay_pipeline.py` | Isolated training, resume and verification entry |
| `pipeline_seed/` | Exact source/config seed used by the recorded pipeline |
| `outputs/pems-bay-evaluation/` | Frozen results, plots and audit records |
| `undergraduate_torch_demo.py` | Synthetic tensor and gradient lesson |
| `docs/` | Protocols, metric definitions and Chinese tutorials |
| `tests/` | Pipeline isolation and failure handling tests |

## Data, attribution and licensing

This package contains code, aggregate research results and synthetic teaching inputs. It excludes raw HDF5, trained weights, prediction/feature caches, actual station time-series windows and thesis Word/PPT files. Older local timeline demos are not part of this public package.

PEMS-BAY access originates from the [DCRNN authors' repository](https://github.com/liyaguang/DCRNN#data-preparation). Upstream software licensing does not serve as this project's permission to redistribute dataset files. See [data policy](DATA_POLICY.md) and [attribution](THIRD_PARTY_NOTICES.md).

Project code and original documentation use the [MIT License](LICENSE). Data and third-party material keep the boundaries in [CODE_LICENSE.md](CODE_LICENSE.md). Older dated protocols describe their historical preparation state.

Questions and reproducibility reports are welcome. See [contribution guidance](CONTRIBUTING.md). GitHub stars measure community interest, not algorithm originality or graduation eligibility.
