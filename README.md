# 🌿 PlantDisease AI

**Tomato leaf disease classification that is honest about what it can and cannot do.**

A deep learning study and web app built on MobileNetV2 transfer learning. Beyond test accuracy, it
measures what matters for real use: how models trained on lab photos hold up on **field photos**,
whether their **confidence can be trusted**, whether they can recognise **images that are not leaf
photos**, what they **look at**, and whether they can run **on a phone**.

> Academic project. Predictions are not a substitute for diagnosis by an agronomist or a lab.

### Headline results

| | Lab photos (PlantVillage) | Field photos (PlantDoc) |
|---|---|---|
| Fine-tuned MobileNetV2 (3 seeds) | **95.3% ± 0.9** | 22.4% ± 0.5 |
| Best lab-only configuration (strong augmentation) | 93.7% | **27.0%** (+5.0, p = 0.013) |
| Always guessing the most common class | 29.5% | 20.2% |
| Mean confidence of the models | 82–98% | 61–89% |

- **Lab accuracy says little about the field.** 15 models score 85–97% on lab photos and 21–29%
  on real field photos.
- **The app warns on unfamiliar photos.** Its out-of-distribution detector flags 95% of field
  photos and 5% of lab photos, and separates non-leaf images with AUROC ≥ 0.999.
- **Calibration and on-device size.** Temperature scaling brings lab calibration error to ≤ 0.02;
  TensorFlow Lite dynamic-range quantisation makes the model 3.5× smaller (9.8 → 2.5 MB) with no
  loss of accuracy.

📄 **Full write-up:** [`reports/REPORT.md`](reports/REPORT.md) · 📊 **All numbers:** [`reports/RESULTS.md`](reports/RESULTS.md)

---

## What's inside

| | |
|---|---|
| **Leakage-aware data split** | Repeat photos of the same leaf, byte-identical and visually identical images are grouped before splitting, so no leaf appears in both training and test data. The split is versioned in `data/splits/`. |
| **14-run experiment suite** | Frozen vs fine-tuned vs from scratch; MobileNetV2 vs EfficientNetB0 vs ResNet50V2; 128 vs 224 px; three augmentation levels; background replacement; mixed-domain training; 3 seeds. |
| **Statistics, not just scores** | Bootstrap intervals, seed repeats, and paired McNemar tests (Holm-corrected) for every claim that one setting beats another. |
| **Field generalisation** | Every model is tested on PlantDoc field photos it never saw during training. |
| **Background replacement** | Uses PlantVillage's leaf segmentations to paste leaves onto random backgrounds, removing the lab-background shortcut. |
| **Calibration** | Temperature scaling, with expected calibration error and reliability diagrams. |
| **"Is this even a leaf?"** | Three out-of-distribution scores (softmax, energy, Mahalanobis) evaluated by AUROC; the app warns on unfamiliar photos when a detector is installed. |
| **Measured explanations** | Grad-CAM attention inside the leaf mask vs. on the background, quantified over 300 images per model. |
| **On-device export** | TensorFlow Lite float32 / dynamic-range / int8, with size, latency and accuracy. |
| **Streamlit app** | Diagnose (input checks, calibrated confidence, Grad-CAM, disease guide), Model info and an Experiments dashboard. |
| **Tests and CI** | 100+ pytest tests (unit, UI, scripts, end-to-end training) and GitHub Actions. |

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt
streamlit run app.py
```

Open http://localhost:8501 and try **Diagnose → Try the sample photo**.

The app runs from the files in the repository; datasets are only needed to train or evaluate.

---

## Reproducing the study

Datasets and training runs are large, so they live in a *work directory* outside the repository:
set `PLANTDISEASE_WORKDIR`, or create a git-ignored `local_paths.json`:

```json
{ "workdir": "C:/Users/<you>/PlantDiseaseWork" }
```

Keep it outside cloud-synced folders (OneDrive, Dropbox) to avoid uploading ~20,000 images.

```bash
python -m scripts.download_data                       # PlantVillage + PlantDoc tomato subsets (~0.9 GB)
python -m scripts.prepare_data                        # leakage-aware split + data audit
python -m scripts.run_experiments --evaluate-baseline # train + evaluate all 14 runs (CPU: ~6 h, resumable)
python -m scripts.benchmark_speed                     # latency of every run, measured together (idle machine)
python -m scripts.compare_runs                        # paired significance tests between runs
python -m scripts.check_detector                      # the app's unfamiliar-photo warning on lab vs field photos
python -m scripts.export_tflite --model-dir <workdir>/runs/<run>
python -m scripts.make_report                         # RESULTS.md + figures
```

Single runs: `python -m scripts.train --manifest --backbone efficientnetb0 --strategy finetune --evaluate`.

| Script | Purpose |
|---|---|
| `download_data` | Fetch only the tomato folders of both datasets (resumable). |
| `prepare_data` | Group duplicates, split 70/15/15, write manifests and `audit.json`. |
| `train` | Train one model (any backbone, strategy, augmentation; folders or manifest). |
| `run_experiments` | The whole suite from `experiments/suite.json`; skips finished runs. |
| `evaluate` | Score a model on any class-folder dataset, with a class-order check. |
| `benchmark_speed` | Time every run in interleaved rounds, so all models see the same machine load. |
| `compare_runs` | Exact McNemar tests between runs on the same images, Holm-corrected. |
| `check_detector` | How often the app's "unfamiliar photo" warning fires on lab vs field photos, and accuracy with and without it. |
| `export_tflite` | TensorFlow Lite variants with size, latency and accuracy. |
| `make_report` | Aggregate results into tables and figures. |
| `explain` | Grad-CAM figure for one image. |
| `preview_augmentation` | Figure of the background-replacement augmentation. |
| `inspect_model` | Check that a model's files are consistent. |

---

## Project structure

```
PlantDiseaseAI/
├── app.py                        # Streamlit entry point
├── plantdisease/                 # Python package
│   ├── inference.py              # loading, preprocessing, calibrated prediction
│   ├── backbones.py              # model factory (MobileNetV2, EfficientNetB0, ResNet50V2)
│   ├── training.py               # training core (frozen / fine-tune / scratch)
│   ├── pipeline.py               # tf.data from manifests + background replacement
│   ├── evaluation.py             # accuracy, field test, calibration, OOD, Grad-CAM, speed
│   ├── metrics.py  calibration.py  ood.py  gradcam.py  quality.py  knowledge.py
│   └── ui/                       # Streamlit pages, charts and components
├── models/                       # deployed model + class names + metadata
├── data/
│   ├── disease_info.json         # disease guide content
│   └── splits/                   # versioned split manifests + data audit
├── experiments/suite.json        # experiment definitions
├── reports/
│   ├── REPORT.md  RESULTS.md     # write-up and generated tables
│   ├── results/                  # per-run metrics (JSON)
│   └── figures/
├── scripts/                      # command-line tools (see above)
├── tests/                        # pytest suite
└── .github/workflows/tests.yml   # CI
```

## Running the tests

```bash
python -m pytest                  # everything (~3 min, includes tiny training runs)
python -m pytest -m "not slow"    # the fast suite, as run in CI
```

## Deploying to Streamlit Community Cloud

Set the main file to `app.py` and pick a Python version supported by your TensorFlow pin
(3.12 is safe). `requirements.txt` and `.streamlit/config.toml` are picked up automatically.

## References

The datasets and methods are cited in [`reports/REPORT.md`](reports/REPORT.md#references).
Datasets: PlantVillage (Hughes & Salathé, 2015) and PlantDoc (Singh et al., 2020, CC BY 4.0).
