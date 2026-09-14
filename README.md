# Geometry-Lag Sparse Relational Attention for EEG

This repository provides the model and pretraining implementation for **Geometry-Lag Sparse Relational Attention for EEG Representation Learning**.

The code includes:

- multi-scale temporal and spectral EEG tokenization;
- geometry-lag temporal, spatial, and spatiotemporal relational attention;
- variable-channel EEG preprocessing and data loading;
- four masked-token sampling strategies;
- waveform-spectral reconstruction pretraining;
- distributed BF16 training with checkpointing, validation, and resume support.

## Environment

Create and activate the Conda environment:

```bash
conda create -n geometry_lag_eeg python=3.10 pip -y
conda activate geometry_lag_eeg

pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1

pip install \
  numpy==1.26.4 \
  scipy==1.10.1 \
  mne==1.6.1 \
  "h5py>=3.10,<4" \
  "scikit-learn>=1.3,<2" \
  "pyyaml>=6.0" \
  "tqdm>=4.66" \
  "pytest>=8.0"
```

The distributed training entry uses `torch.distributed` with the NCCL backend.

## Input preparation

Convert an EEG array in microvolts to the standard window format:

```bash
python examples/prepare_sample.py \
  --input recording.npy \
  --channels channels.json \
  --sample-rate 256 \
  --output prepared_recording.npy
```

Prepared arrays have shape `[windows, channels, 30, 200]`. Create JSONL train and validation manifests following `docs/INPUT_FORMAT.md`.

## Model example

```bash
python examples/model_forward.py
```

## Distributed pretraining

```bash
bash scripts/pretrain.sh \
  data/train.jsonl \
  data/validation.jsonl \
  outputs/pretrain
```

Set a different GPU process count when needed:

```bash
NPROC_PER_NODE=4 bash scripts/pretrain.sh \
  data/train.jsonl data/validation.jsonl outputs/pretrain
```

Training settings are defined in `configs/pretrain.yaml`. See `docs/PRETRAINING.md` for the complete workflow.

## Tests

```bash
pytest -q
```

## Repository layout

- `src/models`: tokenizer, geometry-lag neighborhoods, relational attention, encoder, decoder, and complete model.
- `src/data`: signal preprocessing, deterministic manifest dataset, and dataset sampling.
- `src/training`: masking, reconstruction loss, and distributed pretraining.
- `configs`: pretraining configuration.
- `examples`: input preparation and model-forward examples.
- `tests`: focused correctness tests for the released pipeline.
- `docs`: model, input, preprocessing, and training documentation.

