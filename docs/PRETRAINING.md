# Pretraining

## Configuration

`configs/pretrain.yaml` defines the optimizer, schedule, masking, validation, checkpoint, and distributed-training settings.

The default configuration uses:

- 24,000 optimizer steps;
- AdamW with peak learning rate `2e-4` and weight decay `0.05`;
- 5% linear warmup followed by cosine decay;
- BF16 autocasting;
- gradient norm clipping at 1;
- 15 microbatches per optimizer step and GPU;
- activation checkpointing;
- periodic validation and validation-loss early stopping.

## Sampling and masking

Datasets are sampled in proportion to the square root of their window counts, with the TUH probability capped at 0.4. Every draw selects one variable-channel 30-second window.

Half of the valid tokens are masked. The mask mixture is:

- random token: 0.4;
- temporal block: 0.2;
- whole channel: 0.2;
- spatiotemporal block: 0.2.

## Objective

The objective is:

```text
waveform L1 + 0.1 * log-power spectral L1
```

Both terms are evaluated only on valid masked tokens.

## Launch

From the repository root:

```bash
bash scripts/pretrain.sh TRAIN_MANIFEST VALIDATION_MANIFEST OUTPUT_DIR
```

The training directory records the resolved configuration, periodic metrics, validation metrics, resumable checkpoints, and a completion summary.

