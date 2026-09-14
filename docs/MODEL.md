# Model

## Tokenization

Every one-second EEG patch is encoded by four temporal convolution branches with kernel sizes 3, 7, 15, and 31. Their 384 temporal features are concatenated with 128 projected log-power FFT features and normalized to a 512-D token.

## Geometry-lag neighborhoods

- T connects the same channel at lags `±1...±4`.
- S connects geometrically grouped channels at the same time patch.
- ST connects nearby, middle-distance, and far channels at directed lags 1, 4, and 8 patches.

Temporal relations use normalized lag features, spatial relations use normalized scalp distance, and ST relations use both distance and lag.

## Encoder

The encoder contains four 768-D blocks followed by fifteen 1024-D blocks. Every block computes separate T, S, and ST attention messages, concatenates them, projects them into one shared state, and applies a residual feed-forward update.

## Decoder

A token-wise layer-normalized two-layer MLP reconstructs 200 waveform samples for every token. Pretraining optimizes waveform and log-power reconstruction on masked valid tokens.

