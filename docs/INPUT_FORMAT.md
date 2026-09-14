# Input format

## Prepared arrays

Each NPY file stores one or more normalized EEG windows:

```text
[windows, channels, patches, samples]
```

The default configuration uses 30 one-second patches and 200 samples per patch, giving `[W,C,30,200]`. Values use `float32`.

## Manifest

Training and validation manifests use JSON Lines. Each line describes one prepared array:

```json
{"path":"data/tuh/record_001.npy","id":"tuh_record_001","dataset":"tuh_eeg","split":"train","windows":120,"channel_names":["Fp1","Fp2","F3","F4"]}
```

Required fields:

- `path`: path to the NPY array;
- `id`: stable record identifier;
- `dataset`: dataset identifier used by the sampler;
- `split`: `train` or `validation`;
- `windows`: number of windows in the array;
- `channel_names`: channel labels understood by the standard 10-20 montage.

Coordinates can be supplied directly with a `coordinates` array of `[x,y,z]` entries. A `null` entry marks an unknown position. When `coordinates` is absent, coordinates are derived from `channel_names`; bipolar labels use normalized scalp midpoints.

