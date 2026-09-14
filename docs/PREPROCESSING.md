# EEG preprocessing

`src.data.preprocessing.preprocess_record` converts one continuous EEG record to model-ready patches.

The processing sequence is:

1. convert input to finite `float32` values in microvolts;
2. apply common-average reference to monopolar channel sets;
3. apply a fourth-order 0.5-75 Hz band-pass filter;
4. apply a 50 Hz notch filter when permitted by the source sampling rate;
5. resample to 200 Hz with polyphase resampling;
6. compute record-wise, channel-wise median and 95th absolute-deviation scale;
7. clip normalized values to `[-10,10]`;
8. form non-overlapping 30-second windows;
9. divide every window into 30 one-second patches.

The function accepts a `PreprocessingConfig` object for changing the target rate, frequency limits, notch frequency, window size, patch size, and clipping range.

