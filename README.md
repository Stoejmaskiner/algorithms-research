# Algorithms Research
developing novel and experimental dsp algorithms (of dubious utility)

## Lossy Compression Algorithms
Compressors:
```python
# note: when FFTs matrices are mentioned, they can be intended in two different ways:
# - a collection of single-frame FFTs, one FFT frame per original audio clip
# - a single audio clip turned into multiple sequential FFT frames.
# this distinction should be clear from the description of the algorithms

# === chunkify algorithms ===
# note: they each allow for different centrality measures to be used for downsampling
audio_downsampler: Vec -> Vec       # traditional downsampling, but with more obscure settings allowed, like using median filters instead of lowpass
audio2chunk_const: Vec -> Chunk1D   # group similar samples together as a sequence of identical values, which is a tuple of only two numbers, this version has a fixed number of chunks
audio2chunk_var: Vec -> Chunk1D     # the same as audio2chunk_const, but the number of chunks is allowed to be variable, as in audio with a lot of variation (a higher information density) will have more and smaller chunks
fft_downsampler: Matrix -> Matrix   # uses either filters or pooling algorithms to downsample sequential FFT frames
fft2chunk_const: Matrix -> Chunk2D  # uses a predefined grid size, adjusting the thresholds of the x and y grid lines to preserve as much information as possible
fft2chunk_var2: Matrix -> Chunk2D   # uses binary decision trees 
fft2chunk_var4: Matrix -> Chunk2D   # uses quad decision trees

# === classless statistical learning algorithms ===
audio2pca: Matrix -> Matrix, PCA          # performs principal component analysis on a batch of audio clips (they should have common characteristics to be effective)
fft2pca: Matrix -> Matrix, PCA            # performs principal component analysis on a batch of FFTs
audio2auto_encoder: Matrix -> Matrix, AE  # trains an auto-encoder on a batch of audio clips (they should have common characteristics to be effective), may be used after a PCA stage
fft2auto_encoder: Matrix -> Matrix, AE    # trains an auto-encoder on a batch of FFTs, may be used after a PCA stage

# === classfull SL algorithms ===
audio2lda: Matrix, Labels -> Matrix, LDA  # takes a batch of labelled audio and performs LDA
fft2lda:   Matrix, Labels -> Matrix, LDA  # takes a batch of labelled ffts and performs LDA
```
Decompressors:
```python
# === de-chunkify algorithms ===
audio_upsampler       # regular upsampler algorithms, nothing new, here for completeness
chunk2audio           # expands chunkified audio by using various interpolation methods
fft_upsampler         # upsampler using unpooling or interpolation methods
chunk2fft             # expands chunkified fft data by using various interpolation methods
```
Generative decompressors:

Compound algorithms:

