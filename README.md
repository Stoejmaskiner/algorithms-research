# Algorithms Research
developing novel and experimental dsp algorithms (of dubious utility)

## Lossy Compression Algorithms
Compressors:
```python
# === chunkify algorithms ===
# note: they each allow for different centrality measures to be used for downsampling
vec_downsampler: Vec -> Vec       # traditional downsampling, but with more obscure settings allowed, like using median filters instead of lowpass
vec2chunk_const: Vec -> Chunk1D   # group similar samples together as a sequence of identical values, which is a tuple of only two numbers, this version has a fixed number of chunks
vec2chunk_var: Vec -> Chunk1D     # the same as vec2chunk_const, but the number of chunks is allowed to be variable, as in audio with a lot of variation (a higher information density) will have more and smaller chunks
matrix_downsampler: Matrix -> Matrix   # uses either filters or pooling algorithms to downsample matrices (batches)
matrix2chunk_const: Matrix -> Chunk2D  # uses a predefined grid size, adjusting the thresholds of the x and y grid lines to preserve as much information as possible
matrix2chunk_var2: Matrix -> Chunk2D   # uses binary decision trees 
matrix2chunk_var4: Matrix -> Chunk2D   # uses quad decision trees

# === classless statistical learning algorithms ===
matrix2pca: Matrix -> Matrix, PCA          # performs principal component analysis on a batch (they should have common characteristics to be effective)
matrix2auto_encoder: Matrix -> Matrix, AE  # trains an auto-encoder on a batch (they should have common characteristics to be effective), may be used after a PCA stage

# === classfull SL algorithms ===
matrix2lda: Matrix, Labels -> Matrix, LDA  # takes a labelled batch and performs LDA
```
Decompressors:
```python
# === de-chunkify algorithms ===
vec_upsampler       # regular upsampler algorithms, nothing new, here for completeness
chunk2vec           # expands chunkified vectors by using various interpolation methods
matrix_upsampler         # upsampler using unpooling or interpolation methods
chunk2matrix             # expands chunkified matrix data by using various interpolation methods

# === SL decompressors ===
pca2vec
auto_encoder2vec: Vec, AE -> Vec
lda2vec: Vec, LDA -> Vec
```
Generative decompressors:
```python
gen_ae: RNG, AE -> Vec
```