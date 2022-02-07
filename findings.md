# Algorithm Research Findings
Here are a list of discoveries I've made in my research:


## Lossy Compression
- For one-shot kick samples, fft -> poly KPCA -> inv poly KPCA -> ifft performed way better than any other KPCA kernel showing barely any artifacts down to a tiny fraction of the original principal components