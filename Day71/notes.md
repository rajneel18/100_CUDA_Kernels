# CUDA OCR Binarization 


## What is Binarization?
Convert grayscale image pixels to black (0) or white (255) based on a threshold (e.g., 128).

## CUDA Parallelization Strategy
- Each GPU thread processes one pixel.
- If pixel value > threshold → white (255), else → black (0).
- Massive speed-up using `<<<blocks, threads>>>` layout.

##  Why It Matters for OCR
Binarization is a core preprocessing step before contour detection, segmentation, and character recognition in OCR pipelines.

## Files
- `ocr_bin.cu`: CUDA C implementation of parallel binarization.
- `Makefile` (optional): For building with `nvcc`.

