# Histogram Equalization 

## 1. Introduction to Histogram Equalization
Histogram Equalization is a technique used to enhance the contrast of an image by redistributing its pixel intensity values. The goal is to transform an image so that its histogram becomes approximately uniform, improving the visibility of details in the image. This process enhances both bright and dark areas.

---

## 2. Steps in Histogram Equalization

### Compute Histogram:
- Count how many pixels have each intensity value (0–255).
- This creates a histogram that represents the distribution of pixel intensities in the image.

### Compute Cumulative Distribution Function (CDF):
- The CDF represents the cumulative sum of histogram values.
- Normalize the CDF to map intensity values from 0 to 255.
- This helps determine how pixel intensities should be redistributed.

### Apply Equalization:
- Use the normalized CDF to map each pixel in the original image to a new intensity value.
- This transformation enhances the contrast of the image.

---


### Kernels for Histogram Equalization
1. **Histogram Computation Kernel:**
   - Each CUDA thread processes one pixel and updates the histogram bin corresponding to its intensity.
   - Atomic operations (`atomicAdd`) are used to safely update shared histogram data.
     
    **Atomic Operations for Histogram Update:**
      - Multiple threads may try to update the same histogram bin simultaneously.
      - `atomicAdd` ensures safe, consistent updates.

2. **CDF Computation and Normalization Kernel:**
   - Uses shared memory for fast access to the histogram.
   - A single thread computes the cumulative sum to obtain the CDF and normalizes it to the range [0, 255].
     
     **Shared Memory for CDF Calculation:**
       - Shared memory reduces latency compared to global memory.
       - This improves the performance of cumulative sum calculation.


3. **Apply Equalization Kernel:**
   - Each thread maps the original intensity to the new intensity using the normalized CDF.
   - The output array stores the equalized image.

---

## 7. Limitations and Considerations
- **Noise Amplification:** Histogram equalization may amplify noise in low-quality images.
- **Overprocessing:** Can sometimes lead to an unnatural appearance of the image.
- **Memory Requirements:** Requires sufficient memory on the GPU for large images and histograms.
