# Theory of Downsampling and Intensity Quantization

## 1. Downsampling (Reducing Resolution)  
Downsampling is the process of reducing the resolution of an image by decreasing the number of pixels. This reduces the amount of data, making the image smaller and simpler. However, some fine details may be lost in the process.  

### How it works:  
- Choose a smaller target size (e.g., reduce a 5696 × 3392 image to 2848 × 1696).  
- Map each pixel in the smaller image to a group of pixels in the original image and select one representative value (e.g., the top-left pixel or the average of that group).  

### Mathematical Formula:  
For a downsampled pixel at position \((x, y)\) in the new image:  
\[
\text{origX} = x \times \left( \frac{\text{original width}}{\text{new width}} \right)
\]  
\[
\text{origY} = y \times \left( \frac{\text{original height}}{\text{new height}} \right)
\]  
The pixel at \((\text{origX}, \text{origY})\) in the original image becomes the downsampled pixel value.  

---

## 2. Intensity Quantization (Reducing Gray Levels)  
Quantization reduces the number of possible intensity values (gray levels) in the image. For an 8-bit grayscale image, the intensity values range from 0 to 255. Reducing this range (e.g., to 16 levels) simplifies the image but sacrifices detail.  

### How it works:  
1. Divide the intensity range (0–255) into smaller intervals.  
2. Assign each pixel intensity to the nearest level within the reduced range.  

### Quantization Formula:  
\[
\text{Quantized Pixel} = \left( \frac{\text{Original Pixel}}{\text{Quantization Level}} \right) \times \text{Quantization Level}
\]  

### Example:  
For a pixel value of 75 and a quantization level of 16:  
\[
\text{Quantized Pixel} = \left( \frac{75}{16} \right) \times 16 = 64
\]  

---

## Effect of Quantization:  
- Reduces image complexity by limiting the number of intensity values.  
- Introduces "banding" or visible intensity steps, which may reduce image quality but make storage and processing more efficient.  

---

## Combining Downsampling and Quantization  
The combined process significantly reduces the image size and complexity, making it useful for:  
- **Data compression**  
- **Image analysis with limited resources**  
- **Reducing noise and simplifying visual details for further processing**  
