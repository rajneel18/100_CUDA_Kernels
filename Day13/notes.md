# Theory of Downsampling and Intensity Quantization

## 1. Downsampling (Reducing Resolution)  
Downsampling is the process of reducing the resolution of an image by decreasing the number of pixels. This reduces the amount of data, making the image smaller and simpler. However, some fine details may be lost in the process.  

### How it works:  
- Choose a smaller target size (e.g., reduce a 5696 × 3392 image to 2848 × 1696).  
- Map each pixel in the smaller image to a group of pixels in the original image and select one representative value (e.g., the top-left pixel or the average of that group).  

### Mathematical Formula:  
For a downsampled pixel at position (x,y)(x,y) in the new image:
- origX=x×(original widthnew width)
- origX=x×(new widthoriginal width​)
- origY=y×(original heightnew height)
- origY=y×(new heightoriginal height​)

---

## 2. Intensity Quantization (Reducing Gray Levels)  
Quantization reduces the number of possible intensity values (gray levels) in the image. For an 8-bit grayscale image, the intensity values range from 0 to 255. Reducing this range (e.g., to 16 levels) simplifies the image but sacrifices detail.  

### How it works:  
1. Divide the intensity range (0–255) into smaller intervals.  
2. Assign each pixel intensity to the nearest level within the reduced range.  

### Quantization Formula:  
- Quantized Pixel=(Original PixelQuantization Level)×Quantization Level
- Quantized Pixel=(Quantization LevelOriginal Pixel​)×Quantization Level

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

## Kernel for Downsampling and Quantization

Each thread operates independently and computes one output pixel. The thread’s goal is to:
- **Maps** the current pixel in the output image to a corresponding pixel in the original image.
- **Qauntize** the intensity value of the original pixel.
- Write the quantized value to the output image array.
