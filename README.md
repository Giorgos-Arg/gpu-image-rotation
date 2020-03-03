# gpu-image-rotation

This CUDA program performs a clockwise 90-degree rotation of a given image with efficient use of GPU capabilities.

## Description

The program takes as input a [.ppm (Netpbm Format)](https://en.wikipedia.org/wiki/Netpbm) image. 
For each pixel there are three 8-bit values for red, green, and blue channels. 

## Usage

Compilation:

```bash
nvcc -o rotate_img.exe rotate_img.cu rotate_img.cu
```

Run:

```bash
.\rotate_img.exe <input_image> <kernel_choice>
```

Where:

| kernel_choice | Description                                                                       |
| ------------- | --------------------------------------------------------------------------------- |
| 1             | Kernel 1 - Serial execution on GPU (only one thread)                              |
| 2             | Kernel 2 - One thread per pixel                                                   |
| 3             | Kernel 3 - One thread per pixel - tiled (16 X 16)                                 |
| 4             | Kernel 4 - One thread per matrix element - tiled (16x16) - no shared mem conflict |

## Example

```bash
.\rotate_img.exe img\luna.ppm 3  

Rotating Image "img\luna.ppm" with Height = 853 and Width = 1280.
Using kernel 3: One thread per pixel - tiled (16 X 16)
Elapsed time: 0.345664 ms.
```

# Kernel performance

| kernel_choice | Kernel run-time (ms) |
| ------------- | -------------------- |
| 1             | 373.0188             |
| 2             | 0.4084544            |
| 3             | 0.2894208            |
| 4             | 0.2891072            |

*run-time taken using a 1280 x 720 image
## Author

Giorgos Argyrides (g.aryrides@outlook.com)