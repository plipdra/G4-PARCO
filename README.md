# Enhancing Traffic Sign Recognition Performance Using CUDA-Accelerated Hough Transform: A Comparative Study with C Implementation
Group 4 
CO, Sofia Bianca; DATARIO, Yasmin Audrey; EDRALIN, Philippe Nikos Keyan; and PO, Aliexandra Heart 



This study enhances a Traffic Sign Recognition (TSR) application by applying the Hough Transform algorithm through CUDA parallel computing, with an emphasis on execution time comparisons against traditional C implementations. TSR is critical for the reliability of autonomous driving systems, necessitating precise and rapid detection of traffic signs. The problem addressed is the inefficiency of CPU-based TSR systems, which struggle with real-time processing due to limited parallelization capabilities. By leveraging CUDA's GPU-accelerated architecture, this project seeks to significantly reduce computation time by parallelizing the shape detection process. The methodology involves integrating CUDA with a MATLAB application that already performs HSV-based masking and extended border tracing, facilitating direct execution time comparisons between the CUDA-enhanced and C implementations for the Hough Transform Algorithm. This comparative analysis aims to demonstrate the potential of CUDA in accelerating TSR systems, potentially setting a new benchmark for future automated driving technologies.
