# Note - The course code will be updated soon.

# GPU Programming - CUDA and OpenCL Learning Repository

## Motivation

After seeing how GPU computing can dramatically improve performance—especially in projects like DeepSeek—I wanted to dive deeper into CUDA and OpenCL. This repository is a personal learning space where I explore how to write GPU-based programs, optimize them, and understand the internals of parallel computation.

## What This Repository Contains

* Basic CUDA programs and experiments
* Introductory scripts using OpenCL
* Notes and code snippets on performance tuning
* Some benchmarks and comparisons based on my experiments

## Why CUDA and OpenCL?

**CUDA** is NVIDIA’s toolkit for writing programs that run on their GPUs. It’s powerful, widely used in machine learning, simulations, and many scientific applications.

**OpenCL** is an open standard, designed to support parallel programming across a range of devices—including CPUs, GPUs, and even FPGAs. It’s a great way to write platform-independent parallel code.

## Getting Started

You can use Kaggle or Google Colab if you don't have a GPU locally. Here’s a basic setup guide to try CUDA code in a Jupyter notebook:

1. Install CUDA if it’s not already installed.
2. Run the following to enable CUDA in Jupyter:

   ```bash
   pip install nvcc4jupyter
   %load_ext nvcc4jupyter
   ```
3. To check GPU and compiler info:

   ```bash
   !nvidia-smi
   !nvcc --version
   ```

## Learning Goals

* Understand how GPU parallelism works
* Improve speed and efficiency of code
* Learn memory management techniques for GPUs
* Experiment with kernel-level optimizations

## Contributions

If you’re also exploring GPU programming and want to share ideas, improvements, or corrections, feel free to open an issue or a pull request. Always happy to collaborate and learn more.

Thanks for stopping by.

---

Let me know if you want to add sections like "Recommended Resources," "Project Ideas," or tutorials.
