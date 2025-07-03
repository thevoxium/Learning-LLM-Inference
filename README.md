# Learning-LLM-Inference

## Month 1: Foundations & Core Techniques

### Week 1-2: Mathematical & Architectural Foundations

**Papers & Theory**

-   "Attention Is All You Need" (Vaswani et al.) - implement from scratch
-   "The Annotated Transformer" - Complete tutorial
-   "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al.)
-   "FlashAttention-2: Faster Attention with Better Parallelism"
-   "Multi-Query Attention" (Shazeer)
-   "GQA: Training Generalized Multi-Query Transformer Models" (Ainslie et al.)

**Resources:**

-   Andrej Karpathy's "Let's Build GPT" YouTube series
-   Jay Alammar's "The Illustrated Transformer" blog series
-   Harvard NLP "The Annotated Transformer"
-   3Blue1Brown's attention mechanism videos

**Implementation**

-   **Project 1**: Pure JAX transformer from scratch (no libraries)
-   **Project 2**: Implement FlashAttention algorithm in JAX
-   Benchmark memory usage and speed differences

### Week 3-4: Quantization & Compression

**Papers & Theory**

-   "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (Dettmers et al.)
-   "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
-   "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
-   "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
-   "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al.)

**Resources:**

-   Hugging Face Quantization Documentation
-   NVIDIA TensorRT-LLM documentation
-   Tim Dettmers' blog posts on quantization

**Implementation**

-   **Project 3**: Implement INT8 quantization in JAX
-   **Project 4**: Build GPTQ quantization algorithm
-   Compare perplexity vs speed trade-offs across different bit widths

## Month 2: Advanced Inference & Systems

### Week 5-6: KV-Cache & Memory Optimization

**Papers & Theory**

-   "Efficiently Scaling Transformer Inference" (Pope et al.)
-   "Multi-Head Attention: Collaborate Instead of Concatenate"
-   "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"
-   "Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression"
-   "PagedAttention" from vLLM paper

**Resources:**

-   vLLM GitHub repository and documentation
-   DeepSpeed-Inference documentation
-   NVIDIA FasterTransformer repository

**Implementation**

-   **Project 5**: Implement efficient KV-cache with paging in JAX
-   **Project 6**: Build KV-cache compression algorithms
-   Memory profiling and optimization

### Week 7-8: Speculative Decoding & Parallel Strategies

**Papers & Theory**

-   "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al.)
-   "SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference"
-   "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"
-   "Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding"

**Resources:**

-   Google Research blog on speculative decoding
-   NVIDIA Triton documentation

**Implementation**

-   **Project 7**: Implement speculative decoding with draft model
-   **Project 8**: Build parallel sampling strategies
-   Benchmark latency improvements

## Month 3: Production Systems & Advanced Topics

### Week 9-10: Kernel Optimization & CUDA

**Switch to CUDA focus here**

**Papers & Theory**

-   "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"
-   Study CUTLASS documentation thoroughly
-   "FasterTransformer: an Efficient PyTorch Implementation"
-   NVIDIA cuBLAS and cuDNN documentation

**Resources:**

-   NVIDIA Deep Learning Performance documentation
-   Triton tutorials and examples
-   CUDA C++ Programming Guide
-   "Programming Massively Parallel Processors" book (relevant chapters)

**Implementation**

-   **Project 9**: Write custom CUDA kernels for attention
-   Implement fused attention kernels using Triton
-   Profile and optimize kernel performance

### Week 11-12: Production Serving & Advanced Research

**Papers & Theory**

-   "Orca: A Distributed Serving System for Transformer-Based Generative Models"
-   "AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving"
-   Latest papers from ICLR/NeurIPS 2024/2025 on inference optimization
-   "Mixture of Experts" inference optimization papers

**Resources:**

-   Ray Serve documentation
-   TorchServe advanced guides
-   Kubernetes for ML serving
-   Recent conference proceedings (MLSys, OSDI, SOSP)

**Implementation**

-   **Project 10**: Build complete inference serving system
-   Implement request batching and scheduling
-   Deploy optimized models with monitoring


# 10 Projects

1.  **Pure JAX Transformer** - No libraries, full attention mechanism
2.  **FlashAttention Implementation** - Memory-efficient attention in JAX
3.  **INT8 Quantization Engine** - Post-training quantization with calibration
4.  **GPTQ Quantizer** - Advanced weight quantization algorithm
5.  **KV-Cache Manager** - Efficient cache with compression and paging
6.  **KV-Cache Compression** - Heavy-hitter oracle and pruning strategies
7.  **Speculative Decoding Framework** - Draft-verify with multiple strategies
8.  **Parallel Sampling System** - Beam search and nucleus sampling optimizations
9.  **Custom CUDA Kernels** - Fused attention kernels and memory optimizations
10.  **Production Serving System** - Complete inference server with batching, monitoring, and scaling
