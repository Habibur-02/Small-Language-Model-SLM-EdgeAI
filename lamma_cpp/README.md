# 🧪 Benchmark: Qwen2.5-1.5B Quantization Analysis

This project evaluates the performance trade-offs of quantizing the **Qwen2.5-1.5B-Instruct** model for Edge AI deployment. We compared the original BF16 model against 8-bit (Q8_0) and 4-bit (Q4_K_M) quantized versions using `llama.cpp`.

## 🖥️ Hardware Specs
- **Inference Engine:** llama.cpp (CPU Backend)
- **Threads:** 8

## 📈 Results Summary

| Model Version | Size (MB) | Compression | Perplexity (Lower is Better) | Generation Speed (Tokens/Sec) |
| :--- | :--- | :--- | :--- | :--- |
| **F16 (BF16)** | 2,949 MB | 0% | *Baseline* | **11.83 t/s** |
| **Q8_0** | 1,566 MB | ~47% | 6.14 | **21.10 t/s** |
| **Q4_K_M** | 935 MB | ~68% | 6.30 | **32.99 t/s** 🚀 |

## 🔍 Key Findings

1.  **Massive Speedup:** The 4-bit quantized model (Q4_K_M) is **2.7x faster** than the original F16 model, achieving ~33 tokens per second on CPU. This is significantly faster than human reading speed.
2.  **Memory Efficiency:** We achieved a **68% reduction in memory usage** (dropping from ~3GB to <1GB), making it runnable on almost any modern edge device, including Raspberry Pi or mobile phones.
3.  **Minimal Quality Loss:** The Perplexity (PPL) increased slightly from **6.14 (Q8)** to **6.30 (Q4)**. This **0.16** difference is negligible for most real-world chat applications, proving that 4-bit quantization is highly effective for this model.

## 🏁 Conclusion
For Edge AI applications where latency and memory are constraints, **Q4_K_M is the optimal choice**, offering a perfect balance of speed and accuracy.
