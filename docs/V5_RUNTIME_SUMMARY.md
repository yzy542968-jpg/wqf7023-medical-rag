# V5 Observed Runtime and Resource Usage

This is a limited computational-cost summary from already frozen runtime artifacts. It is not a complete component-wise profile of the full Agent.

## V5 Qwen generation

| Pipeline condition | Records | Total process | Generation only | Generation throughput | Peak allocated GPU memory | Peak reserved GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| Report-only | 360 | 87.86 s | 78.56 s | 4.58 records/s | 3,437 MiB | 4,056 MiB |
| Multimodal | 360 | 98.70 s | 89.31 s | 4.03 records/s | 3,437 MiB | 4,058 MiB |

Both runs used local `Qwen/Qwen2.5-1.5B-Instruct`, CUDA, float16, batch size 16, maximum 256 new tokens, and temperature 0 on an NVIDIA GeForce RTX 5070 Laptop GPU. These timings are machine-, cache-, and generated-length-dependent.

## Previously recorded V4.2 retrieval timing

The earlier V4.2 runtime artifact recorded approximately 6.51 s model cold load, 526 MiB loaded model memory, 14.91 ms mean single-image encoding, 1.73 ms BM25 retrieval, 0.28 ms cached similarity plus reranking, and a 16.93 ms warm paired-request estimate on the same local GPU.

## Interpretation boundary

The available artifacts support reporting selected generation and retrieval costs. They do not support a claim of complete end-to-end component-wise latency, energy use, production throughput, or clinical deployment cost.
