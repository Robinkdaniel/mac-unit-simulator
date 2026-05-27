# MAC Unit Simulator — INT8 Quantization

A software simulation of the fundamental building block inside every CNN hardware accelerator — the **Multiply-Accumulate (MAC) unit** — implemented from scratch in Python using only NumPy.

Built to understand how hardware accelerators actually execute neural network inference at the arithmetic level.

---

## What is a MAC unit?

Every convolution layer in a neural network is fundamentally a series of MAC operations:

```
accumulator += weight × input
```

A 3×3 kernel sliding over an image performs **9 MAC operations per output pixel**.

- 9 MACs × ~200×200 image → ~360,000 MACs per conv layer
- ~30 layers in a typical object detection network → ~10 million MACs per frame
- At 30fps → **300 million MAC operations per second**

A CPU executes these sequentially. A hardware PE array executes hundreds simultaneously. That is why dedicated hardware exists.

---

## What this implements

| File | Stage | What it does |
|------|-------|-------------|
| `mac_float32.py` | 1 | Baseline MAC in float32 — ground truth |
| `mac_int8.py` | 2 | INT8 quantization, integer MAC, dequantization |
| `pe_array.py` | 3 | 64 PEs in parallel, float32 vs INT8 comparison |
| `memory_analysis.py` | 4 | Memory footprint: float32 vs INT8 vs INT4 |

---

## Results from my machine

**Stage 2 — Quantization accuracy:**
```
Float32 result:     214.000000
INT8 MAC result:    214.266220
Quantization error: 0.266220
Error %:            0.124%
```

**Stage 3 — Parallel PE array (64 PEs):**
```
Float32 time:         0.0322 ms
INT8 sim time:        3.7911 ms
Max error %:          10.638%
First 5 (float32):   [   2.673 -189.988 -149.437  -29.502 -191.805]
First 5 (int8):      [   2.388 -189.754 -148.912  -30.078 -190.913]
```

**Stage 4 — Memory:**
```
  Float32:  2304 bytes per layer  (2.25 KB)
  INT8:      576 bytes per layer  (0.56 KB)  — 4x smaller
  INT4:      288 bytes per layer  (0.28 KB)  — 8x smaller

  Full 30-layer network:
  Float32:  0.066 MB
  INT8:     0.016 MB
  INT4:     0.008 MB
```

0.124% quantization error. 4x memory reduction. That tradeoff is why real accelerators use INT8.



## How INT8 quantization works

```python
# Compute scale factor
scale = max(abs(values)) / 127.0

# Quantize to INT8
quantized = round(values / scale)   # range [-128, 127]

# MAC in integer arithmetic — INT32 accumulator prevents overflow
accumulator += int32(weight) * int32(input)

# Dequantize
result = accumulator * (weight_scale * input_scale)
```

INT8 × INT8 = INT16 product, accumulated into INT32. This is exactly what a hardware PE implements in silicon.
