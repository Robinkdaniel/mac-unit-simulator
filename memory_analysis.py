import numpy as np

def memory_analysis(n_pes, kernel_size, n_layers=30):
    weights_per_layer = n_pes * kernel_size
    float32_bytes = weights_per_layer * 4
    int8_bytes    = weights_per_layer * 1
    int4_bytes    = weights_per_layer * 0.5
    print(f"{'':=<50}")
    print(f"  Memory Analysis — {n_pes} PEs x {kernel_size} kernel")
    print(f"{'':=<50}")
    print(f"  Per layer:")
    print(f"    Float32:  {float32_bytes:>8} bytes  ({float32_bytes/1024:.2f} KB)")
    print(f"    INT8:     {int8_bytes:>8} bytes  ({int8_bytes/1024:.2f} KB)  — {float32_bytes//int8_bytes}x smaller")
    print(f"    INT4:     {int4_bytes:>8.0f} bytes  ({int4_bytes/1024:.2f} KB)  — {int(float32_bytes//int4_bytes)}x smaller")
    print(f"  Full network ({n_layers} layers):")
    print(f"    Float32:  {float32_bytes*n_layers/1024/1024:.3f} MB")
    print(f"    INT8:     {int8_bytes*n_layers/1024/1024:.3f} MB")
    print(f"    INT4:     {int4_bytes*n_layers/1024/1024:.3f} MB")
    print(f"{'':=<50}")

memory_analysis(n_pes=64, kernel_size=9, n_layers=30)