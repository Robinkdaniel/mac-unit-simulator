import numpy as np
import time
from mac_int8 import quantize_to_int8, mac_int8

def pe_array_float32(weight_matrix, input_vector):
    return weight_matrix @ input_vector

def pe_array_int8(weight_matrix_f32, input_vector_f32):
    n_pes = weight_matrix_f32.shape[0]
    results = []
    scales = []
    for i in range(n_pes):
        w_int8, w_scale = quantize_to_int8(weight_matrix_f32[i])
        x_int8, x_scale = quantize_to_int8(input_vector_f32)
        result, _ = mac_int8(w_int8, x_int8, w_scale, x_scale)
        results.append(result)
        scales.append(w_scale * x_scale)
    return np.array(results)

N_PES   = 64
KERNEL  = 9

np.random.seed(42)
weights = np.random.uniform(-0.5, 0.5, (N_PES, KERNEL)).astype(np.float32)
inputs  = np.random.uniform(0, 255, KERNEL).astype(np.float32)

t0 = time.perf_counter()
out_f32 = pe_array_float32(weights, inputs)
t_float = time.perf_counter() - t0

t0 = time.perf_counter()
out_int8 = pe_array_int8(weights, inputs)
t_int8 = time.perf_counter() - t0

errors = np.abs(out_f32 - out_int8)
print(f"PE array size:        {N_PES} PEs x {KERNEL} inputs")
print(f"Float32 time:         {t_float*1000:.4f} ms")
print(f"INT8 sim time:        {t_int8*1000:.4f} ms")
print(f"Max error:            {errors.max():.4f}")
print(f"Mean error:           {errors.mean():.4f}")
print(f"Max error %:          {(errors / np.abs(out_f32)).max() * 100:.3f}%")
print(f"\nFirst 5 outputs (float32): {out_f32[:5].round(3)}")
print(f"First 5 outputs (int8):    {out_int8[:5].round(3)}")