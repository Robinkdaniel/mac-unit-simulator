import numpy as np

def quantize_to_int8(values, scale = None):
    
    if scale is None:
        scale = np.max(np.abs(values)) / 127
        
    quantized = np.round(values/scale).astype(np.int8)
        
    quantized = np.clip(quantized, -128,127).astype(np.int8)
    return quantized, scale

def dequantize(quantized, scale):
    return quantized.astype(np.float32) * scale
        
def mac_int8(weights_int8, inputs_int8, weights_scale, inputs_scale):
    accumulator = np.int32(0)      
    
    for w, x in zip(weights_int8, inputs_int8):
        accumulator += np.int32(w) * np.int32(x)
    output_scale = weights_scale * inputs_scale  
    result_float = float(accumulator) * output_scale
    return result_float, accumulator

weights = np.array([0.1, -0.3, 0.5, 0.2, -0.1, 0.4, -0.2, 0.3, 0.1], dtype=np.float32)
inputs  = np.array([120, 80, 200, 150, 90, 170, 60, 130, 100], dtype=np.float32)    

weights_int8, weights_scale = quantize_to_int8(weights)
inputs_int8, inputs_scale = quantize_to_int8(inputs)    

print("Original weights:   ", weights)
print("INT8 weights:       ", weights_int8)
print("Weight scale:       ", round(weights_scale, 6))
print()
print("Original inputs:    ", inputs)
print("INT8 inputs:        ", inputs_int8)
print("Input scale:        ", round(inputs_scale, 6))
print()


result_int8, raw_accumulator = mac_int8(weights_int8, inputs_int8, weights_scale, inputs_scale)

print(f"INT32 accumulator:  {raw_accumulator}")
print(f"INT8 MAC result:    {result_int8:.6f}")


result_float32 = sum(w * x for w, x in zip(weights, inputs))
print(f"Float32 result:     {result_float32:.6f}")
print(f"Quantization error: {abs(result_float32 - result_int8):.6f}")
print(f"Error %:            {abs(result_float32 - result_int8) / abs(result_float32) * 100:.3f}%")