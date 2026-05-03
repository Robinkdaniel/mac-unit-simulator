import numpy as np

def mac_floats(weight, inputs):
    assert len(weight) == len(inputs)
    result = 0.0
    for w, x in zip(weight, inputs):
        result += w * x
        
    return result

weight = np.array([0.1, -0.3, 0.5, 0.2, -0.1, 0.4, -0.2, 0.3, 0.1], dtype=np.float32)
inputs  = np.array([120, 80, 200, 150, 90, 170, 60, 130, 100], dtype=np.float32)

result = mac_floats(weight, inputs)
print(f"mac_float result is: {result:.6f}")