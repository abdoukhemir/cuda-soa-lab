from numba import cuda
import numpy as np
import time

# Create sample matrices
N, M = 512, 512
A = np.random.rand(N, M).astype(np.float32)
B = np.random.rand(N, M).astype(np.float32)
C = np.zeros_like(A)

if cuda.is_available():
    print("CUDA GPU detected!")
    @cuda.jit
    def add_kernel(A, B, C):
        i, j = cuda.grid(2)
        if i < C.shape[0] and j < C.shape[1]:
            C[i, j] = A[i, j] + B[i, j]

    # Transfer to GPU and run kernel
    threads = (16, 16)
    blocks = ((N + threads[0] - 1) // threads[0],
              (M + threads[1] - 1) // threads[1])

    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array_like(A)

    start = time.time()
    add_kernel[blocks, threads](d_A, d_B, d_C)
    cuda.synchronize()
    elapsed = time.time() - start

    result = d_C.copy_to_host()
else:
    print("CUDA GPU not available. Running on CPU.")
    start = time.time()
    result = A + B
    elapsed = time.time() - start

print(f"Addition completed in {elapsed:.6f} sec")
print("Sample result:", result.flatten()[:10])
