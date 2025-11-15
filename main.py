from fastapi import FastAPI, UploadFile, HTTPException, File
import numpy as np
from numba import cuda
import time
import uvicorn
import socket
import random
import subprocess

app = FastAPI()

# ---------------- CUDA Kernel ----------------
@cuda.jit
def add_matrices_gpu(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        C[i, j] = A[i, j] + B[i, j]

# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/add")
async def add_matrices_endpoint(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    try:
        # Load uploaded matrices
        matA = np.load(file_a.file)
        matB = np.load(file_b.file)

        # Extract first array if .npz has keys
        if isinstance(matA, np.lib.npyio.NpzFile):
            matA = matA[list(matA.keys())[0]]
        if isinstance(matB, np.lib.npyio.NpzFile):
            matB = matB[list(matB.keys())[0]]

        # Validate shapes
        if matA.shape != matB.shape:
            raise HTTPException(status_code=400, detail="Matrices must have the same shape")

        start = time.time()
        device_used = "CPU"

        # Use GPU if available
        if cuda.is_available():
            d_A = cuda.to_device(matA)
            d_B = cuda.to_device(matB)
            d_C = cuda.device_array_like(matA)

            threads = (16, 16)
            blocks = ((matA.shape[0] + threads[0] - 1)//threads[0],
                      (matA.shape[1] + threads[1] - 1)//threads[1])

            add_matrices_gpu[blocks, threads](d_A, d_B, d_C)
            cuda.synchronize()
            result = d_C.copy_to_host()
            device_used = "GPU"
        else:
            # CPU fallback
            result = matA + matB

        elapsed = time.time() - start

        return {
            "matrix_shape": list(matA.shape),
            "elapsed_time_sec": elapsed,
            "device": device_used,
            "result_sample": result.flatten()[:10].tolist()  # first 10 elements as sample
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpu-info")
def gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            idx, used, total = line.split(", ")
            gpus.append({
                "gpu": idx,
                "memory_used_MB": int(used),
                "memory_total_MB": int(total)
            })
        return {"gpus": gpus}
    except Exception as e:
        return {"error": str(e)}


# ---------------- Helper: random free port ----------------
def get_free_port():
    while True:
        port = random.randint(8000, 9000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port


# ---------------- Run server ----------------
if __name__ == "__main__":
    port = 8610  # fixed port for easier testing
    print(f"Running FastAPI on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
