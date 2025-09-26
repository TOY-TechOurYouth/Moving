import torch
print("버전:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    print("capability:", torch.cuda.get_device_capability(0))
    print("name:", torch.cuda.get_device_name(0))

    import time
    x = torch.randn(12000, 12000, device="cuda")
    t0 = time.time()
    y = x @ x
    torch.cuda.synchronize()
    print("GPU GEMM OK, secs:", time.time() - t0)
else:
    print("⚠️ CUDA 비활성: 드라이버/설치/가상환경/인덱스 확인 필요")
