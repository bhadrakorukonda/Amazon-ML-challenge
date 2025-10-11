import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(4096,4096, device="cuda")
    y = x @ x.t()
    print("OK GPU sum:", float(y.sum()))
