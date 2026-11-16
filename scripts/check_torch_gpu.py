import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    print("gpu name:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("total VRAM (GB):", round(props.total_memory / 1024**3, 2))
else:
    print("GPU не обнаружена, запуск будет на CPU.")