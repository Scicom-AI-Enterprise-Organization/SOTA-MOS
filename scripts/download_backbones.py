"""Pre-download SSL backbones into the HF cache."""
from huggingface_hub import snapshot_download

MODELS = [
    "facebook/wav2vec2-base",
    "facebook/wav2vec2-large-lv60",
    "microsoft/wavlm-base-plus",
    "microsoft/wavlm-large",
    "facebook/data2vec-audio-large",
    "facebook/wav2vec2-xls-r-300m",
    "facebook/wav2vec2-xls-r-1b",
    "facebook/hubert-base-ls960",
    "facebook/hubert-large-ll60k",
]
for m in MODELS:
    p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    files = __import__("os").listdir(p)
    if not any(f.endswith(".safetensors") for f in files):
        p = snapshot_download(m, allow_patterns=["*.json", "pytorch_model.bin", "*.txt"])
    print(m, sorted(__import__("os").listdir(p)), flush=True)
