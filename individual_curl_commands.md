# Individual Curl Commands for Model Downloads

## Prerequisites
First, create the necessary directories:
```bash
mkdir -p models
mkdir -p loras
```

## Stable Diffusion Models

### ToonYou Model (Cartoon Style)
```bash
curl -L "https://huggingface.co/ckpt/ToonYou/resolve/main/ToonYou_beta6.safetensors" \
     -o "models/toonyou_beta6.safetensors" \
     --progress-bar
```

### MeinaMix Model (Anime Style)
```bash
curl -L "https://huggingface.co/Meina/MeinaMix/resolve/main/MeinaMix.safetensors" \
     -o "models/meina_mix.safetensors" \
     --progress-bar
```

## AnimateDiff Models

### AnimateDiff v1.5 Main Model
```bash
curl -L "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt" \
     -o "models/animatediff_v1-5-pruned.ckpt" \
     --progress-bar
```

### AnimateDiff Motion Module
```bash
curl -L "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt" \
     -o "models/mm_sd_v15_v2.ckpt" \
     --progress-bar
```

## LoRA Models

### Cartoon LoRA
```bash
curl -L "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors" \
     -o "loras/animov.safetensors" \
     --progress-bar
```



## Windows PowerShell Commands

If you're using PowerShell on Windows, use `Invoke-WebRequest` instead:

### ToonYou Model
```powershell
Invoke-WebRequest -Uri "https://huggingface.co/ckpt/ToonYou/resolve/main/ToonYou_beta6.safetensors" -OutFile "models/toonyou_beta6.safetensors"
```

### MeinaMix Model
```powershell
Invoke-WebRequest -Uri "https://huggingface.co/Meina/MeinaMix/resolve/main/MeinaMix.safetensors" -OutFile "models/meina_mix.safetensors"
```

### AnimateDiff v1.5
```powershell
Invoke-WebRequest -Uri "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt" -OutFile "models/animatediff_v1-5-pruned.ckpt"
```

### AnimateDiff Motion Module
```powershell
Invoke-WebRequest -Uri "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt" -OutFile "models/mm_sd_v15_v2.ckpt"
```

### Cartoon LoRA
```powershell
Invoke-WebRequest -Uri "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors" -OutFile "loras/animov.safetensors"
```



## Model Sizes (Approximate)
- ToonYou: ~2.3 GB
- MeinaMix: ~2.3 GB  
- AnimateDiff v1.5: ~1.7 GB
- Motion Module: ~1.8 GB
- Cartoon LoRA: ~144 MB

**Total download size: ~6.0 GB**

## Notes
- Make sure you have sufficient disk space before downloading
- Downloads may take time depending on your internet connection
- Some SharePoint links may require authentication or may change over time
- If any download fails, try running the command again or check the URLs
