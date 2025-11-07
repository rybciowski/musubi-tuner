# Musubi-Tuner na Modal.com - V2 (Auto-Download)

## 🆕 Co Nowego w V2?

**Automatyczne pobieranie modeli z HuggingFace!**

- ❌ **PRZED:** Musisz ręcznie uploadować 40GB+ modeli do volume
- ✅ **TERAZ:** Modele pobierają się automatycznie przy pierwszym użyciu
- ✅ Cache'owane w volume - kolejne runy są instant!

---

## 🚀 Quickstart (Super Prosty!)

```bash
# 1. Setup Modal (5 min)
pip install modal
modal setup
modal volume create musubi-models
modal volume create musubi-data
modal volume create musubi-cache

# 2. Upload tylko datasetu (małe pliki)
modal volume put musubi-data my_dataset/ /local/path/to/dataset/

# 3. Stwórz config (używa standardowych ścieżek)
cat > my_config.json <<EOF
{
  "dit_path": "wan/wan_t2v_14B_bf16.safetensors",
  "vae_path": "wan/wan_2.1_vae.safetensors",
  "t5_path": "wan/models_t5_umt5-xxl-enc-bf16.pth",
  "clip_path": "wan/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
  "dataset_config": "my_dataset/config.toml",
  "output_name": "my_first_lora",
  "network_dim": 32,
  "network_alpha": 16,
  "learning_rate": "1e-4",
  "max_train_epochs": 10
}
EOF

# 4. RUN - modele pobiorą się automatycznie!
modal run run_modal_v2.py --model wan --config my_config.json
```

**Pierwszy run:** modele pobiorą się z HuggingFace (~15-20 min dla 40GB)
**Kolejne runy:** modele już są w cache (instant!)

---

## 📦 Wspierane Modele (Auto-Download)

### Wan 2.1
- `wan/wan_t2v_14B_bf16.safetensors` - DiT T2V
- `wan/wan_i2v_14B_bf16.safetensors` - DiT I2V
- `wan/wan_2.1_vae.safetensors` - VAE
- `wan/models_t5_umt5-xxl-enc-bf16.pth` - T5 Text Encoder
- `wan/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` - CLIP

### Wan 2.2
- `wan/wan_2.2_t2v_14B_low_noise_bf16.safetensors` - DiT Low Noise
- `wan/wan_2.2_t2v_14B_high_noise_bf16.safetensors` - DiT High Noise
- (VAE, T5 takie same jak Wan 2.1)

### Qwen-Image
- `qwen/qwen_image_bf16.safetensors` - DiT Standard
- `qwen/qwen_image_edit_bf16.safetensors` - DiT Edit
- `qwen/qwen_image_edit_2509_bf16.safetensors` - DiT Edit-2509
- `qwen/vae_diffusion_pytorch_model.safetensors` - VAE
- `qwen/qwen_2.5_vl_7b.safetensors` - Text Encoder (Qwen2.5-VL)

**Wszystkie te modele pobierają się automatycznie z HuggingFace!**

---

## 🔄 Workflow

```bash
# PIERWSZY RUN (modele się pobiorą)
modal run run_modal_v2.py --model wan --config config.json

Output:
🔍 Checking if models are available...
📥 Downloading model from HuggingFace: Comfy-Org/Wan_2.1_ComfyUI_repackaged
   Files: wan_t2v_14B_bf16.safetensors
   ████████████████ 27GB in 8 min
   ✓ Saved to: /models/wan/wan_t2v_14B_bf16.safetensors
📥 Downloading model from HuggingFace: ...
   (repeat dla każdego modelu)
✓ All models downloaded and cached!

# Cache + Training jak zwykle...

# KOLEJNY RUN (modele już są!)
modal run run_modal_v2.py --model wan --config config2.json

Output:
🔍 Checking if models are available...
✓ Model already cached: wan/wan_t2v_14B_bf16.safetensors  <- instant!
✓ Model already cached: wan/wan_2.1_vae.safetensors
✓ Model already cached: wan/models_t5_umt5-xxl-enc-bf16.pth
✓ All models ready!

# Cache + Training - szybsze bo modele już są!
```

---

## 📋 Przykładowe Configi

### Wan 2.1 T2V
```json
{
  "task": "wan_t2v_14B",
  "dit_path": "wan/wan_t2v_14B_bf16.safetensors",
  "vae_path": "wan/wan_2.1_vae.safetensors",
  "t5_path": "wan/models_t5_umt5-xxl-enc-bf16.pth",
  "clip_path": "wan/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
  "dataset_config": "my_dataset/config.toml",
  "output_name": "my_wan_lora",
  "network_dim": 32,
  "network_alpha": 16,
  "learning_rate": "1e-4",
  "max_train_epochs": 10,
  "blocks_to_swap": 8,
  "seed": 42
}
```

### Wan 2.2 (Dual DiT)
```json
{
  "task": "wan_t2v_14B",
  "dit_path": "wan/wan_2.2_t2v_14B_low_noise_bf16.safetensors",
  "dit_high_noise_path": "wan/wan_2.2_t2v_14B_high_noise_bf16.safetensors",
  "timestep_boundary": 0.3,
  "vae_path": "wan/wan_2.1_vae.safetensors",
  "t5_path": "wan/models_t5_umt5-xxl-enc-bf16.pth",
  "dataset_config": "my_dataset/config.toml",
  "output_name": "my_wan22_lora",
  "network_dim": 32,
  "learning_rate": "1e-4",
  "max_train_epochs": 10
}
```

### Qwen-Image
```json
{
  "dit_path": "qwen/qwen_image_bf16.safetensors",
  "vae_path": "qwen/vae_diffusion_pytorch_model.safetensors",
  "text_encoder_path": "qwen/qwen_2.5_vl_7b.safetensors",
  "dataset_config": "my_dataset/config.toml",
  "output_name": "my_qwen_lora",
  "network_dim": 16,
  "network_alpha": 8,
  "learning_rate": "5e-5",
  "max_train_epochs": 16
}
```

---

## ⚙️ Zaawansowane

### Custom Modele (nie w registry)

Jeśli używasz custom modeli NIE z HuggingFace:

**Opcja 1:** Upload ręcznie (jak V1)
```bash
modal volume put musubi-models my_custom_model/ ./local/path/
```

**Opcja 2:** Dodaj do registry w `run_modal_v2.py`:
```python
MODEL_REGISTRY = {
    "my_custom_model": {
        "repo_id": "username/repo",
        "files": ["model.safetensors"],
        "cache_path": "custom/model.safetensors"
    },
}
```

### Force Re-Download

Jeśli chcesz ponownie pobrać model:

```bash
# Usuń z volume
modal volume rm musubi-models wan/wan_t2v_14B_bf16.safetensors

# Następny run pobierze go ponownie
modal run run_modal_v2.py --model wan --config config.json
```

---

## 💰 Koszty

**Pierwszy run (z download):**
- Download: gratis (nie płacisz za network)
- Cache: ~30 min A10G = $0.30
- Training: ~2.5h A100-80GB = $7.50
- **Total: ~$7.80**

**Kolejne runy (modele cache'd):**
- Download: 0 (cache'd!)
- Cache: ~30 min A10G = $0.30
- Training: ~2.5h A100-80GB = $7.50
- **Total: ~$7.80**

Identyczne koszty, ale setup jest znacznie prostszy!

---

## 🆚 V1 vs V2

| | V1 (Manual) | V2 (Auto-Download) |
|---|---|---|
| **Setup** | Upload 40GB+ modeli | Tylko volumes |
| **Pierwszy run** | Instant (modele już są) | +15 min (download) |
| **Kolejne runy** | Instant | Instant (cache'd) |
| **Custom modele** | ✅ Pełna kontrola | ⚠️ Manual lub edit registry |
| **Łatwość** | 🔴 Średnia | 🟢 Bardzo łatwa |
| **Kiedy użyć** | Custom modele | Standardowe modele |

**Rekomendacja:** Użyj V2 dla standardowych modeli (90% przypadków)

---

## 🎯 Kroki w Skrócie

```bash
# SETUP (raz)
pip install modal && modal setup
modal volume create musubi-models musubi-data musubi-cache

# TRENING (za każdym razem)
modal volume put musubi-data my_dataset/ ./dataset/
nano config.json  # standardowe ścieżki jak w przykładach
modal run run_modal_v2.py --model wan --config config.json
modal volume get musubi-models outputs/my_lora ./

# DONE! 🎉
```

---

## ❓ FAQ

**Q: Czy muszę mieć modele lokalnie?**
A: NIE! To jest cały punkt V2 - pobierają się automatycznie.

**Q: Gdzie modele się zapisują?**
A: W volume `musubi-models`, są tam na zawsze (do ręcznego usunięcia).

**Q: Czy mogę wybrać inne wersje modeli?**
A: Edytuj `MODEL_REGISTRY` w `run_modal_v2.py` i zmień `repo_id` lub `files`.

**Q: Co jeśli model nie jest w registry?**
A: Upload ręcznie: `modal volume put musubi-models ...` lub dodaj do registry.

**Q: Czy pierwszt download kosztuje?**
A: NIE! Network/bandwidth jest gratis. Płacisz tylko za GPU time.

---

Gotowe! 🚀
