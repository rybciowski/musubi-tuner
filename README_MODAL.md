# Musubi-Tuner na Modal.com

Kompletny setup do trenowania modeli LoRA dla Wan 2.1/2.2 i Qwen-Image na GPU w chmurze Modal.com.

## 🚀 Szybki Start

### 1. Instalacja Modal CLI

```bash
pip install modal
modal setup
```

Zaloguj się gdy zostaniesz poproszony.

### 2. Przygotowanie Modeli

Upload modeli bazowych do Modal Volume:

```bash
# Stwórz volume (jeśli nie istnieje)
modal volume create musubi-models

# Upload modeli Wan
modal volume put musubi-models wan/ /path/to/local/wan_models/

# Upload modeli Qwen
modal volume put musubi-models qwen/ /path/to/local/qwen_models/
```

Struktura w volume:
```
musubi-models/
├── wan/
│   ├── wan_t2v_14B_bf16.safetensors
│   ├── wan_2.1_vae.safetensors
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   └── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
├── qwen/
│   ├── qwen_image_bf16.safetensors
│   ├── vae_diffusion_pytorch_model.safetensors
│   └── qwen_2.5_vl_7b.safetensors
└── outputs/  # tutaj będą zapisywane wytrenowane LoRA
```

### 3. Przygotowanie Datasetu

Upload datasetu do Modal Volume:

```bash
# Stwórz volume
modal volume create musubi-data

# Upload datasetu z konfiguracją TOML
modal volume put musubi-data my_dataset/ /path/to/local/dataset/
```

Struktura:
```
musubi-data/
├── my_dataset/
│   ├── config.toml          # Konfiguracja dataset musubi-tuner
│   ├── images/              # lub videos/
│   │   ├── img001.jpg
│   │   ├── img001.txt       # caption
│   │   ├── img002.jpg
│   │   └── img002.txt
```

### 4. Konfiguracja HuggingFace (opcjonalne)

Jeśli używasz modeli z HuggingFace wymagających tokena:

```bash
modal secret create huggingface-secret HF_TOKEN=your_hf_token_here
```

### 5. Stwórz Config JSON

Przykład dla Wan 2.1:

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
  "optimizer_type": "adamw8bit",
  "max_train_epochs": 10,
  "save_every_n_epochs": 1,
  "blocks_to_swap": 8,
  "seed": 42
}
```

Przykład dla Wan 2.2:

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
  "network_alpha": 16,
  "learning_rate": "1e-4",
  "max_train_epochs": 10,
  "blocks_to_swap": 8,
  "seed": 42
}
```

Przykład dla Qwen-Image:

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
  "optimizer_type": "adamw8bit",
  "max_train_epochs": 16,
  "save_every_n_epochs": 1,
  "timestep_sampling": "shift",
  "weighting_scheme": "none",
  "discrete_flow_shift": 2.2,
  "seed": 42
}
```

Przykład dla Qwen-Image-Edit:

```json
{
  "dit_path": "qwen/qwen_image_edit_bf16.safetensors",
  "vae_path": "qwen/vae_diffusion_pytorch_model.safetensors",
  "text_encoder_path": "qwen/qwen_2.5_vl_7b.safetensors",
  "dataset_config": "my_dataset/config.toml",
  "output_name": "my_qwen_edit_lora",
  "edit": true,
  "network_dim": 16,
  "network_alpha": 8,
  "learning_rate": "5e-5",
  "max_train_epochs": 16,
  "seed": 42
}
```

### 6. Uruchom Trening

```bash
# Pełny pipeline (cache + train)
modal run run_modal.py --model wan --config config.json

# Lub dla Qwen
modal run run_modal.py --model qwen --config config.json
```

### 7. Download Wyników

```bash
# Zobacz co jest w volume
modal volume ls musubi-models outputs/

# Download konkretnego modelu
modal volume get musubi-models outputs/my_wan_lora ./local_outputs/
```

---

## 📋 Zaawansowane Użycie

### Uruchomienie Tylko Cache

```bash
modal run run_modal.py --model wan --config config.json --stage cache
```

### Uruchomienie Tylko Trainingu (po cache)

```bash
modal run run_modal.py --model wan --config config.json --stage train
```

### Monitoring

Logi w czasie rzeczywistym w terminalu. Dodatkowo możesz sprawdzić:

```bash
# Historia uruchomień
modal app list

# Logi z konkretnego uruchomienia
modal app logs musubi-tuner
```

Lub przez Web UI: https://modal.com/apps

---

## 🎛️ Parametry Konfiguracji

### Wspólne dla Wszystkich

| Parametr | Opis | Wymagany |
|----------|------|----------|
| `dataset_config` | Ścieżka do TOML w volume | ✅ |
| `output_name` | Nazwa wyjściowego LoRA | ✅ |
| `network_dim` | Rank LoRA | ✅ |
| `network_alpha` | Alpha LoRA | ✅ |
| `learning_rate` | Learning rate | ✅ |
| `max_train_epochs` | Liczba epok | ✅ |
| `optimizer_type` | "adamw8bit", "prodigy", etc. | Nie (default: adamw8bit) |
| `save_every_n_epochs` | Co ile epok zapisywać | Nie (default: 1) |
| `seed` | Random seed | Nie |
| `additional_args` | Lista dodatkowych argumentów CLI | Nie |

### Specyficzne dla Wan

| Parametr | Opis | Wymagany |
|----------|------|----------|
| `task` | "wan_t2v_14B", "wan_i2v_14B", etc. | ✅ |
| `dit_path` | Ścieżka do DiT w volume | ✅ |
| `vae_path` | Ścieżka do VAE w volume | ✅ |
| `t5_path` | Ścieżka do T5 w volume | ✅ |
| `clip_path` | Ścieżka do CLIP (tylko Wan 2.1) | Nie |
| `dit_high_noise_path` | High noise DiT (tylko Wan 2.2) | Nie |
| `timestep_boundary` | Boundary dla Wan 2.2 (0-1) | Nie |
| `i2v` | Image-to-video mode | Nie (default: false) |
| `blocks_to_swap` | Ile bloków offload do CPU | Nie |
| `fp8_base` | Użyj fp8 dla DiT | Nie (default: false) |
| `fp8_llm` | Użyj fp8 dla T5 | Nie (default: false) |
| `fp8_t5` | Użyj fp8 dla T5 podczas cache | Nie (default: false) |
| `vae_cache_cpu` | Cache VAE na CPU | Nie (default: false) |
| `cache_batch_size` | Batch size dla cache T5 | Nie (default: 16) |

### Specyficzne dla Qwen-Image

| Parametr | Opis | Wymagany |
|----------|------|----------|
| `dit_path` | Ścieżka do DiT w volume | ✅ |
| `vae_path` | Ścieżka do VAE w volume | ✅ |
| `text_encoder_path` | Ścieżka do Qwen2.5-VL w volume | ✅ |
| `edit` | Qwen-Image-Edit mode | Nie (default: false) |
| `edit_plus` | Qwen-Image-Edit-2509 mode | Nie (default: false) |
| `fp8_vl` | Użyj fp8 dla VL | Nie (default: false) |
| `cache_batch_size` | Batch size dla cache (zwykle 1) | Nie (default: 1) |
| `timestep_sampling` | "shift" lub inny | Nie (default: shift) |
| `weighting_scheme` | Schemat wag | Nie (default: none) |
| `discrete_flow_shift` | Flow shift value | Nie (default: 2.2) |

---

## 💰 Koszty

Szacunkowe koszty Modal.com (2025):

| GPU | $/godzina | Użycie |
|-----|-----------|--------|
| A10G | ~$0.60 | Cache, testy, małe treningi |
| A100-40GB | ~$2.50 | Standardowy trening |
| A100-80GB | ~$3.00 | Duże modele, większy batch size |
| H100 | ~$5.00 | Najszybszy trening |

Przykład: Trening 10 epok Wan na A100-80GB ≈ $25-30

Płacisz tylko za rzeczywiste użycie GPU (per second billing).

---

## 🔧 Zarządzanie Volumes

### Lista Volumes

```bash
modal volume list
```

### Lista Plików w Volume

```bash
modal volume ls musubi-models
modal volume ls musubi-data
```

### Upload Plików

```bash
# Pojedynczy plik
modal volume put musubi-models local_file.safetensors remote_path/file.safetensors

# Cały folder
modal volume put musubi-models local_folder/ remote_folder/
```

### Download Plików

```bash
# Pojedynczy plik
modal volume get musubi-models remote_path/file.safetensors ./local_file.safetensors

# Cały folder
modal volume get musubi-models remote_folder/ ./local_folder/
```

### Usuwanie Plików

```bash
modal volume rm musubi-models outputs/old_lora/
```

---

## 🐛 Troubleshooting

### Błąd: "Secret huggingface-secret not found"

Jeśli nie używasz modeli wymagających HuggingFace token:
- Ignoruj to ostrzeżenie (secret jest optional)

Jeśli używasz:
```bash
modal secret create huggingface-secret HF_TOKEN=your_token
```

### Błąd: "Volume not found"

Stwórz volumes:
```bash
modal volume create musubi-models
modal volume create musubi-data
modal volume create musubi-cache
```

### Błąd: "File not found" podczas treningu

Sprawdź czy pliki są w volume:
```bash
modal volume ls musubi-models wan/
modal volume ls musubi-data my_dataset/
```

### Timeout podczas treningu

Zwiększ timeout w `run_modal.py` w definicji `@app.function`:
```python
@app.function(
    gpu="A100-80GB",
    timeout=28800,  # 8 godzin zamiast 4
    ...
)
```

### Out of Memory

Opcje:
1. Zwiększ `blocks_to_swap` w config
2. Użyj `fp8_base` i `fp8_llm`
3. Zmień na GPU z większym VRAM (A100-80GB lub H100)
4. Zmniejsz resolution w dataset config

---

## 📚 Dodatkowe Zasoby

- Musubi-Tuner docs: `docs/wan.md`, `docs/qwen_image.md`
- Modal docs: https://modal.com/docs
- Dataset config: `docs/dataset_config.md`
- Advanced config: `docs/advanced_config.md`

---

## 🔄 Workflow Przykładowy

```bash
# 1. Setup (jednorazowo)
pip install modal
modal setup
modal volume create musubi-models
modal volume create musubi-data
modal secret create huggingface-secret HF_TOKEN=xxx

# 2. Upload modeli (jednorazowo)
modal volume put musubi-models wan/ /path/to/wan_models/

# 3. Upload datasetu
modal volume put musubi-data my_dataset/ /path/to/dataset/

# 4. Stwórz config.json (lokalnie)
nano config.json

# 5. Uruchom trening
modal run run_modal.py --model wan --config config.json

# 6. Czekaj... (możesz wyłączyć komputer, działa w chmurze)

# 7. Download wyników
modal volume get musubi-models outputs/my_wan_lora ./
```

Gotowe! 🎉
