# Musubi-Tuner - Analiza Integracji z Modal.com

## Spis Treści
1. [Wprowadzenie](#wprowadzenie)
2. [Analiza Musubi-Tuner](#analiza-musubi-tuner)
3. [Analiza AI-Toolkit i Modal.com](#analiza-ai-toolkit-i-modalcom)
4. [Architektura Rozwiązania](#architektura-rozwiązania)
5. [Plan Implementacji](#plan-implementacji)
6. [Konfiguracja i Setup](#konfiguracja-i-setup)

---

## Wprowadzenie

Ten dokument zawiera kompleksową analizę repozytorium **musubi-tuner** i plan integracji z platformą **Modal.com** do trenowania modeli LoRA dla architektury **Wan 2.2** i **Qwen-Image**.

### Cel
Stworzenie pipeline'u treningowego, który umożliwi:
- Trenowanie modeli LoRA na Modal.com z dostępem do GPU w chmurze
- Wsparcie dla modeli Wan 2.1/2.2 (video generation)
- Wsparcie dla modeli Qwen-Image (text-to-image i image editing)
- Automatyzację całego procesu: cache → training → inference

---

## Analiza Musubi-Tuner

### Struktura Projektu

```
musubi-tuner/
├── src/musubi_tuner/
│   ├── wan_train_network.py        # Trening LoRA dla Wan
│   ├── wan_cache_latents.py        # Pre-caching latentów
│   ├── wan_cache_text_encoder_outputs.py
│   ├── wan_generate_video.py       # Generowanie wideo
│   ├── qwen_image_train_network.py # Trening LoRA dla Qwen-Image
│   ├── qwen_image_cache_latents.py
│   ├── qwen_image_cache_text_encoder_outputs.py
│   ├── qwen_image_generate_image.py
│   ├── wan/                        # Moduły Wan
│   │   ├── modules/
│   │   ├── configs/
│   │   └── utils/
│   ├── qwen_image/                 # Moduły Qwen-Image
│   ├── dataset/                    # Zarządzanie dataset'ami
│   ├── networks/                   # Implementacje LoRA
│   │   ├── lora_wan.py
│   │   └── lora_qwen_image.py
│   └── utils/
├── docs/
│   ├── wan.md
│   ├── qwen_image.md
│   ├── dataset_config.md
│   └── advanced_config.md
└── pyproject.toml
```

### Wsparcie dla Modeli

#### Wan 2.1/2.2
- **Status**: ✅ Pełne wsparcie
- **Modele**: T2V (text-to-video), I2V (image-to-video)
- **Rozmiary**: 1.3B, 14B
- **Wersje**: Wan 2.1 i Wan 2.2 (dual DiT: high noise + low noise)
- **Fun-Control**: Wsparcie dla warunkowania z kontrolnymi obrazami
- **Formaty wag**: fp16, bf16, fp8_e4m3fn (z `--fp8_base`)

#### Qwen-Image
- **Status**: ✅ Pełne wsparcie
- **Warianty**:
  - Qwen-Image: Standard text-to-image
  - Qwen-Image-Edit: Image editing z pojedynczym control image
  - Qwen-Image-Edit-2509: Image editing z wieloma control images (do 3+)
- **Text Encoder**: Qwen2.5-VL-7B
- **Formaty wag**: bf16

### Workflow Treningowy

1. **Pre-caching** (opcjonalny, ale zalecany):
   ```bash
   # Wan
   python src/musubi_tuner/wan_cache_latents.py --dataset_config config.toml --vae path/to/vae
   python src/musubi_tuner/wan_cache_text_encoder_outputs.py --dataset_config config.toml --t5 path/to/t5

   # Qwen-Image
   python src/musubi_tuner/qwen_image_cache_latents.py --dataset_config config.toml --vae path/to/vae
   python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py --dataset_config config.toml --text_encoder path/to/qwen2.5-vl
   ```

2. **Training**:
   ```bash
   # Wan
   accelerate launch --mixed_precision bf16 src/musubi_tuner/wan_train_network.py \
       --task wan_t2v_14B \
       --dit path/to/dit \
       --vae path/to/vae \
       --t5 path/to/t5 \
       --dataset_config config.toml \
       --network_module networks.lora_wan \
       --network_dim 32 --network_alpha 16 \
       --learning_rate 1e-4 \
       --max_train_epochs 10

   # Qwen-Image
   accelerate launch --mixed_precision bf16 src/musubi_tuner/qwen_image_train_network.py \
       --dit path/to/dit \
       --vae path/to/vae \
       --text_encoder path/to/qwen2.5-vl \
       --dataset_config config.toml \
       --network_module networks.lora_qwen_image \
       --network_dim 16 --network_alpha 8
   ```

3. **Inference**:
   ```bash
   python src/musubi_tuner/wan_generate_video.py --network path/to/lora.safetensors
   python src/musubi_tuner/qwen_image_generate_image.py --network path/to/lora.safetensors
   ```

### Wymagania Sprzętowe

- **VRAM**:
  - Minimum: 12GB (z optymalizacjami: `--blocks_to_swap`, `--fp8_llm`)
  - Zalecane: 24GB+ dla treningu video
- **RAM**: 32GB+ (64GB zalecane)
- **GPU**: NVIDIA z CUDA support (Ampere lub nowsza dla fp8)

### Opcje Optymalizacji Pamięci

- `--blocks_to_swap N`: Offload N bloków DiT do CPU
- `--fp8_base`: Użyj fp8 dla base modelu
- `--fp8_llm`: Użyj fp8 dla text encodera (T5/Qwen)
- `--gradient_checkpointing`: Zaoszczędź pamięć podczas backprop
- `--vae_cache_cpu`: Cache VAE na CPU zamiast GPU

---

## Analiza AI-Toolkit i Modal.com

### Struktura AI-Toolkit z Modal

Ostris AI-Toolkit używa następującej architektury:

#### run_modal.py (główny plik)
```python
import modal
import os

# Definicja Volume do przechowywania modeli
volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)

# Obraz Docker z zależnościami
image = modal.Image.debian_slim(python_version="3.10") \
    .pip_install("torch", "transformers", "diffusers", ...)

# Aplikacja Modal
app = modal.App("ai-toolkit-training")

# Montowanie kodu lokalnego
code_mount = modal.Mount.from_local_dir(
    "/local/path/ai-toolkit",
    remote_path="/root/ai-toolkit"
)

@app.function(
    gpu="A100",  # lub "H100", "A10G", etc.
    timeout=7200,  # 2 godziny
    volumes={"/outputs": volume},
    mounts=[code_mount],
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def train(config_path: str):
    """Funkcja treningowa wykonywana w Modal"""
    import subprocess

    # Uruchom trening
    result = subprocess.run([
        "python", "/root/ai-toolkit/run.py",
        "--config", config_path
    ])

    return result.returncode
```

#### Kluczowe Elementy Modal

1. **modal.App**: Kontener aplikacji
2. **modal.Image**: Definicja obrazu Docker z zależnościami
3. **modal.Volume**: Persistent storage dla modeli i danych
4. **modal.Mount**: Montowanie lokalnego kodu do kontenera
5. **@app.function**: Dekorator definiujący funkcje wykonywane w chmurze
6. **GPU Selection**: Wybór GPU: "A100", "H100", "A10G", "T4", etc.
7. **Secrets**: Bezpieczne przechowywanie API keys (HuggingFace)

### Wzorce Modal.com

#### Pattern 1: Simple Function
```python
@app.function(gpu="A100")
def process():
    import torch
    # kod wykonywany w chmurze
```

#### Pattern 2: Class-based (długie sesje)
```python
@app.cls(gpu="A100", volumes={"/cache": cache_vol})
class Trainer:
    @modal.enter()
    def setup(self):
        # setup wykonywany raz przy starcie
        pass

    @modal.method()
    def train(self):
        # metoda treningowa
        pass
```

#### Pattern 3: Batch Jobs
```python
@app.function(gpu="A100:2")  # 2x A100
def train_batch(configs: list):
    for config in configs:
        train_single(config)
```

---

## Architektura Rozwiązania

### Architektura High-Level

```
┌─────────────────────────────────────────────────────────────┐
│                         LOKALNE                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Przygotowanie Datasetu                           │   │
│  │     - Organizacja obrazów/video                      │   │
│  │     - Przygotowanie captionów                        │   │
│  │     - Konfiguracja TOML                              │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  2. Upload do Modal Volume                           │   │
│  │     modal volume put musubi-data dataset/            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      MODAL.COM                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  3. Pre-caching (GPU)                                │   │
│  │     - Cache latents (VAE encoding)                   │   │
│  │     - Cache text encoder outputs                     │   │
│  │     - Zapisz cache w Volume                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  4. Training (GPU)                                   │   │
│  │     - Załaduj cache z Volume                         │   │
│  │     - Trening LoRA                                   │   │
│  │     - Zapisz checkpointy                             │   │
│  │     - Zapisz finalny model                           │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  5. Inference/Validation (GPU)                       │   │
│  │     - Załaduj wytrenowany LoRA                       │   │
│  │     - Generuj sample images/videos                   │   │
│  │     - Zapisz wyniki                                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                         LOKALNE                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  6. Download Wyników                                 │   │
│  │     modal volume get musubi-models output/           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Struktura Plików Modal

```
musubi-tuner/
├── modal_integration/
│   ├── __init__.py
│   ├── run_modal.py              # Główny entry point
│   ├── modal_config.py           # Konfiguracja Modal
│   ├── modal_cache.py            # Pre-caching w Modal
│   ├── modal_train.py            # Training w Modal
│   ├── modal_inference.py        # Inference w Modal
│   └── utils.py                  # Pomocnicze funkcje
├── modal_configs/
│   ├── wan_t2v_14b.yaml         # Config dla Wan T2V
│   ├── wan_i2v_14b.yaml         # Config dla Wan I2V
│   ├── wan22_14b.yaml           # Config dla Wan 2.2
│   ├── qwen_image.yaml          # Config dla Qwen-Image
│   └── qwen_image_edit.yaml     # Config dla Qwen-Image-Edit
└── README_MODAL.md              # Dokumentacja
```

---

## Plan Implementacji

### Faza 1: Podstawowa Infrastruktura

#### 1.1 Setup Obrazu Docker
Stwórz obraz z wszystkimi zależnościami musubi-tuner:

```python
# modal_integration/modal_config.py

import modal

def create_musubi_image():
    """Tworzy obraz Docker z zależnościami musubi-tuner"""
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("git", "wget", "ffmpeg")  # ffmpeg dla video processing
        .pip_install(
            "torch>=2.5.1",
            "torchvision>=0.20.1",
            "accelerate==1.6.0",
            "diffusers==0.32.1",
            "transformers==4.54.1",
            "safetensors==0.4.5",
            "huggingface-hub==0.34.3",
            "bitsandbytes",
            "einops==0.7.0",
            "opencv-python==4.10.0.84",
            "pillow>=11.3.0",
            "toml==0.10.2",
            "tqdm==4.67.1",
            "voluptuous==0.15.2",
            "av==14.0.1",  # dla video
            "ftfy==6.3.1",  # dla Wan
            "easydict==1.13",  # dla Wan
            "sentencepiece==0.2.1",  # dla FLUX Kontext
            index_url="https://download.pytorch.org/whl/cu124"
        )
        # Opcjonalnie: zainstaluj xformers dla szybszej attention
        .run_commands(
            "pip install xformers --index-url https://download.pytorch.org/whl/cu124"
        )
    )
    return image

# Volumes
DATA_VOLUME = modal.Volume.from_name("musubi-data", create_if_missing=True)
MODELS_VOLUME = modal.Volume.from_name("musubi-models", create_if_missing=True)
CACHE_VOLUME = modal.Volume.from_name("musubi-cache", create_if_missing=True)

# Paths w Modal
DATA_PATH = "/data"
MODELS_PATH = "/models"
CACHE_PATH = "/cache"
CODE_PATH = "/root/musubi-tuner"
```

#### 1.2 Główna Aplikacja Modal

```python
# modal_integration/run_modal.py

import modal
from modal_config import create_musubi_image, DATA_VOLUME, MODELS_VOLUME, CACHE_VOLUME
from modal_config import DATA_PATH, MODELS_PATH, CACHE_PATH, CODE_PATH

app = modal.App("musubi-tuner")
image = create_musubi_image()

# Mount lokalnego kodu (dla development)
code_mount = modal.Mount.from_local_dir(
    ".",  # lokalny katalog musubi-tuner
    remote_path=CODE_PATH,
    condition=lambda path: not any([
        ".git" in path,
        "__pycache__" in path,
        ".venv" in path,
        "outputs" in path,
    ])
)
```

### Faza 2: Pre-caching

#### 2.1 Cache Latents

```python
# modal_integration/modal_cache.py

@app.function(
    image=image,
    gpu="A10G",  # Mniejsze GPU wystarczy dla cache
    timeout=3600,
    volumes={
        DATA_PATH: DATA_VOLUME,
        CACHE_PATH: CACHE_VOLUME,
        MODELS_PATH: MODELS_VOLUME,
    },
    mounts=[code_mount],
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def cache_latents_wan(
    dataset_config_path: str,
    vae_path: str,
    clip_path: str = None,
    task: str = "wan_t2v_14B",
    i2v: bool = False,
):
    """Cache latents dla Wan w Modal"""
    import subprocess
    import os

    # Pełne ścieżki w kontenerze
    config_full = f"{DATA_PATH}/{dataset_config_path}"
    vae_full = f"{MODELS_PATH}/{vae_path}"

    cmd = [
        "python", f"{CODE_PATH}/src/musubi_tuner/wan_cache_latents.py",
        "--dataset_config", config_full,
        "--vae", vae_full,
        "--task", task,
    ]

    if i2v:
        cmd.append("--i2v")

    if clip_path:
        clip_full = f"{MODELS_PATH}/{clip_path}"
        cmd.extend(["--clip", clip_full])

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Commit Volume aby zapisać zmiany
    CACHE_VOLUME.commit()

    return {
        "status": "success" if result.returncode == 0 else "error",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={
        DATA_PATH: DATA_VOLUME,
        CACHE_PATH: CACHE_VOLUME,
        MODELS_PATH: MODELS_VOLUME,
    },
    mounts=[code_mount],
)
def cache_latents_qwen(
    dataset_config_path: str,
    vae_path: str,
    edit: bool = False,
    edit_plus: bool = False,
):
    """Cache latents dla Qwen-Image w Modal"""
    import subprocess

    config_full = f"{DATA_PATH}/{dataset_config_path}"
    vae_full = f"{MODELS_PATH}/{vae_path}"

    cmd = [
        "python", f"{CODE_PATH}/src/musubi_tuner/qwen_image_cache_latents.py",
        "--dataset_config", config_full,
        "--vae", vae_full,
    ]

    if edit:
        cmd.append("--edit")
    elif edit_plus:
        cmd.append("--edit_plus")

    result = subprocess.run(cmd, capture_output=True, text=True)
    CACHE_VOLUME.commit()

    return {
        "status": "success" if result.returncode == 0 else "error",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={
        DATA_PATH: DATA_VOLUME,
        CACHE_PATH: CACHE_VOLUME,
        MODELS_PATH: MODELS_VOLUME,
    },
    mounts=[code_mount],
)
def cache_text_encoder_wan(
    dataset_config_path: str,
    t5_path: str,
    batch_size: int = 16,
    fp8_t5: bool = False,
):
    """Cache text encoder outputs dla Wan"""
    import subprocess

    config_full = f"{DATA_PATH}/{dataset_config_path}"
    t5_full = f"{MODELS_PATH}/{t5_path}"

    cmd = [
        "python", f"{CODE_PATH}/src/musubi_tuner/wan_cache_text_encoder_outputs.py",
        "--dataset_config", config_full,
        "--t5", t5_full,
        "--batch_size", str(batch_size),
    ]

    if fp8_t5:
        cmd.append("--fp8_t5")

    result = subprocess.run(cmd, capture_output=True, text=True)
    CACHE_VOLUME.commit()

    return {
        "status": "success" if result.returncode == 0 else "error",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={
        DATA_PATH: DATA_VOLUME,
        CACHE_PATH: CACHE_VOLUME,
        MODELS_PATH: MODELS_VOLUME,
    },
    mounts=[code_mount],
)
def cache_text_encoder_qwen(
    dataset_config_path: str,
    text_encoder_path: str,
    batch_size: int = 1,
    fp8_vl: bool = False,
    edit: bool = False,
    edit_plus: bool = False,
):
    """Cache text encoder outputs dla Qwen-Image"""
    import subprocess

    config_full = f"{DATA_PATH}/{dataset_config_path}"
    te_full = f"{MODELS_PATH}/{text_encoder_path}"

    cmd = [
        "python", f"{CODE_PATH}/src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py",
        "--dataset_config", config_full,
        "--text_encoder", te_full,
        "--batch_size", str(batch_size),
    ]

    if fp8_vl:
        cmd.append("--fp8_vl")
    if edit:
        cmd.append("--edit")
    elif edit_plus:
        cmd.append("--edit_plus")

    result = subprocess.run(cmd, capture_output=True, text=True)
    CACHE_VOLUME.commit()

    return {
        "status": "success" if result.returncode == 0 else "error",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
```

### Faza 3: Training

#### 3.1 Training Function

```python
# modal_integration/modal_train.py (kontynuacja w run_modal.py)

@app.function(
    image=image,
    gpu="A100-80GB",  # Lub "H100" dla najszybszego treningu
    timeout=14400,  # 4 godziny
    volumes={
        DATA_PATH: DATA_VOLUME,
        CACHE_PATH: CACHE_VOLUME,
        MODELS_PATH: MODELS_VOLUME,
    },
    mounts=[code_mount],
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_wan(
    config_dict: dict,
):
    """
    Trening LoRA dla Wan w Modal

    config_dict zawiera wszystkie parametry treningu:
    - task: "wan_t2v_14B", "wan_i2v_14B", etc.
    - dit_path, vae_path, t5_path
    - dataset_config_path
    - network_dim, network_alpha
    - learning_rate, max_train_epochs
    - etc.
    """
    import subprocess
    import os

    # Przygotuj ścieżki
    dit_full = f"{MODELS_PATH}/{config_dict['dit_path']}"
    vae_full = f"{MODELS_PATH}/{config_dict['vae_path']}"
    t5_full = f"{MODELS_PATH}/{config_dict['t5_path']}"
    config_full = f"{DATA_PATH}/{config_dict['dataset_config_path']}"
    output_dir = f"{MODELS_PATH}/outputs/{config_dict['output_name']}"

    # Konfiguracja accelerate (single GPU w Modal)
    os.environ["ACCELERATE_CONFIG_FILE"] = "none"

    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", config_dict.get("mixed_precision", "bf16"),
        f"{CODE_PATH}/src/musubi_tuner/wan_train_network.py",
        "--task", config_dict["task"],
        "--dit", dit_full,
        "--vae", vae_full,
        "--t5", t5_full,
        "--dataset_config", config_full,
        "--network_module", "networks.lora_wan",
        "--network_dim", str(config_dict.get("network_dim", 32)),
        "--network_alpha", str(config_dict.get("network_alpha", 16)),
        "--learning_rate", str(config_dict.get("learning_rate", "1e-4")),
        "--optimizer_type", config_dict.get("optimizer_type", "adamw8bit"),
        "--max_train_epochs", str(config_dict.get("max_train_epochs", 10)),
        "--save_every_n_epochs", str(config_dict.get("save_every_n_epochs", 1)),
        "--output_dir", output_dir,
        "--output_name", config_dict["output_name"],
        "--mixed_precision", config_dict.get("mixed_precision", "bf16"),
        "--sdpa",  # Użyj PyTorch SDPA
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
    ]

    # Opcjonalne parametry optymalizacji pamięci
    if config_dict.get("blocks_to_swap"):
        cmd.extend(["--blocks_to_swap", str(config_dict["blocks_to_swap"])])
    if config_dict.get("fp8_base"):
        cmd.append("--fp8_base")
    if config_dict.get("fp8_llm"):
        cmd.append("--fp8_llm")

    # Wan 2.2 specific
    if config_dict.get("dit_high_noise_path"):
        dit_high = f"{MODELS_PATH}/{config_dict['dit_high_noise_path']}"
        cmd.extend(["--dit_high_noise", dit_high])
    if config_dict.get("timestep_boundary"):
        cmd.extend(["--timestep_boundary", str(config_dict["timestep_boundary"])])

    # Seed dla reproducibility
    if config_dict.get("seed"):
        cmd.extend(["--seed", str(config_dict["seed"])])

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Commit aby zapisać wytrenowany model
    MODELS_VOLUME.commit()

    return {
        "status": "success" if result.returncode == 0 else "error",
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output_path": output_dir,
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=14400,
    volumes={
        DATA_PATH: DATA_VOLUME,
        CACHE_PATH: CACHE_VOLUME,
        MODELS_PATH: MODELS_VOLUME,
    },
    mounts=[code_mount],
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_qwen(
    config_dict: dict,
):
    """Trening LoRA dla Qwen-Image w Modal"""
    import subprocess
    import os

    dit_full = f"{MODELS_PATH}/{config_dict['dit_path']}"
    vae_full = f"{MODELS_PATH}/{config_dict['vae_path']}"
    te_full = f"{MODELS_PATH}/{config_dict['text_encoder_path']}"
    config_full = f"{DATA_PATH}/{config_dict['dataset_config_path']}"
    output_dir = f"{MODELS_PATH}/outputs/{config_dict['output_name']}"

    os.environ["ACCELERATE_CONFIG_FILE"] = "none"

    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",
        f"{CODE_PATH}/src/musubi_tuner/qwen_image_train_network.py",
        "--dit", dit_full,
        "--vae", vae_full,
        "--text_encoder", te_full,
        "--dataset_config", config_full,
        "--network_module", "networks.lora_qwen_image",
        "--network_dim", str(config_dict.get("network_dim", 16)),
        "--network_alpha", str(config_dict.get("network_alpha", 8)),
        "--learning_rate", str(config_dict.get("learning_rate", "5e-5")),
        "--optimizer_type", config_dict.get("optimizer_type", "adamw8bit"),
        "--max_train_epochs", str(config_dict.get("max_train_epochs", 16)),
        "--save_every_n_epochs", str(config_dict.get("save_every_n_epochs", 1)),
        "--output_dir", output_dir,
        "--output_name", config_dict["output_name"],
        "--sdpa",
        "--gradient_checkpointing",
        "--timestep_sampling", config_dict.get("timestep_sampling", "shift"),
        "--weighting_scheme", config_dict.get("weighting_scheme", "none"),
        "--discrete_flow_shift", str(config_dict.get("discrete_flow_shift", 2.2)),
    ]

    # Edit mode
    if config_dict.get("edit"):
        cmd.append("--edit")
    elif config_dict.get("edit_plus"):
        cmd.append("--edit_plus")

    if config_dict.get("fp8_vl"):
        cmd.append("--fp8_vl")

    if config_dict.get("seed"):
        cmd.extend(["--seed", str(config_dict["seed"])])

    result = subprocess.run(cmd, capture_output=True, text=True)
    MODELS_VOLUME.commit()

    return {
        "status": "success" if result.returncode == 0 else "error",
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output_path": output_dir,
    }
```

### Faza 4: CLI i Orchestration

#### 4.1 Główny CLI Entry Point

```python
# modal_integration/run_modal.py (kontynuacja)

@app.local_entrypoint()
def main(
    config_file: str,
    stage: str = "all",  # all, cache, train, inference
):
    """
    Główny entry point do uruchamiania pipeline

    Usage:
        modal run modal_integration/run_modal.py --config-file modal_configs/wan_t2v_14b.yaml
        modal run modal_integration/run_modal.py --config-file modal_configs/wan_t2v_14b.yaml --stage cache
        modal run modal_integration/run_modal.py --config-file modal_configs/wan_t2v_14b.yaml --stage train
    """
    import yaml

    # Załaduj konfigurację
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    model_type = config.get("model_type")  # "wan" lub "qwen"

    print(f"🚀 Starting Musubi-Tuner pipeline on Modal.com")
    print(f"   Model: {model_type}")
    print(f"   Stage: {stage}")
    print(f"   Config: {config_file}")

    # Stage 1: Cache
    if stage in ["all", "cache"]:
        print("\n📦 Stage 1: Pre-caching...")

        if model_type == "wan":
            print("  → Caching latents...")
            result = cache_latents_wan.remote(
                dataset_config_path=config["dataset_config"],
                vae_path=config["vae_path"],
                clip_path=config.get("clip_path"),
                task=config.get("task", "wan_t2v_14B"),
                i2v=config.get("i2v", False),
            )
            print(f"  ✓ Latents cached: {result['status']}")

            print("  → Caching text encoder outputs...")
            result = cache_text_encoder_wan.remote(
                dataset_config_path=config["dataset_config"],
                t5_path=config["t5_path"],
                batch_size=config.get("cache_batch_size", 16),
                fp8_t5=config.get("fp8_t5", False),
            )
            print(f"  ✓ Text encoder outputs cached: {result['status']}")

        elif model_type == "qwen":
            print("  → Caching latents...")
            result = cache_latents_qwen.remote(
                dataset_config_path=config["dataset_config"],
                vae_path=config["vae_path"],
                edit=config.get("edit", False),
                edit_plus=config.get("edit_plus", False),
            )
            print(f"  ✓ Latents cached: {result['status']}")

            print("  → Caching text encoder outputs...")
            result = cache_text_encoder_qwen.remote(
                dataset_config_path=config["dataset_config"],
                text_encoder_path=config["text_encoder_path"],
                batch_size=config.get("cache_batch_size", 1),
                fp8_vl=config.get("fp8_vl", False),
                edit=config.get("edit", False),
                edit_plus=config.get("edit_plus", False),
            )
            print(f"  ✓ Text encoder outputs cached: {result['status']}")

    # Stage 2: Train
    if stage in ["all", "train"]:
        print("\n🎓 Stage 2: Training...")

        if model_type == "wan":
            result = train_wan.remote(config_dict=config)
        elif model_type == "qwen":
            result = train_qwen.remote(config_dict=config)

        print(f"  ✓ Training completed: {result['status']}")
        print(f"  📁 Model saved to: {result['output_path']}")

        if result['status'] == 'error':
            print(f"\n❌ Error during training:")
            print(result['stderr'])

    print("\n✨ Pipeline completed!")
    print(f"\n📥 To download results:")
    print(f"   modal volume get musubi-models outputs/{config['output_name']} ./")
```

### Faza 5: Konfiguracja YAML

#### 5.1 Przykładowe Konfiguracje

```yaml
# modal_configs/wan_t2v_14b.yaml

model_type: "wan"
task: "wan_t2v_14B"

# Ścieżki do modeli (w Modal Volume)
dit_path: "wan/wan_t2v_14B_bf16.safetensors"
vae_path: "wan/wan_2.1_vae.safetensors"
t5_path: "wan/models_t5_umt5-xxl-enc-bf16.pth"
clip_path: "wan/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"

# Dataset
dataset_config: "datasets/my_video_dataset.toml"
i2v: false

# Cache settings
cache_batch_size: 16
fp8_t5: false

# Training settings
output_name: "wan_t2v_my_lora"
network_dim: 32
network_alpha: 16
learning_rate: 1e-4
optimizer_type: "adamw8bit"
max_train_epochs: 10
save_every_n_epochs: 1
mixed_precision: "bf16"

# Memory optimization
blocks_to_swap: 8
fp8_base: false
fp8_llm: false

# Reproducibility
seed: 42
```

```yaml
# modal_configs/wan22_14b.yaml

model_type: "wan"
task: "wan_t2v_14B"

# Wan 2.2 używa dwóch modeli DiT
dit_path: "wan22/wan_2.2_t2v_14B_low_noise_bf16.safetensors"
dit_high_noise_path: "wan22/wan_2.2_t2v_14B_high_noise_bf16.safetensors"
timestep_boundary: 0.3  # lub inna wartość (0-1)

vae_path: "wan/wan_2.1_vae.safetensors"  # Ten sam VAE co Wan 2.1
t5_path: "wan/models_t5_umt5-xxl-enc-bf16.pth"
# Wan 2.2 nie używa CLIP

dataset_config: "datasets/my_video_dataset.toml"

output_name: "wan22_t2v_my_lora"
network_dim: 32
network_alpha: 16
learning_rate: 1e-4
max_train_epochs: 10

blocks_to_swap: 8
seed: 42
```

```yaml
# modal_configs/qwen_image.yaml

model_type: "qwen"

# Ścieżki
dit_path: "qwen/qwen_image_bf16.safetensors"
vae_path: "qwen/vae_diffusion_pytorch_model.safetensors"
text_encoder_path: "qwen/qwen_2.5_vl_7b.safetensors"

# Dataset
dataset_config: "datasets/my_image_dataset.toml"

# Cache settings
cache_batch_size: 1
fp8_vl: false

# Training
output_name: "qwen_image_my_lora"
network_dim: 16
network_alpha: 8
learning_rate: 5e-5
optimizer_type: "adamw8bit"
max_train_epochs: 16
save_every_n_epochs: 1

# Flow matching parameters
timestep_sampling: "shift"
weighting_scheme: "none"
discrete_flow_shift: 2.2

seed: 42
```

```yaml
# modal_configs/qwen_image_edit.yaml

model_type: "qwen"
edit: true  # lub edit_plus: true dla Edit-2509

dit_path: "qwen/qwen_image_edit_bf16.safetensors"
# lub dla Edit-2509: "qwen/qwen_image_edit_2509_bf16.safetensors"

vae_path: "qwen/vae_diffusion_pytorch_model.safetensors"
text_encoder_path: "qwen/qwen_2.5_vl_7b.safetensors"

dataset_config: "datasets/my_edit_dataset.toml"

cache_batch_size: 1
fp8_vl: false

output_name: "qwen_edit_my_lora"
network_dim: 16
network_alpha: 8
learning_rate: 5e-5
max_train_epochs: 16

timestep_sampling: "shift"
weighting_scheme: "none"
discrete_flow_shift: 2.2

seed: 42
```

---

## Konfiguracja i Setup

### Instalacja Modal

```bash
# Zainstaluj Modal CLI
pip install modal

# Zaloguj się do Modal
modal setup

# Potwierdź login
modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_SECRET
```

### Przygotowanie Środowiska

#### 1. Upload Modeli do Modal Volume

```bash
# Stwórz volume dla modeli
modal volume create musubi-models

# Upload modeli Wan
modal volume put musubi-models wan/ ./local_models/wan/

# Upload modeli Qwen
modal volume put musubi-models qwen/ ./local_models/qwen/
```

#### 2. Upload Datasetu

```bash
# Stwórz volume dla danych
modal volume create musubi-data

# Upload dataset
modal volume put musubi-data datasets/ ./my_dataset/

# Upload TOML config
modal volume put musubi-data datasets/config.toml ./my_dataset/config.toml
```

#### 3. Konfiguracja Secrets (HuggingFace)

```bash
# Stwórz secret dla HuggingFace
modal secret create huggingface-secret HF_TOKEN=your_hf_token_here
```

### Uruchomienie Pipeline

#### Pełny Pipeline (cache + train)

```bash
modal run modal_integration/run_modal.py \
    --config-file modal_configs/wan_t2v_14b.yaml
```

#### Tylko Cache

```bash
modal run modal_integration/run_modal.py \
    --config-file modal_configs/wan_t2v_14b.yaml \
    --stage cache
```

#### Tylko Training (po wykonaniu cache)

```bash
modal run modal_integration/run_modal.py \
    --config-file modal_configs/wan_t2v_14b.yaml \
    --stage train
```

### Download Wyników

```bash
# Pokaż pliki w volume
modal volume ls musubi-models outputs/

# Download konkretnego modelu
modal volume get musubi-models outputs/wan_t2v_my_lora ./local_outputs/

# Download wszystkich outputów
modal volume get musubi-models outputs/ ./local_outputs/
```

### Monitoring

Modal zapewnia automatyczny monitoring:

```bash
# Zobacz logi z ostatniego uruchomienia
modal app logs musubi-tuner

# Real-time monitoring przez web UI
# Przejdź do https://modal.com/apps
```

---

## Podsumowanie

### Zalety Rozwiązania

✅ **Dostęp do GPU**: A100, H100 on-demand
✅ **Skalowalność**: Automatyczne zarządzanie zasobami
✅ **Koszt**: Płacisz tylko za rzeczywiste użycie GPU
✅ **Persistent Storage**: Modal Volumes dla modeli i cache
✅ **Łatwe Deploy**: Jeden plik konfiguracji YAML
✅ **Monitoring**: Built-in logi i metryki
✅ **Reprodukowalność**: Spójne środowisko Docker

### Następne Kroki

1. **Implementacja kodu**: Stwórz pliki opisane w sekcji "Plan Implementacji"
2. **Testy**: Przetestuj z małym dataset'em
3. **Optymalizacja**: Dobierz optymalne GPU i parametry pamięci
4. **Dokumentacja**: Stwórz szczegółowe instrukcje dla użytkowników
5. **CI/CD**: Opcjonalnie dodaj automatyzację

### Estymowany Koszt (Modal.com)

- **A10G** (~$0.60/hr): Cache, mniejsze treningi
- **A100-40GB** (~$2.50/hr): Standardowy trening
- **A100-80GB** (~$3.00/hr): Większe modele, batch size
- **H100** (~$5.00/hr): Najszybszy trening

Przykład: 10 epochs x 1h = $25-30 na A100-80GB

---

## Kontakt i Wsparcie

Jeśli masz pytania lub problemy z implementacją:
- Sprawdź dokumentację Modal: https://modal.com/docs
- Sprawdź dokumentację Musubi-Tuner: docs/
- GitHub Issues: https://github.com/kohya-ss/musubi-tuner/issues

**Autor dokumentu**: Claude (Anthropic)
**Data**: 2025-10-31
**Wersja**: 1.0
