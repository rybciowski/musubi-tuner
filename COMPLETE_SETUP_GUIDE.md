# 🚀 KOMPLETNA INSTRUKCJA - OD ZERA DO WYTRENOWANEGO LORA

## 📍 Struktura - Gdzie Co Będzie

```
/home/user/                          (Twój katalog domowy)
│
├── musubi-tuner/                    (REPO - tu będzie kod)
│   ├── run_modal_v2.py              (skrypt do uruchamiania)
│   ├── src/musubi_tuner/            (kod musubi-tuner)
│   ├── docs/                        (dokumentacja)
│   └── my_config.json               (TU tworzysz config)
│
└── my_datasets/                     (Twoje datasety - POZA repo)
    ├── wan_dataset_1/
    │   ├── videos/
    │   │   ├── video001.mp4
    │   │   ├── video001.txt
    │   │   └── ...
    │   └── config.toml
    │
    └── qwen_dataset_1/
        ├── images/
        │   ├── img001.jpg
        │   ├── img001.txt
        │   └── ...
        └── config.toml
```

---

## KROK 0: Wymagania

```bash
# Sprawdź Python
python --version  # Potrzebujesz 3.10+

# Jeśli nie masz lub masz starszy:
# - Linux: sudo apt install python3.10
# - Mac: brew install python@3.10
# - Windows: pobierz z python.org
```

---

## KROK 1: Sklonuj Repo - 2 minuty

```bash
# Przejdź do katalogu domowego
cd ~

# Sklonuj musubi-tuner
git clone https://github.com/kohya-ss/musubi-tuner.git

# Wejdź do folderu
cd musubi-tuner

# Sprawdź czy masz run_modal_v2.py
ls -la run_modal_v2.py

# Jeśli NIE MA tego pliku:
# Pobierz z mojego brancha:
git fetch origin claude/analyze-lora-training-repo-011CUecwmPeMRMZWQqZH52qX
git checkout claude/analyze-lora-training-repo-011CUecwmPeMRMZWQqZH52qX

# TERAZ powinieneś mieć:
ls -la run_modal_v2.py  # ✓ powinien istnieć
```

**GDZIE JESTEŚ TERAZ:**
```
/home/user/musubi-tuner/  <- TU jesteś
```

---

## KROK 2: Zainstaluj Modal - 3 minuty

```bash
# JESTEŚ W: /home/user/musubi-tuner/

# Zainstaluj Modal CLI
pip install modal

# Zaloguj się (otworzy przeglądarkę)
modal setup
```

**W przeglądarce:**
1. Zaloguj się przez GitHub/Google/Email
2. Zaakceptuj
3. Wróć do terminala

**Sprawdź czy działa:**
```bash
modal --version
# Powinno pokazać: modal, version X.X.X
```

---

## KROK 3: Stwórz Volumes - 2 minuty

```bash
# JESTEŚ W: /home/user/musubi-tuner/

# Stwórz 3 volumes (dyski w chmurze)
modal volume create musubi-models
modal volume create musubi-data
modal volume create musubi-cache

# Sprawdź czy są:
modal volume list
# Powinno pokazać:
# musubi-models
# musubi-data
# musubi-cache
```

**Co to robi:**
- Tworzy 3 "dyski" w chmurze Modal
- Tam będą: modele, dane, cache

---

## KROK 4: Przygotuj Dataset - 30 minut

### Krok 4.1: Stwórz Folder POZA Repo

```bash
# WYJDŹ z repo
cd ~

# Stwórz folder na datasety
mkdir -p my_datasets

# Stwórz konkretny dataset (przykład Wan)
mkdir -p my_datasets/wan_dataset_1/videos

# Struktura:
# ~/my_datasets/
#   └── wan_dataset_1/
#       └── videos/
```

### Krok 4.2: Wrzuć Swoje Pliki

**Dla Wan (video):**
```bash
# Skopiuj swoje video do:
# ~/my_datasets/wan_dataset_1/videos/

# Struktura finalna:
# videos/
#   ├── video001.mp4
#   ├── video001.txt    (caption: "woman playing guitar")
#   ├── video002.mp4
#   ├── video002.txt    (caption: "man riding bike")
#   └── ...

# Każdy video musi mieć .txt z captionem!
```

**Dla Qwen (obrazy):**
```bash
# Stwórz:
mkdir -p my_datasets/qwen_dataset_1/images

# Skopiuj swoje obrazy do:
# ~/my_datasets/qwen_dataset_1/images/

# Struktura finalna:
# images/
#   ├── img001.jpg
#   ├── img001.txt    (caption: "red car on street")
#   ├── img002.jpg
#   ├── img002.txt
#   └── ...
```

### Krok 4.3: Stwórz config.toml

```bash
# Dla Wan:
cd ~/my_datasets/wan_dataset_1/
nano config.toml
```

**Wklej to:**
```toml
[general]
resolution = [1280, 720]
enable_bucket = true
batch_size = 1

[[datasets]]
[[datasets.subsets]]
video_directory = "videos"
num_repeats = 10
target_frames = [81]
```

**Zapisz:** Ctrl+O, Enter, Ctrl+X

```bash
# Dla Qwen:
cd ~/my_datasets/qwen_dataset_1/
nano config.toml
```

**Wklej to:**
```toml
[general]
resolution = [1024, 1024]
enable_bucket = true
batch_size = 1

[[datasets]]
[[datasets.subsets]]
image_directory = "images"
num_repeats = 10
```

**Zapisz:** Ctrl+O, Enter, Ctrl+X

### Krok 4.4: Upload do Modal

```bash
# Wróć do home
cd ~

# Upload datasetu do Modal
modal volume put musubi-data wan_dataset_1/ ./my_datasets/wan_dataset_1/

# Lub dla Qwen:
# modal volume put musubi-data qwen_dataset_1/ ./my_datasets/qwen_dataset_1/

# CZEKAJ 10-30 minut (zależnie od rozmiaru)
```

**Sprawdź czy uploadowało:**
```bash
modal volume ls musubi-data wan_dataset_1/

# Powinieneś zobaczyć:
# wan_dataset_1/config.toml
# wan_dataset_1/videos/
# wan_dataset_1/videos/video001.mp4
# ...
```

---

## KROK 5: Stwórz Config Treningu - 2 minuty

```bash
# WRÓĆ DO REPO
cd ~/musubi-tuner/

# JESTEŚ TERAZ W: /home/user/musubi-tuner/

# Stwórz config JSON
nano my_wan_config.json
```

**Dla Wan 2.1 - WKLEJ TO:**
```json
{
  "task": "wan_t2v_14B",
  "dit_path": "wan/wan_t2v_14B_bf16.safetensors",
  "vae_path": "wan/wan_2.1_vae.safetensors",
  "t5_path": "wan/models_t5_umt5-xxl-enc-bf16.pth",
  "clip_path": "wan/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
  "dataset_config": "wan_dataset_1/config.toml",
  "output_name": "my_first_wan_lora",
  "network_dim": 32,
  "network_alpha": 16,
  "learning_rate": "1e-4",
  "max_train_epochs": 10,
  "blocks_to_swap": 8,
  "seed": 42
}
```

**ZAPISZ:** Ctrl+O, Enter, Ctrl+X

**Dla Qwen - stwórz drugi plik:**
```bash
nano my_qwen_config.json
```

**WKLEJ TO:**
```json
{
  "dit_path": "qwen/qwen_image_bf16.safetensors",
  "vae_path": "qwen/vae_diffusion_pytorch_model.safetensors",
  "text_encoder_path": "qwen/qwen_2.5_vl_7b.safetensors",
  "dataset_config": "qwen_dataset_1/config.toml",
  "output_name": "my_first_qwen_lora",
  "network_dim": 16,
  "network_alpha": 8,
  "learning_rate": "5e-5",
  "max_train_epochs": 16,
  "seed": 42
}
```

**ZAPISZ:** Ctrl+O, Enter, Ctrl+X

**Sprawdź co masz:**
```bash
ls -la *.json

# Powinieneś zobaczyć:
# my_wan_config.json
# my_qwen_config.json (jeśli zrobiłeś)
```

---

## KROK 6: URUCHOM TRENING - 3-4 godziny

```bash
# UPEWNIJ SIĘ ŻE JESTEŚ W REPO
pwd
# Powinno pokazać: /home/user/musubi-tuner

# URUCHOM
modal run run_modal_v2.py --model wan --config my_wan_config.json

# Lub dla Qwen:
# modal run run_modal_v2.py --model qwen --config my_qwen_config.json
```

**CO ZOBACZYSZ:**

```
✓ Initialized. View run at https://modal.com/...
✓ Created objects.

🚀 Musubi-Tuner Modal Pipeline (Auto-Download Enabled)
======================================================================
Model:  wan
Stage:  all
Config: my_wan_config.json
======================================================================
ℹ️  Models will be auto-downloaded from HuggingFace on first use
ℹ️  Subsequent runs will use cached models (fast!)

📦 Stage 1/3: Caching latents...
🔍 Checking if models are available...
📥 Downloading model from HuggingFace: Comfy-Org/Wan_2.1_ComfyUI_repackaged
   Files: ['split_files/diffusion_models/wan_t2v_14B_bf16.safetensors']
   Downloading: ████████████████ 27GB in 8 min
   ✓ Saved to: /models/wan/wan_t2v_14B_bf16.safetensors

[... więcej modeli się pobierze ...]

Running: Caching latents for Wan
============================================================
[INFO] Loading VAE...
[INFO] Processing video 1/50: video001.mp4
[... cache trwa 20-30 min ...]

✓ Latents cached: success

📦 Stage 2/3: Caching text encoder outputs...
[... cache T5, 10-15 min ...]

✓ Text encoder outputs cached: success

🎓 Stage 3/3: Training...
[INFO] Loading DiT model...
[INFO] Creating LoRA network...
[INFO] Training started

Epoch 1/10:
  2%|▏ | 10/500 [01:23<1:08:45, 8.42s/it, loss=0.7234]
[... trening trwa 2-3 godziny ...]

Epoch 10/10:
100%|██████████| 500/500 [1:09:12<00:00, 8.30s/it, loss=0.0156]

✓ Training completed: success

📁 Model saved to: /models/outputs/my_first_wan_lora

💾 To download results:
   modal volume get musubi-models outputs/my_first_wan_lora ./

====================================
✨ Pipeline completed!
====================================
```

**ZOSTAW TERMINAL OTWARTY** - logi drukują się w czasie rzeczywistym!

---

## KROK 7: Download Wytrenowanego LoRA - 2 minuty

```bash
# JESTEŚ W: /home/user/musubi-tuner/

# Zobacz co jest w volume
modal volume ls musubi-models outputs/

# Powinieneś zobaczyć:
# my_first_wan_lora/
#   my_first_wan_lora_000001.safetensors
#   my_first_wan_lora_000010.safetensors  <- FINAŁ

# Stwórz folder lokalny
mkdir -p ~/trained_loras

# Download
modal volume get musubi-models outputs/my_first_wan_lora ~/trained_loras/

# CZEKAJ 2-5 minut
```

**SPRAWDŹ CO MASZ:**
```bash
ls -la ~/trained_loras/my_first_wan_lora/

# Powinieneś zobaczyć:
# my_first_wan_lora_000001.safetensors (312 MB)
# my_first_wan_lora_000002.safetensors
# ...
# my_first_wan_lora_000010.safetensors (312 MB) <- UŻYJ TEGO
```

---

## ✅ GOTOWE! Masz LoRA!

**Twój LoRA jest tutaj:**
```
~/trained_loras/my_first_wan_lora/my_first_wan_lora_000010.safetensors
```

---

## 📂 STRUKTURA FINALNA - Co Gdzie Masz

```
/home/user/
│
├── musubi-tuner/                              (Repo - sklonowane)
│   ├── run_modal_v2.py                        (Uruchamiasz TO)
│   ├── my_wan_config.json                     (Config treningu - UTWORZYŁEŚ)
│   ├── my_qwen_config.json                    (Opcjonalny)
│   ├── src/musubi_tuner/                      (Kod)
│   └── docs/                                  (Dokumentacja)
│
├── my_datasets/                               (Datasety - UTWORZYŁEŚ)
│   ├── wan_dataset_1/
│   │   ├── config.toml                        (UTWORZYŁEŚ)
│   │   └── videos/
│   │       ├── video001.mp4                   (WRZUCIŁEŚ)
│   │       ├── video001.txt                   (UTWORZYŁEŚ)
│   │       └── ...
│   │
│   └── qwen_dataset_1/
│       ├── config.toml
│       └── images/
│           ├── img001.jpg
│           ├── img001.txt
│           └── ...
│
└── trained_loras/                             (Downloaded z Modal)
    └── my_first_wan_lora/
        └── my_first_wan_lora_000010.safetensors  <- GOTOWY LORA!
```

---

## 🎯 KOMENDY W SKRÓCIE (Quick Reference)

```bash
# SETUP (raz)
cd ~
git clone https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
git checkout claude/analyze-lora-training-repo-011CUecwmPeMRMZWQqZH52qX
pip install modal
modal setup
modal volume create musubi-models musubi-data musubi-cache

# DATASET (za każdym razem)
cd ~
mkdir -p my_datasets/wan_dataset_1/videos
# (wrzuć pliki + stwórz config.toml)
modal volume put musubi-data wan_dataset_1/ ./my_datasets/wan_dataset_1/

# CONFIG + RUN (za każdym razem)
cd ~/musubi-tuner
nano my_wan_config.json  # (wklej config)
modal run run_modal_v2.py --model wan --config my_wan_config.json

# DOWNLOAD (po treningu)
mkdir -p ~/trained_loras
modal volume get musubi-models outputs/my_first_wan_lora ~/trained_loras/

# GOTOWE!
```

---

## 🆘 NAJCZĘSTSZE PROBLEMY

### "run_modal_v2.py: No such file"

```bash
# Jesteś w złym folderze!
cd ~/musubi-tuner

# Sprawdź:
pwd
# Powinno być: /home/user/musubi-tuner

# Sprawdź czy plik jest:
ls -la run_modal_v2.py
```

### "Volume not found"

```bash
# Nie stworzyłeś volumes!
modal volume create musubi-models
modal volume create musubi-data
modal volume create musubi-cache
```

### "dataset_config: file not found"

```bash
# Dataset nie został uploadowany albo zła ścieżka!

# Sprawdź co jest w volume:
modal volume ls musubi-data

# Sprawdź czy ścieżka w JSON jest poprawna:
nano my_wan_config.json
# "dataset_config": "wan_dataset_1/config.toml"  <- musi się zgadzać!
```

### "modal: command not found"

```bash
# Modal nie zainstalowany!
pip install modal

# Sprawdź:
modal --version
```

---

## 🎓 KOLEJNY TRENING (inny dataset)

```bash
# 1. Nowy dataset
cd ~
mkdir -p my_datasets/wan_dataset_2/videos
# (wrzuć nowe pliki + config.toml)
modal volume put musubi-data wan_dataset_2/ ./my_datasets/wan_dataset_2/

# 2. Nowy config
cd ~/musubi-tuner
nano my_wan_config_v2.json
# ZMIEŃ:
# - "dataset_config": "wan_dataset_2/config.toml"
# - "output_name": "my_second_wan_lora"  (MUSI BYĆ INNE!)

# 3. Run (modele już są w cache - szybko!)
modal run run_modal_v2.py --model wan --config my_wan_config_v2.json

# 4. Download
modal volume get musubi-models outputs/my_second_wan_lora ~/trained_loras/
```

---

**PYTANIA? Pytaj o konkretny krok z numerem!**
