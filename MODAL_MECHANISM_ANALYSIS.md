# Szczegółowa Analiza Mechanizmu Modal.com w AI-Toolkit

## 🎯 Cel Dokumentu

Ten dokument wyjaśnia **JAK DOKŁADNIE DZIAŁA** integracja ostris/ai-toolkit z Modal.com - nie konfiguracja, ale sam mechanizm wykonania kodu w chmurze.

---

## 📊 Flow Wykonania: Od Local CLI do Cloud GPU

### Schemat High-Level

```
┌─────────────────────────────────────────────────────────────────┐
│  TWÓJ KOMPUTER (Local)                                          │
│                                                                  │
│  $ modal run run_modal.py \                                     │
│      --config-file-list-str=/root/ai-toolkit/config/my.yml     │
│                                                                  │
│  Modal CLI:                                                     │
│  1. Parsuje run_modal.py                                        │
│  2. Znajduje @app.function(gpu="A100", ...)                     │
│  3. Serializuje argumenty (config_file_list_str)                │
│  4. Przesyła request do Modal API                               │
│  5. Montuje lokalny kod przez Mount.from_local_dir()            │
│  6. KONIEC - lokalnie nic więcej się nie dzieje                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP Request do Modal API
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODAL.COM INFRASTRUCTURE (Cloud)                               │
│                                                                  │
│  1. Tworzy kontener z image (debian + pip packages)             │
│  2. Alokuje GPU (A100)                                          │
│  3. Montuje Volume: "flux-lora-models" → /root/.../modal_output│
│  4. Kopiuje lokalny kod: ai-toolkit → /root/ai-toolkit          │
│  5. Uruchamia funkcję main(config_file_list_str="...")         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  KONTENER NA MODAL (w chmurze, ma GPU)                          │
│                                                                  │
│  def main(config_file_list_str: str):                           │
│      # WSZYSTKO TO DZIAŁA W CHMURZE, NIE LOKALNIE!             │
│                                                                  │
│      config_file_list = config_file_list_str.split(",")        │
│      for config_file in config_file_list:                       │
│          job = get_job(config_file, name)  # ← ładuje YAML     │
│                                                                  │
│          # ZMIENIA training_folder na modal_output              │
│          job.config['process'][0]['training_folder'] = MOUNT_DIR│
│                                                                  │
│          job.run()  # ← TUTAJ DZIEJE SIĘ TRENING               │
│                                                                  │
│          model_volume.commit()  # ← Zapisz do persistent storage│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  job.run() - Wewnętrzny Flow AI-Toolkit                         │
│                                                                  │
│  1. get_job() tworzy ExtensionJob                               │
│  2. ExtensionJob.load_processes() ładuje proces (sd_trainer)    │
│  3. process.run() uruchamia BaseSDTrainProcess                  │
│  4. BaseSDTrainProcess:                                         │
│     - Ładuje model z HuggingFace                                │
│     - Ładuje dataset z folder_path                              │
│     - Tworzy LoRA network                                       │
│     - Uruchamia pętlę treningową                                │
│     - Zapisuje checkpointy do training_folder (Volume)          │
│     - Generuje sample images co N kroków                        │
│  5. Kończy, wszystko zapisane w Volume                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODAL VOLUME (Persistent Storage)                              │
│                                                                  │
│  /flux-lora-models/                                             │
│  ├── my_first_flux_lora_v1/                                     │
│  │   ├── my_first_flux_lora_v1_000500.safetensors             │
│  │   ├── my_first_flux_lora_v1_001000.safetensors             │
│  │   ├── my_first_flux_lora_v1_002000.safetensors             │
│  │   ├── samples/                                              │
│  │   │   ├── sample_000500_0.webp                             │
│  │   │   ├── sample_001000_0.webp                             │
│  │   └── optimizer.pt                                          │
│                                                                  │
│  Model zapisany! Można pobrać lokalnie:                         │
│  $ modal volume get flux-lora-models my_first_flux_lora_v1 ./  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Szczegółowa Analiza Kodu run_modal.py

### 1. Definicja Image (Środowisko Docker)

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")  # System dependencies
    .pip_install(
        "torch",
        "diffusers[torch]",
        "transformers",
        # ... długa lista pakietów
    )
)
```

**Co to robi:**
- Tworzy **definicję** obrazu Docker (nie uruchamia!)
- Modal zbuduje ten obraz raz, a potem cache'uje
- Każde wywołanie `@app.function` używa tego samego obrazu
- Obraz jest budowany **na serwerach Modal**, nie lokalnie

**Kiedy jest budowany:**
- Pierwszym razem gdy wywołujesz `modal run`
- Gdy zmienisz definicję (dodasz/usuniesz pakiet)
- Po tym jest cache'owany (szybkie uruchomienia)

### 2. Volume - Persistent Storage

```python
model_volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)
MOUNT_DIR = "/root/ai-toolkit/modal_output"
```

**Co to robi:**
- `modal.Volume` - to jak zewnętrzny dysk twardy w chmurze
- Przeżywa **między wywołaniami** funkcji
- Można do niego zapisywać pliki i one **zostają**
- Bez Volume - wszystko znika po zakończeniu funkcji!

**Gdzie jest Volume:**
- Nie na Twoim komputerze
- Na serwerach Modal - storage S3-like
- Dostępny przez: https://modal.com/storage/USERNAME/VOLUME_NAME

**Jak działa zapis:**
```python
# W funkcji main():
job.run()  # zapisuje pliki do /root/ai-toolkit/modal_output
model_volume.commit()  # <-- BEZ TEGO PLIKI ZNIKAJĄ!
```

**WAŻNE:** `volume.commit()` jest KONIECZNY! Bez niego pliki w Volume nie zostaną zapisane.

### 3. Mount - Transfer Lokalnego Kodu

```python
code_mount = modal.Mount.from_local_dir(
    "/Users/username/ai-toolkit",  # Ścieżka lokalna
    remote_path="/root/ai-toolkit"  # Ścieżka w kontenerze
)
```

**Co to robi:**
- Kopiuje CAŁY folder lokalny do kontenera w chmurze
- Dzieje się to **automatycznie** przed uruchomieniem funkcji
- Kod jest **read-only** w kontenerze

**Kiedy jest kopiowany:**
- Za każdym razem gdy wywołujesz funkcję
- Modal wykrywa zmiany i wysyła tylko delty (szybkie)
- Nie musisz ręcznie uploadować

**Condition Filter:**
```python
condition=lambda path: not any([
    ".git" in path,
    "__pycache__" in path,
])
```
- Możesz wykluczyć pewne foldery z kopiowania
- Przyspiesza transfer

### 4. App i Function Decorator

```python
app = modal.App(
    name="flux-lora-training",
    image=image,
    mounts=[code_mount],
    volumes={MOUNT_DIR: model_volume}
)

@app.function(
    gpu="A100",
    timeout=7200,  # 2 godziny
)
def main(config_file_list_str: str, recover: bool = False, name: str = None):
    # ... kod funkcji ...
```

**Co to robi:**

#### `modal.App`
- Kontener dla całej aplikacji
- Definiuje współdzielone zasoby (image, volumes, mounts)
- Jedna aplikacja może mieć wiele funkcji

#### `@app.function(gpu="A100", timeout=7200)`
- **To jest magiczne!** Oznacza funkcję która będzie wykonana W CHMURZE
- `gpu="A100"` - alokuje GPU A100 na czas wykonania
- `timeout=7200` - maksymalny czas wykonania (2h)

**Kluczowe zrozumienie:**
```python
def main(config_file_list_str: str):
    print("To drukuje SIĘ W CHMURZE, nie lokalnie!")
    # Cała funkcja działa na serwerze Modal z GPU
```

### 5. Wywołanie Funkcji

**Local CLI:**
```bash
modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/my.yml
```

**Co się dzieje:**
1. Modal CLI parsuje `run_modal.py`
2. Znajduje `if __name__ == "__main__":` na końcu
3. Parsuje argumenty CLI
4. Konwertuje je na parametry funkcji `main()`
5. Wywołuje `main.call(config_file_list_str="...")` - przesyła do Modal API

**W chmurze (automatycznie):**
```python
def main(config_file_list_str: str, recover: bool = False, name: str = None):
    # TA FUNKCJA DZIAŁA NA SERWERZE MODAL!

    # 1. Split config files
    config_file_list = config_file_list_str.split(",")

    # 2. Dla każdego config
    for config_file in config_file_list:
        # 3. Załaduj job z YAML
        job = get_job(config_file, name)

        # 4. ZMIEŃ training_folder na Volume path
        job.config['process'][0]['training_folder'] = MOUNT_DIR
        os.makedirs(MOUNT_DIR, exist_ok=True)

        # 5. URUCHOM TRENING (największa część czasu)
        job.run()

        # 6. ZAPISZ wyniki do Volume
        model_volume.commit()  # <-- KRYTYCZNE!

        # 7. Cleanup
        job.cleanup()
```

---

## 🎓 Mechanizm job.run() - AI-Toolkit Internals

### Flow w ai-toolkit

```python
# 1. get_job() - toolkit/job.py
def get_job(config_path: str, name=None):
    config = get_config(config_path, name)  # ładuje YAML
    job = config['job']  # np. "extension"

    if job == 'extension':
        from jobs import ExtensionJob
        return ExtensionJob(config)
```

```python
# 2. ExtensionJob - jobs/ExtensionJob.py
class ExtensionJob(BaseJob):
    def run(self):
        for process in self.process:
            process.run()  # uruchamia proces sd_trainer
```

```python
# 3. BaseSDTrainProcess - jobs/process/BaseSDTrainProcess.py
class BaseSDTrainProcess:
    def run(self):
        # Ładuje model
        self.model = self.load_model()

        # Ładuje dataset
        self.dataset = self.load_dataset()

        # Tworzy LoRA network
        self.network = self.create_network()

        # PĘTLA TRENINGOWA
        for step in range(total_steps):
            loss = self.train_step(batch)

            if step % save_every == 0:
                self.save_checkpoint(step)

            if step % sample_every == 0:
                self.generate_sample(step)

        # Zapisz finalny model
        self.save_final_model()
```

### Kluczowa Zmiana dla Modal

```python
# W main() w run_modal.py:
job.config['process'][0]['training_folder'] = MOUNT_DIR

# Przed:
# training_folder: "output"  (lokalne)

# Po zmianie:
# training_folder: "/root/ai-toolkit/modal_output"  (Volume!)
```

**Dlaczego to ważne:**
- `training_folder` to miejsce gdzie zapisywane są:
  - Checkpointy modelu (.safetensors)
  - Sample images/videos
  - Optimizer state
  - Logi
- Musi być w Volume, żeby przetrwało po zakończeniu funkcji!

---

## 🚀 Kluczowe Różnice: Local vs Cloud Execution

### Co Dzieje się LOKALNIE:

```python
# Na Twoim komputerze:
$ modal run run_modal.py --config-file-list-str=...

# Modal CLI:
# 1. Parsuje plik
# 2. Znajduje @app.function
# 3. Wysyła request HTTP do Modal API
# 4. Kopiuje kod przez Mount
# 5. Koniec - NIE uruchamia funkcji lokalnie!
```

**Twój komputer NIE:**
- ❌ Nie ładuje modelu
- ❌ Nie uruchamia treningu
- ❌ Nie używa GPU
- ❌ Nie zapisuje plików
- ✅ TYLKO wysyła request do Modal

### Co Dzieje Się W CHMURZE (Modal):

```python
# Na serwerze Modal (z GPU):
# 1. Tworzy kontener z image
# 2. Montuje Volume
# 3. Kopiuje kod z Mount
# 4. Uruchamia funkcję main()
# 5. main() wywołuje job.run()
# 6. Trening działa godzinami
# 7. Zapisuje do Volume
# 8. Volume.commit()
# 9. Kończy, usuwa kontener
```

**Serwer Modal:**
- ✅ Ładuje model z HuggingFace
- ✅ Ładuje dataset z folder_path
- ✅ Uruchamia trening z GPU
- ✅ Zapisuje checkpointy do Volume
- ✅ Generuje samples
- ✅ Wszystko dzieje się tu, nie lokalnie!

---

## 💾 Volume Persistence - Jak Działają Pliki

### Bez Volume (zły sposób):

```python
@app.function(gpu="A100")
def train():
    model.save("model.safetensors")  # Zapisuje do /tmp
    # Po zakończeniu funkcji - PLIK ZNIKA! 😱
```

### Z Volume (poprawny sposób):

```python
volume = modal.Volume.from_name("my-volume", create_if_missing=True)

@app.function(gpu="A100", volumes={"/outputs": volume})
def train():
    model.save("/outputs/model.safetensors")  # Zapisuje do Volume
    volume.commit()  # KONIECZNE!
    # Po zakończeniu - plik POZOSTAJE w Volume ✅
```

### Download z Volume:

```bash
# Lista plików
modal volume ls flux-lora-models

# Output:
# my_first_flux_lora_v1/
# my_first_flux_lora_v1_000500.safetensors
# my_first_flux_lora_v1_002000.safetensors

# Download
modal volume get flux-lora-models my_first_flux_lora_v1 ./local_folder/

# Teraz masz pliki lokalnie!
```

---

## 🔧 Parametry GPU

```python
@app.function(gpu="A100")
@app.function(gpu="A100-80GB")
@app.function(gpu="H100")
@app.function(gpu="A10G")
@app.function(gpu="T4")
@app.function(gpu=["A100", "H100"])  # Preferuje A100, fallback H100
@app.function(gpu="A100:2")  # 2x A100 (multi-GPU)
```

**Wybór GPU:**
- `A10G` - najtańsze (~$0.60/hr), 24GB, dobre do cache i testów
- `A100-40GB` - standardowy (~$2.50/hr), dobry do większości treningów
- `A100-80GB` - większy (~$3.00/hr), dla dużych modeli
- `H100` - najszybszy (~$5.00/hr), najnowsza architektura

**Modal automatycznie:**
- Wybiera dostępne GPU
- Czeka jeśli wszystkie zajęte
- Alokuje i dealokuje na żądanie

---

## 🎯 Kluczowe Wnioski dla Musubi-Tuner

### 1. Struktura którą musimy stworzyć:

```python
# run_modal_musubi.py

import modal

# 1. Definicja Image
image = modal.Image.debian_slim(python_version="3.10").pip_install(...)

# 2. Volumes
models_volume = modal.Volume.from_name("musubi-models", create_if_missing=True)
data_volume = modal.Volume.from_name("musubi-data", create_if_missing=True)

# 3. Mount lokalnego kodu
code_mount = modal.Mount.from_local_dir(".", remote_path="/root/musubi-tuner")

# 4. App
app = modal.App("musubi-tuner")

# 5. Funkcja treningowa
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=14400,
    volumes={
        "/models": models_volume,
        "/data": data_volume,
    },
    mounts=[code_mount],
)
def train_wan(config: dict):
    import subprocess

    # Musubi-tuner używa accelerate launch
    cmd = [
        "accelerate", "launch",
        "--mixed_precision", "bf16",
        "/root/musubi-tuner/src/musubi_tuner/wan_train_network.py",
        "--dit", f"/models/{config['dit_path']}",
        "--vae", f"/models/{config['vae_path']}",
        # ... więcej argumentów
    ]

    result = subprocess.run(cmd)

    # KONIECZNE!
    models_volume.commit()

    return {"status": "success" if result.returncode == 0 else "error"}
```

### 2. Różnice vs AI-Toolkit:

| Aspekt | AI-Toolkit | Musubi-Tuner |
|--------|-----------|--------------|
| **Struktura** | Własny framework (job.run()) | Standalone scripts + accelerate |
| **Config** | YAML z zagnieżdżoną strukturą | TOML dataset + CLI args |
| **Uruchomienie** | `job = get_job(); job.run()` | `subprocess.run(["accelerate", ...])` |
| **Pre-caching** | Wbudowane (cache_latents_to_disk) | Osobne skrypty (cache_latents.py) |
| **Modularność** | Wysokie abstrakcje (BaseSDTrainProcess) | Bezpośrednie skrypty |

### 3. Co musimy zaadaptować:

1. **Musubi używa accelerate** - musimy uruchomić przez subprocess
2. **Musubi ma osobne skrypty cache** - potrzebujemy oddzielnych funkcji Modal
3. **Musubi używa TOML + CLI args** - nie YAML
4. **Musubi nie ma "job" abstrakcji** - bezpośrednie wywołania skryptów

---

## 📝 Podsumowanie Mechanizmu Modal

### Modal to NIE jest:
- ❌ Prosty wrapper na subprocess
- ❌ SSH do zdalnego serwera
- ❌ Docker compose w chmurze

### Modal TO:
- ✅ Serverless compute platform
- ✅ Funkcje wykonywane on-demand w chmurze
- ✅ Automatyczne zarządzanie kontenerami i GPU
- ✅ Persistent storage przez Volumes
- ✅ Pay-per-second billing

### Fundamentalny Flow:
```
LOCAL:
  modal run script.py

MODAL API:
  ↓ Tworzy kontener
  ↓ Alokuje GPU
  ↓ Montuje volumes
  ↓ Kopiuje kod

CLOUD (with GPU):
  ↓ Uruchamia @app.function
  ↓ Wykonuje kod
  ↓ Zapisuje do volume
  ↓ Commit volume

RESULT:
  ↓ Pliki w Volume

LOCAL:
  modal volume get ... (download)
```

### Dla Musubi-Tuner:
1. Stworzymy funkcje Modal dla każdego etapu:
   - `cache_latents_wan()`
   - `cache_text_encoder_wan()`
   - `train_wan()`
   - `train_qwen()`
2. Każda funkcja uruchamia odpowiedni skrypt przez subprocess
3. Wszystko zapisujemy do Volumes
4. Użytkownik wywołuje: `modal run run_modal_musubi.py --model wan --config my_config.yaml`

---

**Następny Krok:** Implementacja konkretnego kodu dla musubi-tuner bazując na tym mechanizmie.
