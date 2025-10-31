"""
Musubi-Tuner Modal.com Integration

Run LoRA training for Wan 2.1/2.2 and Qwen-Image models on Modal.com cloud GPUs.

Usage:
    # Full pipeline (cache + train)
    modal run run_modal.py --model wan --config config.json
    modal run run_modal.py --model qwen --config config.json

    # Individual stages
    modal run run_modal.py --model wan --config config.json --stage cache
    modal run run_modal.py --model wan --config config.json --stage train

    # Download results
    modal volume get musubi-models outputs/my_lora ./local_outputs/
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import sys
import json
import modal

# =============================================================================
# CONFIGURATION
# =============================================================================

# Modal Volumes for persistent storage
MODELS_VOLUME = modal.Volume.from_name("musubi-models", create_if_missing=True)
DATA_VOLUME = modal.Volume.from_name("musubi-data", create_if_missing=True)
CACHE_VOLUME = modal.Volume.from_name("musubi-cache", create_if_missing=True)

# Paths in Modal container
MODELS_PATH = "/models"
DATA_PATH = "/data"
CACHE_PATH = "/cache"
CODE_PATH = "/root/musubi-tuner"

# =============================================================================
# DOCKER IMAGE DEFINITION
# =============================================================================

def create_musubi_image():
    """Create Modal image with all musubi-tuner dependencies"""
    return (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install(
            "git",
            "wget",
            "ffmpeg",  # for video processing
            "libgl1-mesa-glx",  # for opencv
            "libglib2.0-0",
        )
        .pip_install(
            # PyTorch with CUDA
            "torch>=2.5.1",
            "torchvision>=0.20.1",
            index_url="https://download.pytorch.org/whl/cu124",
        )
        .pip_install(
            # Core dependencies
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
            "av==14.0.1",
            # Wan specific
            "ftfy==6.3.1",
            "easydict==1.13",
            # FLUX Kontext specific
            "sentencepiece==0.2.1",
            # Optional but recommended
            "hf_transfer",
        )
        # Install xformers for faster attention (optional but recommended)
        .run_commands(
            "pip install xformers --index-url https://download.pytorch.org/whl/cu124 || echo 'xformers install failed, continuing without it'"
        )
    )

image = create_musubi_image()

# Mount local code
code_mount = modal.Mount.from_local_dir(
    ".",
    remote_path=CODE_PATH,
    condition=lambda path: not any([
        ".git" in path,
        "__pycache__" in path,
        ".venv" in path,
        "venv" in path,
        ".pytest_cache" in path,
        "outputs" in path,
        ".modal" in path,
    ])
)

# Create Modal App
app = modal.App("musubi-tuner", image=image)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_command(cmd: list, description: str = ""):
    """Run subprocess command and return result"""
    import subprocess

    if description:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"{'='*60}\n")

    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

# =============================================================================
# WAN 2.1/2.2 FUNCTIONS
# =============================================================================

@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={
        MODELS_PATH: MODELS_VOLUME,
        DATA_PATH: DATA_VOLUME,
        CACHE_PATH: CACHE_VOLUME,
    },
    mounts=[code_mount],
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)
def wan_cache_latents(
    dataset_config: str,
    vae_path: str,
    clip_path: str = None,
    task: str = "wan_t2v_14B",
    i2v: bool = False,
    vae_cache_cpu: bool = False,
):
    """Cache latents for Wan training"""

    cmd = [
        "python", f"{CODE_PATH}/src/musubi_tuner/wan_cache_latents.py",
        "--dataset_config", f"{DATA_PATH}/{dataset_config}",
        "--vae", f"{MODELS_PATH}/{vae_path}",
        "--task", task,
    ]

    if i2v:
        cmd.append("--i2v")

    if clip_path:
        cmd.extend(["--clip", f"{MODELS_PATH}/{clip_path}"])

    if vae_cache_cpu:
        cmd.append("--vae_cache_cpu")

    result = run_command(cmd, "Caching latents for Wan")

    # Commit cache to volume
    CACHE_VOLUME.commit()

    return {
        "status": "success" if result["returncode"] == 0 else "error",
        "returncode": result["returncode"],
    }


@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={
        MODELS_PATH: MODELS_VOLUME,
        DATA_PATH: DATA_VOLUME,
        CACHE_PATH: CACHE_VOLUME,
    },
    mounts=[code_mount],
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)
def wan_cache_text_encoder(
    dataset_config: str,
    t5_path: str,
    batch_size: int = 16,
    fp8_t5: bool = False,
):
    """Cache text encoder outputs for Wan training"""

    cmd = [
        "python", f"{CODE_PATH}/src/musubi_tuner/wan_cache_text_encoder_outputs.py",
        "--dataset_config", f"{DATA_PATH}/{dataset_config}",
        "--t5", f"{MODELS_PATH}/{t5_path}",
        "--batch_size", str(batch_size),
    ]

    if fp8_t5:
        cmd.append("--fp8_t5")

    result = run_command(cmd, "Caching text encoder outputs for Wan")

    # Commit cache to volume
    CACHE_VOLUME.commit()

    return {
        "status": "success" if result["returncode"] == 0 else "error",
        "returncode": result["returncode"],
    }


@app.function(
    gpu="A100-80GB",  # Can be changed via config
    timeout=14400,    # 4 hours, can be changed via config
    volumes={
        MODELS_PATH: MODELS_VOLUME,
        DATA_PATH: DATA_VOLUME,
        CACHE_PATH: CACHE_VOLUME,
    },
    mounts=[code_mount],
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)
def wan_train(config: dict):
    """
    Train Wan LoRA model

    config should contain:
    - task: "wan_t2v_14B", "wan_i2v_14B", etc.
    - dit_path, vae_path, t5_path, clip_path (optional)
    - dataset_config: path to TOML
    - output_name: name for output files
    - network_dim, network_alpha
    - learning_rate, max_train_epochs
    - ... other training parameters
    """
    import os

    # Prepare paths
    dit_full = f"{MODELS_PATH}/{config['dit_path']}"
    vae_full = f"{MODELS_PATH}/{config['vae_path']}"
    t5_full = f"{MODELS_PATH}/{config['t5_path']}"
    config_full = f"{DATA_PATH}/{config['dataset_config']}"
    output_dir = f"{MODELS_PATH}/outputs/{config['output_name']}"

    # Set accelerate to work without config file (single GPU mode)
    os.environ["ACCELERATE_CONFIG_FILE"] = "none"

    # Build command
    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", config.get("mixed_precision", "bf16"),
        f"{CODE_PATH}/src/musubi_tuner/wan_train_network.py",
        "--task", config["task"],
        "--dit", dit_full,
        "--vae", vae_full,
        "--t5", t5_full,
        "--dataset_config", config_full,
        "--network_module", "networks.lora_wan",
        "--network_dim", str(config.get("network_dim", 32)),
        "--network_alpha", str(config.get("network_alpha", 16)),
        "--learning_rate", str(config.get("learning_rate", "1e-4")),
        "--optimizer_type", config.get("optimizer_type", "adamw8bit"),
        "--max_train_epochs", str(config.get("max_train_epochs", 10)),
        "--save_every_n_epochs", str(config.get("save_every_n_epochs", 1)),
        "--output_dir", output_dir,
        "--output_name", config["output_name"],
        "--sdpa",
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
    ]

    # Optional: CLIP for Wan 2.1
    if config.get("clip_path"):
        cmd.extend(["--clip", f"{MODELS_PATH}/{config['clip_path']}"])

    # Optional: Wan 2.2 high noise model
    if config.get("dit_high_noise_path"):
        cmd.extend(["--dit_high_noise", f"{MODELS_PATH}/{config['dit_high_noise_path']}"])

    if config.get("timestep_boundary") is not None:
        cmd.extend(["--timestep_boundary", str(config["timestep_boundary"])])

    # Memory optimization flags
    if config.get("blocks_to_swap"):
        cmd.extend(["--blocks_to_swap", str(config["blocks_to_swap"])])

    if config.get("fp8_base"):
        cmd.append("--fp8_base")

    if config.get("fp8_llm"):
        cmd.append("--fp8_llm")

    if config.get("seed"):
        cmd.extend(["--seed", str(config["seed"])])

    # Additional args from config
    if config.get("additional_args"):
        cmd.extend(config["additional_args"])

    result = run_command(cmd, "Training Wan LoRA")

    # Commit model to volume - CRITICAL!
    MODELS_VOLUME.commit()

    return {
        "status": "success" if result["returncode"] == 0 else "error",
        "returncode": result["returncode"],
        "output_path": output_dir,
    }


# =============================================================================
# QWEN-IMAGE FUNCTIONS
# =============================================================================

@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={
        MODELS_PATH: MODELS_VOLUME,
        DATA_PATH: DATA_VOLUME,
        CACHE_PATH: CACHE_VOLUME,
    },
    mounts=[code_mount],
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)
def qwen_cache_latents(
    dataset_config: str,
    vae_path: str,
    edit: bool = False,
    edit_plus: bool = False,
):
    """Cache latents for Qwen-Image training"""

    cmd = [
        "python", f"{CODE_PATH}/src/musubi_tuner/qwen_image_cache_latents.py",
        "--dataset_config", f"{DATA_PATH}/{dataset_config}",
        "--vae", f"{MODELS_PATH}/{vae_path}",
    ]

    if edit:
        cmd.append("--edit")
    elif edit_plus:
        cmd.append("--edit_plus")

    result = run_command(cmd, "Caching latents for Qwen-Image")

    # Commit cache to volume
    CACHE_VOLUME.commit()

    return {
        "status": "success" if result["returncode"] == 0 else "error",
        "returncode": result["returncode"],
    }


@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={
        MODELS_PATH: MODELS_VOLUME,
        DATA_PATH: DATA_VOLUME,
        CACHE_PATH: CACHE_VOLUME,
    },
    mounts=[code_mount],
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)
def qwen_cache_text_encoder(
    dataset_config: str,
    text_encoder_path: str,
    batch_size: int = 1,
    fp8_vl: bool = False,
    edit: bool = False,
    edit_plus: bool = False,
):
    """Cache text encoder outputs for Qwen-Image training"""

    cmd = [
        "python", f"{CODE_PATH}/src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py",
        "--dataset_config", f"{DATA_PATH}/{dataset_config}",
        "--text_encoder", f"{MODELS_PATH}/{text_encoder_path}",
        "--batch_size", str(batch_size),
    ]

    if fp8_vl:
        cmd.append("--fp8_vl")
    if edit:
        cmd.append("--edit")
    elif edit_plus:
        cmd.append("--edit_plus")

    result = run_command(cmd, "Caching text encoder outputs for Qwen-Image")

    # Commit cache to volume
    CACHE_VOLUME.commit()

    return {
        "status": "success" if result["returncode"] == 0 else "error",
        "returncode": result["returncode"],
    }


@app.function(
    gpu="A100-80GB",  # Can be changed via config
    timeout=14400,    # 4 hours, can be changed via config
    volumes={
        MODELS_PATH: MODELS_VOLUME,
        DATA_PATH: DATA_VOLUME,
        CACHE_PATH: CACHE_VOLUME,
    },
    mounts=[code_mount],
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)
def qwen_train(config: dict):
    """
    Train Qwen-Image LoRA model

    config should contain:
    - dit_path, vae_path, text_encoder_path
    - dataset_config: path to TOML
    - output_name: name for output files
    - network_dim, network_alpha
    - learning_rate, max_train_epochs
    - edit: bool (for Qwen-Image-Edit)
    - edit_plus: bool (for Qwen-Image-Edit-2509)
    - ... other training parameters
    """
    import os

    # Prepare paths
    dit_full = f"{MODELS_PATH}/{config['dit_path']}"
    vae_full = f"{MODELS_PATH}/{config['vae_path']}"
    te_full = f"{MODELS_PATH}/{config['text_encoder_path']}"
    config_full = f"{DATA_PATH}/{config['dataset_config']}"
    output_dir = f"{MODELS_PATH}/outputs/{config['output_name']}"

    # Set accelerate to work without config file (single GPU mode)
    os.environ["ACCELERATE_CONFIG_FILE"] = "none"

    # Build command
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
        "--network_dim", str(config.get("network_dim", 16)),
        "--network_alpha", str(config.get("network_alpha", 8)),
        "--learning_rate", str(config.get("learning_rate", "5e-5")),
        "--optimizer_type", config.get("optimizer_type", "adamw8bit"),
        "--max_train_epochs", str(config.get("max_train_epochs", 16)),
        "--save_every_n_epochs", str(config.get("save_every_n_epochs", 1)),
        "--output_dir", output_dir,
        "--output_name", config["output_name"],
        "--sdpa",
        "--gradient_checkpointing",
        "--timestep_sampling", config.get("timestep_sampling", "shift"),
        "--weighting_scheme", config.get("weighting_scheme", "none"),
        "--discrete_flow_shift", str(config.get("discrete_flow_shift", 2.2)),
    ]

    # Edit mode
    if config.get("edit"):
        cmd.append("--edit")
    elif config.get("edit_plus"):
        cmd.append("--edit_plus")

    if config.get("fp8_vl"):
        cmd.append("--fp8_vl")

    if config.get("seed"):
        cmd.extend(["--seed", str(config["seed"])])

    # Additional args from config
    if config.get("additional_args"):
        cmd.extend(config["additional_args"])

    result = run_command(cmd, "Training Qwen-Image LoRA")

    # Commit model to volume - CRITICAL!
    MODELS_VOLUME.commit()

    return {
        "status": "success" if result["returncode"] == 0 else "error",
        "returncode": result["returncode"],
        "output_path": output_dir,
    }


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main(
    model: str,
    config: str,
    stage: str = "all",
):
    """
    Main entrypoint for Musubi-Tuner Modal pipeline

    Args:
        model: "wan" or "qwen"
        config: Path to JSON config file
        stage: "all", "cache", or "train"

    Example:
        modal run run_modal.py --model wan --config my_config.json
        modal run run_modal.py --model wan --config my_config.json --stage cache
        modal run run_modal.py --model qwen --config my_config.json --stage train
    """

    # Load config
    with open(config, 'r') as f:
        cfg = json.load(f)

    print(f"\n{'='*70}")
    print(f"🚀 Musubi-Tuner Modal Pipeline")
    print(f"{'='*70}")
    print(f"Model:  {model}")
    print(f"Stage:  {stage}")
    print(f"Config: {config}")
    print(f"{'='*70}\n")

    if model == "wan":
        # WAN PIPELINE
        if stage in ["all", "cache"]:
            print("\n📦 Stage 1/3: Caching latents...")
            result = wan_cache_latents.remote(
                dataset_config=cfg["dataset_config"],
                vae_path=cfg["vae_path"],
                clip_path=cfg.get("clip_path"),
                task=cfg.get("task", "wan_t2v_14B"),
                i2v=cfg.get("i2v", False),
                vae_cache_cpu=cfg.get("vae_cache_cpu", False),
            )
            print(f"✓ Latents cached: {result['status']}")

            if result['status'] == 'error':
                print(f"❌ Error caching latents, aborting.")
                return

            print("\n📦 Stage 2/3: Caching text encoder outputs...")
            result = wan_cache_text_encoder.remote(
                dataset_config=cfg["dataset_config"],
                t5_path=cfg["t5_path"],
                batch_size=cfg.get("cache_batch_size", 16),
                fp8_t5=cfg.get("fp8_t5", False),
            )
            print(f"✓ Text encoder outputs cached: {result['status']}")

            if result['status'] == 'error':
                print(f"❌ Error caching text encoder, aborting.")
                return

        if stage in ["all", "train"]:
            print("\n🎓 Stage 3/3: Training...")
            result = wan_train.remote(config_dict=cfg)
            print(f"✓ Training completed: {result['status']}")

            if result['status'] == 'success':
                print(f"\n📁 Model saved to: {result['output_path']}")
                print(f"\n💾 To download results:")
                print(f"   modal volume get musubi-models outputs/{cfg['output_name']} ./")
            else:
                print(f"❌ Training failed")

    elif model == "qwen":
        # QWEN PIPELINE
        if stage in ["all", "cache"]:
            print("\n📦 Stage 1/3: Caching latents...")
            result = qwen_cache_latents.remote(
                dataset_config=cfg["dataset_config"],
                vae_path=cfg["vae_path"],
                edit=cfg.get("edit", False),
                edit_plus=cfg.get("edit_plus", False),
            )
            print(f"✓ Latents cached: {result['status']}")

            if result['status'] == 'error':
                print(f"❌ Error caching latents, aborting.")
                return

            print("\n📦 Stage 2/3: Caching text encoder outputs...")
            result = qwen_cache_text_encoder.remote(
                dataset_config=cfg["dataset_config"],
                text_encoder_path=cfg["text_encoder_path"],
                batch_size=cfg.get("cache_batch_size", 1),
                fp8_vl=cfg.get("fp8_vl", False),
                edit=cfg.get("edit", False),
                edit_plus=cfg.get("edit_plus", False),
            )
            print(f"✓ Text encoder outputs cached: {result['status']}")

            if result['status'] == 'error':
                print(f"❌ Error caching text encoder, aborting.")
                return

        if stage in ["all", "train"]:
            print("\n🎓 Stage 3/3: Training...")
            result = qwen_train.remote(config=cfg)
            print(f"✓ Training completed: {result['status']}")

            if result['status'] == 'success':
                print(f"\n📁 Model saved to: {result['output_path']}")
                print(f"\n💾 To download results:")
                print(f"   modal volume get musubi-models outputs/{cfg['output_name']} ./")
            else:
                print(f"❌ Training failed")

    else:
        print(f"❌ Unknown model: {model}. Must be 'wan' or 'qwen'")
        return

    print(f"\n{'='*70}")
    print(f"✨ Pipeline completed!")
    print(f"{'='*70}\n")
