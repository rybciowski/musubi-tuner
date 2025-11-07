"""
Musubi-Tuner Modal.com Integration - V2 with Auto Model Download

Run LoRA training for Wan 2.1/2.2 and Qwen-Image models on Modal.com cloud GPUs.
Models are automatically downloaded from HuggingFace on first use.

Usage:
    modal run run_modal.py --model wan --config config.json
    modal run run_modal.py --model qwen --config config.json

Models are cached in volume after first download - no need to re-download!
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import sys
import json
import modal
from pathlib import Path

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
# MODEL REGISTRY - HuggingFace repos for auto-download
# =============================================================================

MODEL_REGISTRY = {
    # Wan 2.1 models
    "wan21_t2v_14b_bf16": {
        "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "files": ["split_files/diffusion_models/wan_t2v_14B_bf16.safetensors"],
        "cache_path": "wan/wan_t2v_14B_bf16.safetensors"
    },
    "wan21_i2v_14b_bf16": {
        "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "files": ["split_files/diffusion_models/wan_i2v_14B_bf16.safetensors"],
        "cache_path": "wan/wan_i2v_14B_bf16.safetensors"
    },
    "wan21_vae": {
        "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "files": ["split_files/vae/wan_2.1_vae.safetensors"],
        "cache_path": "wan/wan_2.1_vae.safetensors"
    },
    "wan21_t5": {
        "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
        "files": ["models_t5_umt5-xxl-enc-bf16.pth"],
        "cache_path": "wan/models_t5_umt5-xxl-enc-bf16.pth"
    },
    "wan21_clip": {
        "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
        "files": ["models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        "cache_path": "wan/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    },

    # Wan 2.2 models (bf16)
    "wan22_t2v_14b_low_bf16": {
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "files": ["split_files/diffusion_models/wan_2.2_t2v_14B_low_noise_bf16.safetensors"],
        "cache_path": "wan/wan_2.2_t2v_14B_low_noise_bf16.safetensors"
    },
    "wan22_t2v_14b_high_bf16": {
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "files": ["split_files/diffusion_models/wan_2.2_t2v_14B_high_noise_bf16.safetensors"],
        "cache_path": "wan/wan_2.2_t2v_14B_high_noise_bf16.safetensors"
    },

    # Wan 2.2 models (fp16)
    "wan22_t2v_14b_low_fp16": {
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "files": ["split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors"],
        "cache_path": "wan/wan2.2_t2v_low_noise_14B_fp16.safetensors"
    },
    "wan22_t2v_14b_high_fp16": {
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "files": ["split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors"],
        "cache_path": "wan/wan2.2_t2v_high_noise_14B_fp16.safetensors"
    },

    # Qwen-Image models
    "qwen_image_bf16": {
        "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
        "files": ["split_files/diffusion_models/qwen_image_bf16.safetensors"],
        "cache_path": "qwen/qwen_image_bf16.safetensors"
    },
    "qwen_image_edit_bf16": {
        "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
        "files": ["split_files/diffusion_models/qwen_image_edit_bf16.safetensors"],
        "cache_path": "qwen/qwen_image_edit_bf16.safetensors"
    },
    "qwen_image_edit_2509_bf16": {
        "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
        "files": ["split_files/diffusion_models/qwen_image_edit_2509_bf16.safetensors"],
        "cache_path": "qwen/qwen_image_edit_2509_bf16.safetensors"
    },
    "qwen_vae": {
        "repo_id": "Qwen/Qwen-Image",
        "files": ["vae/diffusion_pytorch_model.safetensors"],
        "cache_path": "qwen/vae_diffusion_pytorch_model.safetensors"
    },
    "qwen_vl_7b": {
        "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
        "files": ["split_files/text_encoders/qwen_2.5_vl_7b.safetensors"],
        "cache_path": "qwen/qwen_2.5_vl_7b.safetensors"
    },
}

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
            "ffmpeg",
            "libgl1-mesa-glx",
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
            "ftfy==6.3.1",
            "easydict==1.13",
            "sentencepiece==0.2.1",
            "hf_transfer",
        )
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
# MODEL DOWNLOAD HELPER
# =============================================================================

@app.function(
    volumes={MODELS_PATH: MODELS_VOLUME},
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
    timeout=3600,
)
def ensure_model_downloaded(model_path: str, force_download: bool = False):
    """
    Ensure model is available in volume. Download from HuggingFace if needed.

    Args:
        model_path: Path in volume (e.g., "wan/wan_t2v_14B_bf16.safetensors")
        force_download: Force re-download even if file exists

    Returns:
        Full path to model in volume
    """
    from huggingface_hub import hf_hub_download
    import os

    full_path = f"{MODELS_PATH}/{model_path}"

    # Check if file already exists
    if os.path.exists(full_path) and not force_download:
        print(f"✓ Model already cached: {model_path}")
        return full_path

    # Try to find model in registry
    model_info = None
    for key, info in MODEL_REGISTRY.items():
        if info["cache_path"] == model_path:
            model_info = info
            break

    if model_info is None:
        print(f"⚠ Model {model_path} not in registry - assuming user uploaded it")
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Model {model_path} not found in volume and not in auto-download registry. "
                f"Please upload manually: modal volume put musubi-models {model_path} /local/path/to/model"
            )
        return full_path

    # Download from HuggingFace
    print(f"📥 Downloading model from HuggingFace: {model_info['repo_id']}")
    print(f"   Files: {model_info['files']}")

    # Create directory if needed
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # Download each file
    for hf_file in model_info["files"]:
        print(f"   Downloading: {hf_file}")
        try:
            downloaded_path = hf_hub_download(
                repo_id=model_info["repo_id"],
                filename=hf_file,
                cache_dir="/tmp/hf_cache",
                resume_download=True,
            )

            # Copy to volume
            import shutil
            shutil.copy2(downloaded_path, full_path)
            print(f"   ✓ Saved to: {full_path}")

        except Exception as e:
            print(f"   ✗ Error downloading {hf_file}: {e}")
            raise

    # Commit to volume
    MODELS_VOLUME.commit()
    print(f"✓ Model downloaded and cached: {model_path}")

    return full_path

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
    """Cache latents for Wan training - auto-downloads models if needed"""

    # Ensure models are downloaded
    print("🔍 Checking if models are available...")
    vae_full = ensure_model_downloaded.local(vae_path)

    if clip_path:
        clip_full = ensure_model_downloaded.local(clip_path)
    else:
        clip_full = None

    cmd = [
        "python", f"{CODE_PATH}/src/musubi_tuner/wan_cache_latents.py",
        "--dataset_config", f"{DATA_PATH}/{dataset_config}",
        "--vae", vae_full,
        "--task", task,
    ]

    if i2v:
        cmd.append("--i2v")

    if clip_full:
        cmd.extend(["--clip", clip_full])

    if vae_cache_cpu:
        cmd.append("--vae_cache_cpu")

    result = run_command(cmd, "Caching latents for Wan")

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
    """Cache text encoder outputs for Wan training - auto-downloads T5 if needed"""

    print("🔍 Checking if T5 model is available...")
    t5_full = ensure_model_downloaded.local(t5_path)

    cmd = [
        "python", f"{CODE_PATH}/src/musubi_tuner/wan_cache_text_encoder_outputs.py",
        "--dataset_config", f"{DATA_PATH}/{dataset_config}",
        "--t5", t5_full,
        "--batch_size", str(batch_size),
    ]

    if fp8_t5:
        cmd.append("--fp8_t5")

    result = run_command(cmd, "Caching text encoder outputs for Wan")

    CACHE_VOLUME.commit()

    return {
        "status": "success" if result["returncode"] == 0 else "error",
        "returncode": result["returncode"],
    }


@app.function(
    gpu="A100-80GB",
    timeout=14400,
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
    Train Wan LoRA model - auto-downloads all models if needed
    """
    import os

    print("🔍 Checking if all models are available...")

    # Ensure all models are downloaded
    dit_full = ensure_model_downloaded.local(config['dit_path'])
    vae_full = ensure_model_downloaded.local(config['vae_path'])
    t5_full = ensure_model_downloaded.local(config['t5_path'])

    if config.get('clip_path'):
        clip_full = ensure_model_downloaded.local(config['clip_path'])
    else:
        clip_full = None

    if config.get('dit_high_noise_path'):
        dit_high_full = ensure_model_downloaded.local(config['dit_high_noise_path'])
    else:
        dit_high_full = None

    config_full = f"{DATA_PATH}/{config['dataset_config']}"
    output_dir = f"{MODELS_PATH}/outputs/{config['output_name']}"

    os.environ["ACCELERATE_CONFIG_FILE"] = "none"

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
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", str(config.get("max_data_loader_n_workers", 2)),
        "--persistent_data_loader_workers",
    ]

    # Attention mechanism: xformers or sdpa
    if config.get("use_xformers", False):
        cmd.append("--xformers")
    else:
        cmd.append("--sdpa")

    if clip_full:
        cmd.extend(["--clip", clip_full])

    if dit_high_full:
        cmd.extend(["--dit_high_noise", dit_high_full])

    # Timestep parameters (for dual LoRA training)
    if config.get("min_timestep") is not None:
        cmd.extend(["--min_timestep", str(config["min_timestep"])])

    if config.get("max_timestep") is not None:
        cmd.extend(["--max_timestep", str(config["max_timestep"])])

    if config.get("timestep_boundary") is not None:
        cmd.extend(["--timestep_boundary", str(config["timestep_boundary"])])

    if config.get("timestep_sampling"):
        cmd.extend(["--timestep_sampling", config["timestep_sampling"]])

    if config.get("discrete_flow_shift") is not None:
        cmd.extend(["--discrete_flow_shift", str(config["discrete_flow_shift"])])

    if config.get("preserve_distribution_shape"):
        cmd.append("--preserve_distribution_shape")

    # Optimizer parameters
    if config.get("gradient_accumulation_steps"):
        cmd.extend(["--gradient_accumulation_steps", str(config["gradient_accumulation_steps"])])

    if config.get("optimizer_args"):
        cmd.extend(["--optimizer_args", config["optimizer_args"]])

    if config.get("max_grad_norm") is not None:
        cmd.extend(["--max_grad_norm", str(config["max_grad_norm"])])

    # Learning rate scheduler parameters
    if config.get("lr_scheduler"):
        cmd.extend(["--lr_scheduler", config["lr_scheduler"]])

    if config.get("lr_scheduler_power"):
        cmd.extend(["--lr_scheduler_power", str(config["lr_scheduler_power"])])

    if config.get("lr_scheduler_min_lr_ratio"):
        cmd.extend(["--lr_scheduler_min_lr_ratio", str(config["lr_scheduler_min_lr_ratio"])])

    # Weighting scheme
    if config.get("weighting_scheme"):
        cmd.extend(["--weighting_scheme", config["weighting_scheme"]])

    # Metadata
    if config.get("metadata_title"):
        cmd.extend(["--metadata_title", config["metadata_title"]])

    if config.get("metadata_author"):
        cmd.extend(["--metadata_author", config["metadata_author"]])

    # Memory optimization
    if config.get("blocks_to_swap"):
        cmd.extend(["--blocks_to_swap", str(config["blocks_to_swap"])])

    if config.get("fp8_base"):
        cmd.append("--fp8_base")

    if config.get("fp8_llm"):
        cmd.append("--fp8_llm")

    # Other
    if config.get("seed"):
        cmd.extend(["--seed", str(config["seed"])])

    if config.get("additional_args"):
        cmd.extend(config["additional_args"])

    result = run_command(cmd, "Training Wan LoRA")

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
    """Cache latents for Qwen-Image training - auto-downloads VAE if needed"""

    print("🔍 Checking if VAE is available...")
    vae_full = ensure_model_downloaded.local(vae_path)

    cmd = [
        "python", f"{CODE_PATH}/src/musubi_tuner/qwen_image_cache_latents.py",
        "--dataset_config", f"{DATA_PATH}/{dataset_config}",
        "--vae", vae_full,
    ]

    if edit:
        cmd.append("--edit")
    elif edit_plus:
        cmd.append("--edit_plus")

    result = run_command(cmd, "Caching latents for Qwen-Image")

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
    """Cache text encoder outputs for Qwen-Image - auto-downloads Qwen2.5-VL if needed"""

    print("🔍 Checking if Qwen2.5-VL is available...")
    te_full = ensure_model_downloaded.local(text_encoder_path)

    cmd = [
        "python", f"{CODE_PATH}/src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py",
        "--dataset_config", f"{DATA_PATH}/{dataset_config}",
        "--text_encoder", te_full,
        "--batch_size", str(batch_size),
    ]

    if fp8_vl:
        cmd.append("--fp8_vl")
    if edit:
        cmd.append("--edit")
    elif edit_plus:
        cmd.append("--edit_plus")

    result = run_command(cmd, "Caching text encoder outputs for Qwen-Image")

    CACHE_VOLUME.commit()

    return {
        "status": "success" if result["returncode"] == 0 else "error",
        "returncode": result["returncode"],
    }


@app.function(
    gpu="A100-80GB",
    timeout=14400,
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
    Train Qwen-Image LoRA model - auto-downloads all models if needed
    """
    import os

    print("🔍 Checking if all models are available...")

    # Ensure all models are downloaded
    dit_full = ensure_model_downloaded.local(config['dit_path'])
    vae_full = ensure_model_downloaded.local(config['vae_path'])
    te_full = ensure_model_downloaded.local(config['text_encoder_path'])

    config_full = f"{DATA_PATH}/{config['dataset_config']}"
    output_dir = f"{MODELS_PATH}/outputs/{config['output_name']}"

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

    if config.get("edit"):
        cmd.append("--edit")
    elif config.get("edit_plus"):
        cmd.append("--edit_plus")

    if config.get("fp8_vl"):
        cmd.append("--fp8_vl")

    if config.get("seed"):
        cmd.extend(["--seed", str(config["seed"])])

    if config.get("additional_args"):
        cmd.extend(config["additional_args"])

    result = run_command(cmd, "Training Qwen-Image LoRA")

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
    Main entrypoint for Musubi-Tuner Modal pipeline with auto model download

    Args:
        model: "wan" or "qwen"
        config: Path to JSON config file
        stage: "all", "cache", or "train"

    Example:
        modal run run_modal.py --model wan --config my_config.json
    """

    # Load config
    with open(config, 'r') as f:
        cfg = json.load(f)

    print(f"\n{'='*70}")
    print(f"🚀 Musubi-Tuner Modal Pipeline (Auto-Download Enabled)")
    print(f"{'='*70}")
    print(f"Model:  {model}")
    print(f"Stage:  {stage}")
    print(f"Config: {config}")
    print(f"{'='*70}\n")
    print(f"ℹ️  Models will be auto-downloaded from HuggingFace on first use")
    print(f"ℹ️  Subsequent runs will use cached models (fast!)\n")

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
            result = wan_train.remote(config=cfg)
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
