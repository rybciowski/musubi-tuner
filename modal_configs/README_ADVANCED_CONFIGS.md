# Advanced WAN 2.2 Training Configs

This folder contains example JSON configs for advanced WAN 2.2 LoRA training on Modal.com.

## 📁 Available Configs

### 1. **Dual LoRA Training** (High-Noise + Low-Noise)

**Files:**
- `wan22_high_noise_dual_lora.json` - Train on timesteps 875-1000 (structure/composition)
- `wan22_low_noise_dual_lora.json` - Train on timesteps 0-875 (details/refinement)

**Usage:**
```bash
# Train HIGH-noise LoRA first
modal run run_modal_v2.py --model wan --config modal_configs/wan22_high_noise_dual_lora.json

# Train LOW-noise LoRA second (same dataset!)
modal run run_modal_v2.py --model wan --config modal_configs/wan22_low_noise_dual_lora.json

# Use BOTH LoRAs together during inference for best quality!
```

**Why Dual LoRA?**
- **High-noise (875-1000)**: Controls overall structure, composition, motion patterns
- **Low-noise (0-875)**: Controls fine details, textures, sharpness
- **Together**: Better quality than single LoRA alone

**Cost:** ~$8 × 2 = $16 total (train both)

---

### 2. **Vankoala Style** (Single LoRA, Long Training)

**File:** `wan22_vankoala_style.json`

**Usage:**
```bash
modal run run_modal_v2.py --model wan --config modal_configs/wan22_vankoala_style.json
```

**Features:**
- Single LoRA (simpler workflow)
- Higher network_dim (32 vs 16) = more capacity
- 500 epochs (longer training)
- Saves every 50 epochs (monitor progress)
- Good for character/style LoRAs

**Cost:** ~$25-30 (500 epochs, A100-80GB)

---

## 🎛️ Parameter Guide

### **Timestep Parameters** (Dual LoRA)

```json
"min_timestep": 875,    // Start of noise range
"max_timestep": 1000,   // End of noise range
"timestep_sampling": "shift",
"discrete_flow_shift": 1.0,
"preserve_distribution_shape": true
```

**Presets:**
- High-noise: `min=875, max=1000`
- Low-noise: `min=0, max=875`
- Full range: omit both (default)

---

### **Learning Rate Scheduler**

```json
"lr_scheduler": "polynomial",
"lr_scheduler_power": 8,           // Higher = steeper decay
"lr_scheduler_min_lr_ratio": "5e-5" // Min LR at end
```

**Options:**
- `polynomial` - Smooth decay (recommended for long training)
- `constant` - No decay
- `cosine` - Cosine annealing

---

### **Optimizer Settings**

```json
"optimizer_type": "adamw",           // Full AdamW
"optimizer_args": "weight_decay=0.1",
"max_grad_norm": 0,                  // 0 = no clipping
"gradient_accumulation_steps": 1
```

**Optimizer Types:**
- `adamw` - Best quality, high VRAM
- `adamw8bit` - Lower VRAM, slightly lower quality
- `adafactor` - Lowest VRAM

---

### **Network Dimensions**

```json
"network_dim": 32,      // LoRA rank (higher = more capacity)
"network_alpha": 32     // Scaling factor (usually = dim)
```

**Recommendations:**
- **Simple style:** dim=16, alpha=16
- **Character/complex:** dim=32, alpha=32
- **Very detailed:** dim=64, alpha=64 (longer training)

---

### **Memory Optimization**

```json
"fp8_base": true,              // Quantize base model to FP8
"use_xformers": true,          // Use xformers attention
"blocks_to_swap": 8,           // Offload blocks to RAM
"mixed_precision": "fp16"      // fp16 or bf16
```

**VRAM Usage:**
- `fp8_base=true, blocks_to_swap=8` → ~60GB VRAM (A100-80GB)
- `fp8_base=false` → ~80GB+ VRAM (won't fit!)

---

### **Training Duration**

```json
"max_train_epochs": 100,
"save_every_n_epochs": 100
```

**Recommendations:**
- **Quick test:** 10 epochs (~30 min)
- **Dual LoRA:** 100 epochs (~2.5h each)
- **Long training:** 500+ epochs (~12h+)

---

## 🚀 Quick Start

### Step 1: Prepare Dataset

```toml
# datasets/wan_dataset/config.toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]
image_directory = "images"
cache_directory = "cache"
num_repeats = 10
```

### Step 2: Upload to Modal

```bash
modal volume put musubi-data datasets/wan_dataset wan_dataset
```

### Step 3: Choose Config & Train

**For Dual LoRA:**
```bash
# Copy and customize
cp modal_configs/wan22_high_noise_dual_lora.json my_high.json
cp modal_configs/wan22_low_noise_dual_lora.json my_low.json

# Edit: change output_name, metadata_author, etc.
notepad my_high.json
notepad my_low.json

# Train both
modal run run_modal_v2.py --model wan --config my_high.json
modal run run_modal_v2.py --model wan --config my_low.json
```

**For Single LoRA:**
```bash
cp modal_configs/wan22_vankoala_style.json my_config.json
notepad my_config.json
modal run run_modal_v2.py --model wan --config my_config.json
```

### Step 4: Download Results

```bash
modal volume get musubi-models outputs/my_character_high_noise .
modal volume get musubi-models outputs/my_character_low_noise .
```

---

## 💡 Tips

1. **Dual LoRA is Worth It:** ~2x cost but noticeably better quality
2. **Monitor Progress:** Use `save_every_n_epochs` to check intermediate results
3. **Learning Rate:** Start with `3e-4`, use `2e-4` for more stable/refined training
4. **Batch Size:** Increase if you have VRAM headroom (smoother gradients)
5. **num_repeats in TOML:** Increase if you have few images (< 20)

---

## 📊 Cost Estimates

| Config | Epochs | Time | Cost |
|--------|--------|------|------|
| Dual LoRA (both) | 100 each | ~5h total | $16 |
| Vankoala Style | 500 | ~12h | $28 |
| Quick Test | 10 | ~30 min | $1.50 |

*Based on A100-80GB ($2.33/hr on Modal)*

---

## ❓ Troubleshooting

**Q: "CUDA out of memory"**
- Set `fp8_base: true`
- Increase `blocks_to_swap` to 12-16
- Lower `batch_size` in TOML to 1

**Q: "Loss not decreasing"**
- Lower learning rate (`2e-4` or `1e-4`)
- Check your captions (quality matters!)
- Increase `num_repeats` if dataset is too small

**Q: "LoRA has no effect"**
- Check `network_alpha` matches `network_dim`
- Try higher weight during inference (1.0-1.5)
- Train longer (more epochs)

---

Happy training! 🚀
