# Modal Config Examples

Przykładowe konfiguracje JSON dla treningu na Modal.com.

## Użycie

1. Skopiuj przykładowy config:
   ```bash
   cp wan_t2v_example.json my_config.json
   ```

2. Edytuj według potrzeb:
   ```bash
   nano my_config.json
   ```

3. Uruchom:
   ```bash
   modal run run_modal.py --model wan --config modal_configs/my_config.json
   ```

## Dostępne Przykłady

- `wan_t2v_example.json` - Wan 2.1 Text-to-Video
- `wan22_example.json` - Wan 2.2 (dual DiT)
- `qwen_image_example.json` - Qwen-Image Text-to-Image
- `qwen_image_edit_example.json` - Qwen-Image-Edit

## Uwaga

Pamiętaj dostosować:
- `dataset_config` - ścieżka do Twojego TOML w volume
- `output_name` - unikalna nazwa dla wytrenowanego modelu
- Ścieżki do modeli (`dit_path`, `vae_path`, etc.)
- Parametry treningowe według potrzeb
