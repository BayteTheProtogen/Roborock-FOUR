Żeby użyć SAM 3.1 stwórz venv, zainstaluj requirements, przejdź do odpowiedniego folderu, potem użyj komendy
```bash
python scripts/export_video_overlay.py <video_albo_katalog_obrazów> <katalog_output> --image-mode --preset <full, quality, fast> --device mps --sample-every <15> --restart-every <48> --prefer-metal --export-format yolo --text "sidewalk" --text "person" --text "animal" <--text cokolwiek> --batching <high, medium, low, none> --confidence <0.8> <--drop-empty>
```