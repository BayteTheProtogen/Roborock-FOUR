Do użycia video.py należy najpierw połączyć urządzenie z kamerą telefonu poprzez Droidcam. Następnie sprawdzić adres kamery.

```bash
pip install -r requirements.txt

#run the file with basic example file
python main.py

# Or with your own video (its soundtrack drives the hearing pathway too):
python main.py --video path/to/your_clip.mp4

# Or your own camera

python main.py --camera-device /dev/video5
```

