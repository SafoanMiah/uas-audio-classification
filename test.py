from pathlib import Path
import config as cfg

for name, rel in [
    ("svanstrom", "svanstrom/audio"),
    ("al_emadi", "al_emadi/Binary_Drone_Audio"),
    ("esc50", "esc50/meta"),
]:
    p = cfg.DATA_ROOT / rel
    print(f"{name}: {p} exists={p.exists()}")
    if p.exists():
        contents = list(p.iterdir())[:5]
        print(f"  first 5: {[x.name for x in contents]}")
