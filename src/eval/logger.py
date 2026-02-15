from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional

def save_run_log(
    filename_prefix: str,
    content: str,
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"runs/{filename_prefix}_{ts}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
