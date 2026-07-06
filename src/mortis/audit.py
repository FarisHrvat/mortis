"""
MORTIS Audit Module
===================
Generates cryptographic receipts of hardware, environment, and parameters
to guarantee verifiable reproducibility.
"""

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import psutil


def generate_audit_receipt(
    analysis_name: str,
    params: Dict[str, Any],
    output_dir: str = "."
) -> str:
    """
    Creates a cryptographic JSON receipt logging hardware, OS, and parameters.
    """
    system_info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": platform.python_version(),
    }

    receipt = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "analysis": analysis_name,
        "hardware_environment": system_info,
        "parameters": params,
    }

    # Create a deterministic string representation for hashing
    receipt_str = json.dumps(receipt, sort_keys=True)
    receipt_hash = hashlib.sha256(receipt_str.encode("utf-8")).hexdigest()

    receipt["sha256_hash"] = receipt_hash

    out_path = Path(output_dir) / f"mortis_audit_{analysis_name}_{receipt_hash[:8]}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=4)

    print(f"[MORTIS] Audit receipt generated: {out_path.name}")
    return str(out_path)
