"""
SPDX-License-Identifier: MIT
Copyright © 2026 dragos2001
"""
import json

file_path = "/mnt/e/workspace/iNatSounds/dataset/annotations/train.json"
data = []

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                pass  # skip lines that aren’t valid JSON

print(f"Loaded {len(data)} entries")
print(data[:5])
