#!/usr/bin/env python3
"""
Patch aiter library to support gfx1201 (RDNA 4) GPU architecture.

This script modifies the aiter chip_info.py file to:
1. Add gfx1201 to the GFX_MAP dictionary
2. Replace the exception with a fallback for unsupported architectures
"""

import os
import re
import sys


def patch_aiter_for_gfx1201():
    """Patch aiter library to support gfx1201."""
    path = "/usr/local/lib/python3.12/dist-packages/aiter/jit/utils/chip_info.py"
    
    if not os.path.exists(path):
        print("⚠️  aiter chip_info.py not found, skipping patch")
        return False
    
    try:
        # Read the file
        with open(path, "r") as f:
            content = f.read()
        
        # Add gfx1201 to GFX_MAP if not present
        if "gfx1201" not in content:
            content = re.sub(
                r"(GFX_MAP = {.*?)}",
                r'\1    16: "gfx1201",\n}',
                content,
                flags=re.DOTALL
            )
        
        # Replace exception with fallback
        content = content.replace(
            'raise RuntimeError("Unsupported gfx")',
            "return gfx"
        )
        
        # Write back
        with open(path, "w") as f:
            f.write(content)
        
        print("✅ Patched aiter with gfx1201 support")
        return True
        
    except Exception as e:
        print(f"❌ Error patching aiter: {e}")
        return False


if __name__ == "__main__":
    success = patch_aiter_for_gfx1201()
    sys.exit(0 if success else 1)
