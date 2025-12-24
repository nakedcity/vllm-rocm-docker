#!/usr/bin/env python3
"""
Patch aiter library to support gfx1201 (RDNA 4) GPU architecture.

This script modifies the aiter chip_info.py file to:
1. Add gfx1201 to the GFX_MAP dictionary
2. Update get_device_name to support gfx1201 (mapping to MI300 as fallback/proxy or returning generic)
"""

import os
import re
import sys


def patch_aiter_for_gfx1201():
    """Patch aiter library to support gfx1201."""
    # Target file in the container
    path = "/usr/local/lib/python3.12/dist-packages/aiter/jit/utils/chip_info.py"

    if not os.path.exists(path):
        print(f"⚠️  aiter chip_info.py not found at {path}")
        return False

    try:
        print(f"🔧 Reading {path}...")
        with open(path, "r") as f:
            content = f.read()

        modified = False

        # 1. Add gfx1201 to GFX_MAP if not present
        if '"gfx1201"' not in content and "'gfx1201'" not in content:
            print("🔧 Adding gfx1201 to GFX_MAP...")
            # Pattern to match the end of GFX_MAP dictionary
            # Matches the last element and the closing brace
            pattern = r"(GFX_MAP\s*=\s*\{.*?)(\})"
            replacement = r'\1    16: "gfx1201",\n\2'

            new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

            if new_content != content:
                content = new_content
                modified = True
                print("   ✅ Added gfx1201 to GFX_MAP")
            else:
                print("   ⚠️  Failed to match GFX_MAP structure")
        else:
            print("   ℹ️  gfx1201 already in GFX_MAP")

        # 2. Patch get_device_name to avoid RuntimeError
        # We'll make it return "MI300" for gfx1201 as a best-effort fallback if that helps AITemplate
        # or just return the arch string.
        if 'elif gfx == "gfx1201":' not in content:
            print("🔧 Patching get_device_name for gfx1201...")
            # Look for the last elif or if and insert our check
            # The block usually ends with else: raise RuntimeError

            # Fallback: simple replace of the error raising block if we can't be smarter
            # But let's try to inject properly.

            target_str = 'elif gfx == "gfx950":\n        return "MI350"'
            injection = 'elif gfx == "gfx950":\n        return "MI350"\n    elif gfx == "gfx1201":\n        return "MI300"  # Proxy for RDNA4 compatibility'

            if target_str in content:
                content = content.replace(target_str, injection)
                modified = True
                print("   ✅ Patched get_device_name logic")
            else:
                # Fallback: simpler replace of the exception
                print(
                    "   ⚠️  Could not find specific insertion point in get_device_name, trying fallback..."
                )
                err_str = 'raise RuntimeError("Unsupported gfx")'
                fallback_str = 'return "MI300" # Fallback for gfx1201'
                if err_str in content:
                    content = content.replace(err_str, fallback_str)
                    modified = True
                    print("   ✅ Patched RuntimeError fallback")

        if modified:
            print(f"💾 Saving changes to {path}...")
            with open(path, "w") as f:
                f.write(content)
            print("✅ Successfully patched aiter for gfx1201")
            return True
        else:
            print("ℹ️  No changes needed (file might be already patched)")
            return True

    except Exception as e:
        print(f"❌ Error patching aiter: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    if patch_aiter_for_gfx1201():
        sys.exit(0)
    else:
        sys.exit(1)
