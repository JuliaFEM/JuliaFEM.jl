#!/usr/bin/env python3
"""
Fix Element type signatures in vendor packages
Converts Element{BasisType} → Element{M, BasisType} where M
"""
import re
import sys
from pathlib import Path

# Element types to fix
ELEMENT_TYPES = [
    "Seg2",
    "Seg3",
    "Poi1",
    "Tri3",
    "Tri6",
    "Quad4",
    "Quad8",
    "Quad9",
    "Tet4",
    "Tet10",
    "Pyr5",
    "Wedge6",
    "Wedge15",
    "Hex8",
    "Hex20",
    "Hex27",
]


def fix_element_types(filepath):
    """Fix Element type signatures in a single file."""
    with open(filepath, "r") as f:
        content = f.read()

    original = content

    # Pattern 1: Element{Type} in function parameters/returns
    for etype in ELEMENT_TYPES:
        # Fix ::Element{Type}
        content = re.sub(
            rf"::Element\{{{etype}\}}", rf"::Element{{M, {etype}}}", content
        )

        # Fix Vector{Element{Type}}
        content = re.sub(
            rf"::Vector\{{Element\{{M, {etype}\}}\}}",
            rf"::Vector{{Element{{M, {etype}}}}}",
            content,
        )

    # Pattern 2: Add "where M" to function signatures
    lines = content.split("\n")
    new_lines = []

    for i, line in enumerate(lines):
        # Skip if already has where M
        if "where M" in line or "where {M" in line or "where E" in line:
            new_lines.append(line)
            continue

        # Check if this is a function signature with Element{M,
        if re.search(r"function\s+.*Element\{M,", line):
            # Check if function signature closes on this line
            if ")" in line and "end" not in line:
                # Add where M before any trailing comment
                line = line.rstrip()
                if not line.endswith(" where M"):
                    # Handle inline functions (one-liners)
                    if "=" in line and line.count(")") == line.count("("):
                        # function foo(...) = expr
                        line = re.sub(r"\)(\s*=)", r") where M\1", line)
                    else:
                        line = line + " where M"

        new_lines.append(line)

    content = "\n".join(new_lines)

    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        return True
    return False


def main():
    vendor_dir = Path("/home/juajukka/dev/JuliaFEM.jl/vendor")
    fixed_count = 0
    file_count = 0

    print("Fixing Element type signatures in vendor packages...")
    print("=" * 60)

    # Find all .jl files in vendor subdirectories
    for package_dir in vendor_dir.iterdir():
        if not package_dir.is_dir():
            continue

        src_dir = package_dir / "src"
        if not src_dir.exists():
            continue

        for jl_file in src_dir.rglob("*.jl"):
            file_count += 1
            if fix_element_types(jl_file):
                print(f"✓ Fixed: {jl_file.relative_to(vendor_dir)}")
                fixed_count += 1

    print("=" * 60)
    print(f"Processed {file_count} files")
    print(f"Fixed {fixed_count} files")

    if fixed_count > 0:
        print("\n⚠️  IMPORTANT: Review changes before committing!")
        print("Run: git diff vendor/")


if __name__ == "__main__":
    main()
