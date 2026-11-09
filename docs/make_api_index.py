import os
from collections import defaultdict
from pathlib import Path

# Modules to exclude from the API index
EXCLUDE_MODULES = {"__init__"}

# Set the current working directory to the directory of this script
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)


def make_api_index():
    """
    Generate a properly formatted `api_index.rst` file for
    Sphinx documentation, grouped by submodule and excluding test modules.
    """

    api_path = Path("../src/photon_mosaic")

    # submodule -> list of module names
    grouped_modules = defaultdict(list)

    for path in sorted(api_path.rglob("*.py")):
        rel_path = path.relative_to(api_path.parent)

        if rel_path.stem in EXCLUDE_MODULES:
            continue

        # Skip test modules/packages anywhere in the path
        if "tests" in rel_path.parts:
            continue

        # Skip hidden/cache directories (e.g. .ipynb_checkpoints, __pycache__)
        if any(part.startswith(".") or part == "__pycache__" for part in rel_path.parts):
            continue

        module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")

        # rel_path.parts[0] == "photon_mosaic"; parts[1] is the submodule
        # (e.g. "core", "extractors"). Modules directly under the package
        # root fall back to a "misc" group.
        submodule = rel_path.parts[1] if len(rel_path.parts) > 2 else "misc"

        grouped_modules[submodule].append(module_name)

    # Directory that will hold one .rst file per submodule
    api_dir = Path("source") / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    # Write one .rst file per submodule
    for submodule in sorted(grouped_modules):
        title = submodule.replace("_", " ").title()
        lines = [
            title,
            "=" * len(title),
            "",
            ".. autosummary::",
            "   :toctree: .",
            "   :nosignatures:",
            "",
        ]
        lines += [f"   {module}" for module in grouped_modules[submodule]]
        lines.append("")

        submodule_path = api_dir / f"{submodule}.rst"
        submodule_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Generated {submodule_path}")

    # Write the top-level index that links to each submodule page
    index_lines = [
        ".. _target-api:",
        "",
        "API Reference",
        "=============",
        "",
        "This section contains automatically generated documentation for the",
        "`photon-mosaic` package.",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]
    index_lines += [f"   api/{submodule}" for submodule in sorted(grouped_modules)]
    index_lines.append("")

    output_path = Path("source") / "api_index.rst"
    output_path.write_text("\n".join(index_lines), encoding="utf-8")

    print(f"Generated {output_path}")


if __name__ == "__main__":
    make_api_index()
