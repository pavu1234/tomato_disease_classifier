"""Import the requested three RGB classes, preserving upstream leaf IDs when known.

No fallback is promoted to a verified group. No source image is modified.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config_loader import load_config, path_for
from src.manifest import build_manifest
from src.utils import sha256, write_json, now

CLASS_MAP = {
    "Tomato___Early_blight": "early_blight",
    "Tomato___healthy": "healthy",
    "Tomato___Late_blight": "late_blight",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    c = load_config()
    raw = path_for(c, "raw_data_dir")
    if any(p.is_file() and p.name != ".gitkeep" for p in raw.rglob("*")):
        parser.error("Raw data already exists. Refusing to overwrite or mix sources.")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    map_path = source / "leaf_grouping/leaf-map.json"
    leaf_map = json.loads(map_path.read_text())
    records = {}
    for upstream, label in CLASS_MAP.items():
        paths = sorted((source / "raw/color" / upstream).glob("*"))
        if not paths:
            parser.error(f"No original color images for {upstream}; sparse-checkout that directory")
        for p in paths:
            if not p.is_file():
                continue
            dest = raw / label / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            # Match the repository loader's image identifier normalization.
            identifier = p.name.replace("_final_masked", "").split("___")[-1]
            identifier = identifier.split("copy")[0]
            for extension in [".jpg", ".JPG", ".png", ".PNG"]:
                identifier = identifier.replace(extension, "")
            key = identifier.lower().strip()
            candidates = leaf_map.get(key, [])
            candidates = [x for x in candidates if x.startswith(upstream + ":::")]
            resolved = candidates[0] if len(candidates) == 1 else None
            records[dest.relative_to(raw).as_posix()] = dict(
                source_path=p.relative_to(source).as_posix(),
                source_sha256=sha256(p),
                lookup_key=key,
                leaf_id=resolved,
                candidates=candidates,
            )
    df = build_manifest(c)
    for index, row in df.iterrows():
        record = records[row.relative_path]
        df.at[index, "camera_or_source"] = "PlantVillage original color; " + commit
        if record["leaf_id"]:
            df.at[index, "source_group"] = "PlantVillage:" + record["leaf_id"]
            df.at[index, "plant_id"] = ""  # Leaf ID is NOT a verified plant ID.
            df.at[index, "source_type"] = "verified_group"
            df.at[index, "notes"] = (
                "Leaf grouping resolved from upstream leaf-map.json; leaf ID="
                + record["leaf_id"]
                + ". Does not establish independent plants, farms, or capture sessions."
            )
        else:
            df.at[index, "notes"] = (
                "No unambiguous class-matching leaf ID in upstream map. "
                "Fallback group is unverified; do not claim leaf/plant independence."
            )
    df.to_csv(path_for(c, "manifest_path"), index=False)
    summary = {
        "repository": "https://github.com/spMohanty/PlantVillage-Dataset",
        "commit": commit,
        "retrieved_at": now(),
        "variant": "original RGB/color only; no grayscale or segmented duplicates",
        "leaf_map_sha256": sha256(map_path),
        "class_mapping": CLASS_MAP,
        "class_counts": {label: int((df.class_label == label).sum()) for label in CLASS_MAP.values()},
        "resolved_leaf_metadata_counts": {
            label: int(((df.class_label == label) & (df.source_type == "verified_group")).sum())
            for label in CLASS_MAP.values()
        },
        "unresolved_count": int((df.source_type != "verified_group").sum()),
        "scope": "Upstream leaf identities only, not verified plant/farm/session identities.",
        "images": records,
    }
    write_json(path_for(c, "manifest_path").parent / "plantvillage_provenance.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "images"}, indent=2))


if __name__ == "__main__":
    main()
