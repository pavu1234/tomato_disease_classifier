import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps
from .config_loader import path_for
from .utils import sha256, files_under, checked_path

COLUMNS = [
    "image_id",
    "relative_path",
    "filename",
    "class_label",
    "file_extension",
    "file_size_bytes",
    "image_width",
    "image_height",
    "image_mode",
    "is_valid_image",
    "sha256_hash",
    "perceptual_hash",
    "source_group",
    "source_type",
    "plant_id",
    "capture_session",
    "capture_date",
    "camera_or_source",
    "notes",
    "audit_flags",
]
MANUAL = [
    "source_group",
    "source_type",
    "plant_id",
    "capture_session",
    "capture_date",
    "camera_or_source",
    "notes",
]


def imagehash_module():
    try:
        import imagehash
    except ImportError as e:
        raise RuntimeError(
            "Install hashing support: pip install ImageHash Pillow"
        ) from e
    return imagehash


def inspect_file(p, c):
    ih = imagehash_module()
    flags = []
    result = dict(
        file_extension=p.suffix.lower(),
        file_size_bytes=p.stat().st_size,
        image_width=0,
        image_height=0,
        image_mode="",
        is_valid_image=False,
        sha256_hash=sha256(p),
        perceptual_hash="",
    )
    if p.is_symlink():
        flags.append("symlink_rejected")
    if p.suffix.lower() not in c["image"]["supported_extensions"]:
        flags.append("unsupported_extension")
    if p.stat().st_size == 0:
        flags.append("zero_byte")
    try:
        with Image.open(p) as im:
            im.load()
            result.update(
                image_width=im.width, image_height=im.height, image_mode=im.mode
            )
            if min(im.size) < c["image"].get("minimum_dimension", 32):
                flags.append("extremely_small")
            if im.mode != "RGB":
                flags.append("convert_to_rgb")
            if getattr(im, "n_frames", 1) > 1:
                flags.append("animated_image_rejected")
            result["perceptual_hash"] = str(
                ih.phash(ImageOps.exif_transpose(im).convert("RGB"))
            )
    except (OSError, ValueError, Image.DecompressionBombError):
        flags.append("corrupt_or_unsafe_image")
    result["is_valid_image"] = not any(
        f in flags
        for f in [
            "symlink_rejected",
            "unsupported_extension",
            "zero_byte",
            "extremely_small",
            "corrupt_or_unsafe_image",
            "animated_image_rejected",
        ]
    )
    result["audit_flags"] = ";".join(flags)
    return result


def build_manifest(c, save=True):
    root = path_for(c, "raw_data_dir")
    out = path_for(c, "manifest_path")
    old = (
        pd.read_csv(out, keep_default_na=False)
        .set_index("relative_path")
        .to_dict("index")
        if out.exists() and out.stat().st_size
        else {}
    )
    rows = []
    for p in files_under(root):
        rel = p.relative_to(root)
        label = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        row = {k: "" for k in COLUMNS}
        row.update(inspect_file(p, c))
        uid = hashlib.sha256(rel.as_posix().encode()).hexdigest()[:24]
        row.update(
            image_id=uid,
            relative_path=rel.as_posix(),
            filename=p.name,
            class_label=label,
        )
        # Same nested folder token in different class folders maps to one group.
        row["source_group"] = (
            "inferred:" + "/".join(rel.parts[1:-1])
            if len(rel.parts) > 2
            else "fallback:" + uid
        )
        row["source_type"] = (
            "inferred_unverified" if len(rel.parts) > 2 else "fallback_group"
        )
        row["notes"] = (
            "Verify source independence before setting source_type=verified_group"
        )
        for k in MANUAL:
            if rel.as_posix() in old:
                row[k] = old[rel.as_posix()].get(k, row[k])
        if label not in c["classes"]["expected"]:
            row["is_valid_image"] = False
            row["audit_flags"] += ";unknown_class"
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["is_valid_image"] = df["is_valid_image"].astype(bool)
    if save:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def read_manifest(c, verify=True):
    p = path_for(c, "manifest_path")
    if not p.exists():
        raise ValueError("Create the manifest first")
    df = pd.read_csv(p, keep_default_na=False)
    if not set(COLUMNS).issubset(df.columns):
        raise ValueError("Manifest columns missing; regenerate")
    if df.empty:
        raise ValueError(
            "No real images supplied. Populate data/raw and create the manifest"
        )
    if df.relative_path.duplicated().any() or df.image_id.duplicated().any():
        raise ValueError("Duplicate manifest rows")
    df["is_valid_image"] = df.is_valid_image.astype(str).str.lower().eq("true")
    if verify:
        fresh = build_manifest(c, save=False).set_index("relative_path")
        if set(fresh.index) != set(df.relative_path):
            raise ValueError("Raw inventory changed; regenerate manifest")
        fields = ["sha256_hash", "perceptual_hash", "class_label", "is_valid_image"]
        for row in df.to_dict("records"):
            new = fresh.loc[row["relative_path"]]
            if any(str(new[k]) != str(row[k]) for k in fields):
                raise ValueError(
                    "Raw image/derived metadata changed: "
                    + row["relative_path"]
                    + "; regenerate manifest"
                )
    return df
