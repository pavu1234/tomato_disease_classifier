"""Dataset assignment and cryptographic holdout integrity.
Hashes are integrity checks, not authentication against a malicious filesystem owner.
"""

import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from .config_loader import path_for
from .manifest import read_manifest
from .duplicate_detection import find_duplicates, components
from .reporting import save_report
from .utils import (
    sha256,
    digest,
    now,
    write_json,
    read_json,
    checked_path,
    files_under,
    require_not_exposed,
    LOG,
)

FALLBACK_WARNING = "WARNING: Source-group metadata is not verified. This split prevents exact/near-duplicate overlap where detected, but cannot guarantee independence by plant, field, or capture session. Final test performance should be called preliminary internal performance."
SMALL_WARNING = "WARNING: The locked test set is smaller than 20 images per class. Final metrics will be statistically unstable and must be presented as preliminary."
SPLITS = ["train", "val", "test_locked"]
KEYS = {"train": "train_dir", "val": "val_dir", "test_locked": "locked_test_dir"}


def assign_groups(df, c, seed, minimum_test):
    names = c["classes"]["expected"]
    groups = sorted(df.component.unique())
    counts = np.array(
        [
            [((df.component == g) & (df.class_label == name)).sum() for name in names]
            for g in groups
        ],
        dtype=float,
    )
    if len(groups) < 3 or any(
        np.count_nonzero(counts[:, j]) < 3 for j in range(len(names))
    ):
        raise ValueError(
            "At least three independent connected groups per class are required"
        )
    target = np.array(
        [c["split"]["train_ratio"], c["split"]["val_ratio"], c["split"]["test_ratio"]]
    )
    total = counts.sum(axis=0)
    rng = np.random.default_rng(seed)
    best = None
    # Search ONLY metadata/counts, never pixels, model predictions, or difficulty.
    for _ in range(4000):
        assignment = rng.choice(3, len(groups), p=target)
        n = np.array([counts[assignment == s].sum(axis=0) for s in range(3)])
        if (n < 1).any() or (n[2] < minimum_test).any():
            continue
        score = float(np.square(n / total - target[:, None]).sum())
        if best is None or score < best[0]:
            best = (score, assignment.copy())
    if best is None:
        raise ValueError(
            "No feasible group-preserving split found in 4000 seeded candidates. Add independent sources or inspect group sizes; do not break groups. Use --allow-small-test-set only if justified."
        )
    return {g: SPLITS[int(best[1][i])] for i, g in enumerate(groups)}


def check_leakage(df, exact, near):
    source_overlap = df.groupby("source_group").split.nunique().gt(1).sum()
    hash_overlap = df.groupby("sha256_hash").split.nunique().gt(1).sum()
    where = df.set_index("image_id").split.to_dict()
    cross = lambda pairs: sum(
        where[a] != where[b] for a, b in zip(pairs.image_id_a, pairs.image_id_b)
    )
    e, n = cross(exact), cross(near)
    return dict(
        passed=bool(not (source_overlap or hash_overlap or e or n)),
        shared_source_groups=int(source_overlap),
        exact_hash_overlap=int(hash_overlap),
        exact_duplicate_pairs_across_splits=int(e),
        near_duplicate_pairs_across_splits=int(n),
    )


def split_dataset(
    c,
    dry_run=False,
    force_rebuild=False,
    allow_fallback_groups=False,
    allow_small_test_set=False,
    seed=None,
):
    require_not_exposed(c)
    if (path_for(c, "models_dir") / "frozen_model.json").exists():
        raise ValueError(
            "A model was already selected. Use a new experiment directory; do not replace its split."
        )
    seed = c["seed"] if seed is None else seed
    if (
        not dry_run
        and not force_rebuild
        and (
            path_for(c, "test_lock_manifest").exists()
            or any(files_under(path_for(c, k)) for k in KEYS.values())
        )
    ):
        raise ValueError(
            "Processed data already exists. Use --force-rebuild before training only"
        )
    df = read_manifest(c)
    df = df[df.is_valid_image].copy()
    if set(df.class_label) != set(c["classes"]["expected"]):
        raise ValueError("A class is missing valid images")
    if df.source_group.eq("").any():
        raise ValueError("Missing source_group; regenerate or fill metadata")
    unverified = (df.source_type != "verified_group") | df.source_group.str.startswith(
        ("fallback:", "inferred:")
    )
    if unverified.any() and not allow_fallback_groups:
        raise ValueError(
            FALLBACK_WARNING
            + " Pass --allow-fallback-groups or verify metadata and replace inferred/fallback IDs."
        )
    if unverified.any():
        LOG.warning(FALLBACK_WARNING)
    r = path_for(c, "reports_dir")
    r.mkdir(parents=True, exist_ok=True)
    exact, near = find_duplicates(df, c["split"]["duplicate_hamming_threshold"])
    conflicts = pd.concat([exact, near], ignore_index=True)
    conflicts = conflicts[conflicts.label_conflict.astype(bool)]
    conflicts.to_csv(r / "label_conflict_candidates.csv", index=False)
    near.to_csv(r / "near_duplicates.csv", index=False)
    exact.to_csv(r / "exact_duplicates.csv", index=False)
    if len(conflicts):
        raise ValueError(
            "Unresolved label-conflict candidates. Review reports/label_conflict_candidates.csv; split refused. Correct labels or curate an external copy manually and regenerate manifest."
        )
    if len(near):
        LOG.warning(
            "Near-duplicate candidates require manual review; all detected connected clusters stay in ONE split."
        )
    df["component"] = df.image_id.map(components(df, exact, near))
    minimum = 1 if allow_small_test_set else c["split"]["minimum_test_images_per_class"]
    mapping = assign_groups(df, c, seed, minimum)
    df["split"] = df.component.map(mapping)
    # Stable unique PNG paths; copy-normalize format, mode and EXIF without changing raw files.
    df["processed_path"] = [
        (Path(c["paths"][KEYS[s]]) / cl / (i + ".png")).as_posix()
        for s, cl, i in zip(df.split, df.class_label, df.image_id)
    ]
    leak = check_leakage(df, exact, near)
    if not leak["passed"]:
        raise ValueError("Split leakage checks failed")
    counts = {
        s: {
            cl: int(((df.split == s) & (df.class_label == cl)).sum())
            for cl in c["classes"]["expected"]
        }
        for s in SPLITS
    }
    small = min(counts["test_locked"].values()) < 20
    if small:
        LOG.warning(SMALL_WARNING)
    summary = dict(
        seed=int(seed),
        class_counts=counts,
        group_counts={s: int(df[df.split == s].source_group.nunique()) for s in SPLITS},
        source_groups={
            s: sorted(df[df.split == s].source_group.unique()) for s in SPLITS
        },
        fallback_group_count=int(
            df[df.source_type == "fallback_group"].source_group.nunique()
        ),
        unverified_group_count=int(df[unverified].source_group.nunique()),
        preliminary=bool(unverified.any() or small),
        dry_run=dry_run,
        passed=True,
        deviation_from_target={
            s: {
                cl: counts[s][cl] / int((df.class_label == cl).sum())
                - c["split"][ratio]
                for cl in c["classes"]["expected"]
            }
            for s, ratio in zip(SPLITS, ["train_ratio", "val_ratio", "test_ratio"])
        },
        limitations="Groups are user-declared; pHash misses some duplicates and can over-group unrelated images. Duplicates within a split reduce effective sample size.",
        **{k: v for k, v in leak.items() if k != "passed"}
    )
    if dry_run:
        # Never overwrite authoritative records of an existing locked dataset.
        df.to_csv(r / "split_manifest_dry_run.csv", index=False)
        save_report(r, "split_summary_dry_run", summary)
        save_report(r, "split_leakage_check_dry_run", leak)
        return summary
    from PIL import Image, ImageOps

    with tempfile.TemporaryDirectory(prefix="tomato_split_", dir=c["root"]) as tmp:
        tmp = Path(tmp)
        hashes = []
        for row in df.to_dict("records"):
            src = checked_path(path_for(c, "raw_data_dir"), row["relative_path"])
            if sha256(src) != row["sha256_hash"]:
                raise ValueError("Raw file changed while splitting")
            dst = tmp / row["processed_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as im:
                ImageOps.exif_transpose(im).convert("RGB").save(dst, format="PNG")
            hashes.append(sha256(dst))
        df["processed_sha256"] = hashes
        if df.groupby("processed_sha256").split.nunique().gt(1).any():
            raise ValueError("Normalized pixel duplicates crossed splits")
        # Commit only after all images have converted and checks passed.
        lockpath = path_for(c, "test_lock_manifest")
        if lockpath.exists():
            lockpath.unlink()  # Interrupted commit fails closed (no lock).
        for s in SPLITS:
            dst = path_for(c, KEYS[s])
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(tmp / c["paths"][KEYS[s]]), str(dst))
    df.to_csv(r / "split_manifest.csv", index=False)
    save_report(r, "split_summary", summary)
    save_report(r, "split_leakage_check", leak)
    records = df[
        [
            "image_id",
            "processed_path",
            "processed_sha256",
            "sha256_hash",
            "class_label",
            "source_group",
            "split",
        ]
    ].to_dict("records")
    payload = dict(
        timestamp=now(),
        seed=int(seed),
        counts=counts,
        preliminary=summary["preliminary"],
        test_images=[x for x in records if x["split"] == "test_locked"],
        all_assignments=records,
        split_manifest_sha256=sha256(r / "split_manifest.csv"),
        duplicate_threshold=c["split"]["duplicate_hamming_threshold"],
    )
    payload["manifest_checksum"] = digest(payload)
    write_json(lockpath, payload)
    verify_lock(c)
    return summary


def read_lock(c):
    p = path_for(c, "test_lock_manifest")
    if not p.exists():
        raise ValueError("No test lock. Complete a valid split first")
    data = read_json(p)
    body = {k: v for k, v in data.items() if k != "manifest_checksum"}
    if digest(body) != data.get("manifest_checksum"):
        raise ValueError("Test lock manifest checksum mismatch")
    if data["test_images"] != [
        x for x in data["all_assignments"] if x["split"] == "test_locked"
    ]:
        raise ValueError("Inconsistent test entries")
    return data


def verify_lock(c, scan_test=True, save=True):
    errors = []
    try:
        lock = read_lock(c)
        if (
            sha256(path_for(c, "reports_dir") / "split_manifest.csv")
            != lock["split_manifest_sha256"]
        ):
            raise ValueError("Split manifest changed")
        records = lock["all_assignments"]
        df = pd.DataFrame(records)
        for column in ["source_group", "sha256_hash", "processed_sha256"]:
            if df.groupby(column).split.nunique().gt(1).any():
                errors.append(column + " crosses splits")
        for s in SPLITS:
            subset = [x for x in records if x["split"] == s]
            if set(x["class_label"] for x in subset) != set(c["classes"]["expected"]):
                errors.append("Missing class in " + s)
            if s == "test_locked" and not scan_test:
                continue
            actual = {p.resolve() for p in files_under(path_for(c, KEYS[s]))}
            expected = {checked_path(c["root"], x["processed_path"]) for x in subset}
            if actual != expected:
                errors.append(s + " file inventory changed")
            for row in subset:
                p = checked_path(c["root"], row["processed_path"])
                if not p.is_relative_to(path_for(c, KEYS[s]).resolve()):
                    errors.append("Assignment outside split")
                    continue
                if not p.is_file() or sha256(p) != row["processed_sha256"]:
                    errors.append("Changed or missing: " + row["processed_path"])
        if not scan_test:
            receipt = read_json(
                path_for(c, "test_lock_manifest").parent
                / "test_verification_receipt.json"
            )
            if receipt["lock_checksum"] != lock["manifest_checksum"]:
                errors.append("Verification receipt stale")
            # stat only: training never opens, hashes, decodes or loads holdout pixels.
            for item in receipt["test_stats"]:
                p = checked_path(c["root"], item["path"])
                st = p.stat()
                if [st.st_size, st.st_mtime_ns, st.st_ctime_ns] != item["stat"]:
                    errors.append(
                        "Test changed since verification; run verify_locked_test.py"
                    )
            actual = {p.resolve() for p in files_under(path_for(c, "locked_test_dir"))}
            expected = {
                checked_path(c["root"], x["processed_path"])
                for x in lock["test_images"]
            }
            if actual != expected:
                errors.append("Test inventory changed")
        elif not errors:
            stats = []
            for row in lock["test_images"]:
                st = checked_path(c["root"], row["processed_path"]).stat()
                stats.append(
                    dict(
                        path=row["processed_path"],
                        stat=[st.st_size, st.st_mtime_ns, st.st_ctime_ns],
                    )
                )
            write_json(
                path_for(c, "test_lock_manifest").parent
                / "test_verification_receipt.json",
                dict(
                    lock_checksum=lock["manifest_checksum"],
                    verified_at=now(),
                    test_stats=stats,
                ),
            )
    except (OSError, ValueError, KeyError, TypeError) as e:
        errors.append(str(e))
    result = dict(
        passed=not errors,
        errors=errors,
        mode=(
            "full_hash_verification"
            if scan_test
            else "train_val_hashes_and_test_stat_receipt"
        ),
        timestamp=now(),
    )
    if save:
        save_report(path_for(c, "reports_dir"), "test_lock_verification", result)
    if errors:
        raise ValueError("Test lock validation failed: " + "; ".join(errors))
    return result
