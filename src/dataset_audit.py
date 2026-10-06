import pandas as pd
from .manifest import build_manifest
from .duplicate_detection import find_duplicates
from .config_loader import path_for
from .reporting import save_report


def audit(c):
    # Rescan read-only so stale CSV cannot conceal corrupt/changed files.
    df = build_manifest(c, save=False)
    r = path_for(c, "reports_dir")
    r.mkdir(parents=True, exist_ok=True)
    exact, near = find_duplicates(df, c["split"]["duplicate_hamming_threshold"])
    exact.to_csv(r / "exact_duplicates.csv", index=False)
    near.to_csv(r / "near_duplicates.csv", index=False)
    conflicts = pd.concat([exact, near], ignore_index=True)
    conflicts = conflicts[conflicts.label_conflict.astype(bool)]
    conflicts.to_csv(r / "label_conflict_candidates.csv", index=False)
    valid = df[df.is_valid_image]
    counts = {k: int((valid.class_label == k).sum()) for k in c["classes"]["expected"]}
    dims = valid[["image_width", "image_height"]].astype(float).copy()
    dims["aspect_ratio"] = dims.image_width / dims.image_height if len(dims) else []
    stats = (
        {
            k: {
                str(q): float(v)
                for q, v in dims[k].quantile([0, 0.25, 0.5, 0.75, 1]).items()
            }
            for k in dims
        }
        if len(dims)
        else {}
    )

    def dup_stats(p):
        return {
            "total": len(p),
            "within_class": int((p.label_a == p.label_b).sum()),
            "across_classes": int((p.label_a != p.label_b).sum()),
            "across_source_groups": int((p.source_group_a != p.source_group_b).sum()),
        }

    result = dict(
        total_files=len(df),
        valid_images=len(valid),
        invalid_images=len(df) - len(valid),
        class_counts=counts,
        class_balance_ratio=(
            max(counts.values()) / min(counts.values())
            if min(counts.values())
            else None
        ),
        dimension_quantiles=stats,
        small_image_warnings=df[
            df.audit_flags.str.contains("extremely_small")
        ].relative_path.tolist(),
        flagged_files=df[df.audit_flags.ne("")][
            ["relative_path", "audit_flags"]
        ].to_dict("records"),
        exact_duplicates=dup_stats(exact),
        near_duplicates=dup_stats(near),
        label_conflicts=len(conflicts),
        source_groups=valid.groupby(["class_label", "source_group"])
        .size()
        .reset_index(name="count")
        .to_dict("records"),
        metadata_declared_sufficient=bool(
            len(valid)
            and valid.source_type.eq("verified_group").all()
            and valid.source_group.ne("").all()
            and not valid.source_group.str.startswith(("fallback:", "inferred:")).any()
        ),
        note="Near duplicates: manual review required. Metadata declarations cannot prove independent plants. No raw files modified.",
    )
    save_report(r, "raw_dataset_audit", result)
    return result
