"""Exhaustive 64-bit pHash pairs: bounded memory, O(N²) comparisons.
Conservative connected components merge transitive similarity chains.
"""

import pandas as pd

PAIR_COLUMNS = [
    "image_id_a",
    "image_id_b",
    "path_a",
    "path_b",
    "label_a",
    "label_b",
    "source_group_a",
    "source_group_b",
    "kind",
    "hamming_distance",
    "label_conflict",
    "review",
]


def find_duplicates(df, threshold=6):
    rows = df[df.is_valid_image].to_dict("records")
    exact = []
    near = []
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            same = a["sha256_hash"] == b["sha256_hash"]
            distance = (
                int(a["perceptual_hash"], 16) ^ int(b["perceptual_hash"], 16)
            ).bit_count()
            if not same and distance > threshold:
                continue
            item = dict(
                image_id_a=a["image_id"],
                image_id_b=b["image_id"],
                path_a=a["relative_path"],
                path_b=b["relative_path"],
                label_a=a["class_label"],
                label_b=b["class_label"],
                source_group_a=a["source_group"],
                source_group_b=b["source_group"],
                kind="exact" if same else "near",
                hamming_distance=distance,
                label_conflict=a["class_label"] != b["class_label"],
                review=(
                    "critical label conflict"
                    if same and a["class_label"] != b["class_label"]
                    else "manual review required"
                ),
            )
            (exact if same else near).append(item)
    return pd.DataFrame(exact, columns=PAIR_COLUMNS), pd.DataFrame(
        near, columns=PAIR_COLUMNS
    )


def components(df, exact, near):
    parent = {i: i for i in df.image_id}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b):
        parent[find(b)] = find(a)

    for _, g in df.groupby("source_group"):
        ids = list(g.image_id)
        for i in ids[1:]:
            join(ids[0], i)
    for pairs in [exact, near]:
        for p in pairs.itertuples():
            join(p.image_id_a, p.image_id_b)
    return {i: find(i) for i in parent}
