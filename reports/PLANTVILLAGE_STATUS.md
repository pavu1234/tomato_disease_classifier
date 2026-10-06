# Historical PlantVillage import audit

**Subsequent decision:** The user approved the 2,902-image conservative subset. The original stop below was resolved by that approval. Current training status and results are in `../../RUN_STATUS.md` and this report’s sibling `approved_subset/` directory. The findings below describe the original 4,500-image import, before curation.

Dataset retrieved on 2026-10-06 from https://github.com/spMohanty/PlantVillage-Dataset at commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`.

Only `raw/color/Tomato___Early_blight`, `raw/color/Tomato___healthy`, and `raw/color/Tomato___Late_blight` were imported. No grayscale or segmented variants were mixed in. Source files were not modified or deleted. This package includes source code, manifest, provenance, audit reports and review imagery; it excludes the bulk source/processed dataset.

## Actual measured audit

| Class | Valid images | Resolved upstream leaf metadata | Unresolved leaf metadata |
|---|---:|---:|---:|
| early_blight | 1,000 | 999 | 1 |
| healthy | 1,591 | 999 | 592 |
| late_blight | 1,909 | 920 | 989 |
| Total | 4,500 | 2,918 | 1,582 |

All imported images decoded successfully and measured 256×256. No corrupt, zero-byte, unsupported or extremely small images were found in these selected folders.

- 14 exact duplicate **pairs**, all within a class; none across class labels.
- 51 near-duplicate **pairs** at 64-bit pHash distance ≤6: 48 within a class and 3 across labels.
- All 3 cross-label pairs have distance 6 and require review. These are candidates, not automatically confirmed labelling errors.
- Leaf-map resolution verifies a reference to the upstream **leaf** identity only. It does not prove plant-, farm-, or capture-session independence.

## Review images

Open `label_conflict_review.png` for side-by-side images, and `label_conflict_candidates.csv` for exact paths/hashes/group IDs.

Pair 1: `RS_Erly.B 7844.JPG` (early blight) versus `RS_HL 0482.JPG` (healthy).

Pair 2: `RS_Erly.B 9440.JPG` (early blight) versus `RS_Late.B 5421.JPG` (late blight).

Pair 3: `GH_HL Leaf 170.1.JPG` (healthy) versus `GH_HL Leaf 170.JPG` (late blight). The shared filename identifier and visual similarity warrant particular attention, but the correct diagnosis has not been established.

Visual review suggests the first two candidates have different leaf details; pHash can match a similar silhouette/background. No automated dismissal, relabelling, similarity-threshold reduction, or safety override was applied.

## Why training stopped

The original project specification explicitly requires stopping and reporting missing source-group information and unresolved label conflicts. The attempted strict dry run refused the unverified metadata. The separate audit also established unresolved cross-label similarity candidates. No processed split, test lock, training run, trained model, test exposure, final evaluation or disease accuracy exists for this dataset.

## Concrete proposal for the next decision

A conservative, **not yet applied** subset rule is recorded in `proposed_subset_review.csv` and `proposed_subset_summary.json`:

1. Form connected components from source groups, exact duplicates and all detected near duplicates.
2. Exclude every component touching unresolved source metadata or a cross-label similarity candidate.
3. Keep all remaining images with their upstream labels; leave the original dataset intact.

This leaves **2,902 images across 726 leaf groups**:

| Class | Proposed retained images |
|---|---:|
| early_blight | 991 |
| healthy | 995 |
| late_blight | 916 |

The policy excludes 1,582 images lacking resolved metadata and 16 additional images in affected verified leaf groups, for 1,598 exclusions in total. It is a provenance/conflict rule applied before any split or model prediction, not performance-based cherry-picking. Even then, final scores would describe this curated, metadata-resolved PlantVillage subset, not all 4,500 source images or field/drone performance. No diagnostic decision is made about excluded images.

The user's decision is needed before adopting this narrower training population under the original stop-on-blocker requirement. An alternative is to obtain verified missing source metadata and expert adjudication of conflicts; the software must not fabricate these.

## Reproduce import and audit

From a fresh extracted project (bulk image folders must be empty):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git ../PlantVillage-source
git -C ../PlantVillage-source fetch origin 7f7ecc7e1eaca78107e3affe7cb5abd9427e139a
git -C ../PlantVillage-source checkout 7f7ecc7e1eaca78107e3affe7cb5abd9427e139a
git -C ../PlantVillage-source sparse-checkout set raw/color/Tomato___Early_blight raw/color/Tomato___Late_blight raw/color/Tomato___healthy leaf_grouping
python scripts/import_plantvillage.py --source ../PlantVillage-source
python scripts/audit_dataset.py
python scripts/split_dataset.py --dry-run
```

The final command is expected to refuse the unresolved grouping metadata. Do not add fallback flags or remove conflicts without documenting the chosen policy. The importer is intentionally non-overwriting; an existing populated raw directory is not overwritten.

Dataset citation: Mohanty, Hughes and Salathé (2016), *Using Deep Learning for Image-Based Plant Disease Detection*, Frontiers in Plant Science, DOI 10.3389/fpls.2016.01419. Consult upstream documentation for applicable dataset terms.
