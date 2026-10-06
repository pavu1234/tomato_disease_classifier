"""Atomic, portable checkpoint bundles; excludes images and temporary files."""
import zipfile
import json
from pathlib import Path

def snapshot_project(root, stage, epoch):
    root = Path(root)
    state = json.loads((root / 'logs/training_state.json').read_text())
    (root / 'RUN_STATUS.md').write_text(
        '# Training checkpoint — not final results\n\n'
        + f"Stage: {stage}; completed epochs in this stage: {epoch}.\n\n"
        + 'Training is in progress. This recovery bundle contains resumable weights, source code, '
        + 'approved metadata, and fixed split assignments, but not bulk images. '
        + 'Use logs/training_state.json for the current stage. Final model selection and exports '
        + 'have not necessarily completed. Do not interpret interim validation results as test performance.\n'
    )
    snapshots = root.parent / 'training_snapshots'
    snapshots.mkdir(exist_ok=True)
    dest = snapshots / f'{stage}_{epoch:03d}.zip'
    temp = dest.with_suffix('.tmp')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(root.rglob('*')):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if any(x in rel.parts for x in ['__pycache__', '.pytest_cache']):
                continue
            if len(rel.parts)>1 and rel.parts[0]=='data' and rel.parts[1] in ['raw','processed','curated']:
                if p.name!='.gitkeep':
                    continue
            if p.name.endswith('.tmp'):
                continue
            archive.write(p,Path('tomato_disease_classifier')/rel)
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Checkpoint archive failed integrity verification')
    temp.replace(dest)
    print(f'RECOVERY_BUNDLE: {dest}',flush=True)
    return str(dest)
