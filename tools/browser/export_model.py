"""Export only checksum-verified public checkpoints for browser inference.

Run from the repository root, with the project CPU/torch dependencies installed:
  python tools/browser/export_model.py
An optional --checkpoint-dir reuses local copies of the same public files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def export(output: Path, checkpoint_dir: Path | None = None) -> dict:
    if checkpoint_dir is not None:
        os.environ['WLCR_SEA_CHECKPOINT_DIR'] = str(checkpoint_dir.resolve())
    import numpy as np
    import torch
    from demo.model_loader import (
        CHECKPOINT_SPECS, MODEL_REPO_ID, MODEL_REVISION, MODEL_VARIANT,
        _resolve_checkpoint,
    )
    chunks = []
    members = []
    offset = 0
    for spec in CHECKPOINT_SPECS:
        path = _resolve_checkpoint(spec)  # verifies the public registry SHA-256
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['seed'] == spec.seed
        variant = payload['variant']
        assert variant['name'] == MODEL_VARIANT
        assert variant['attention'] == 'entmax15' and variant['hard_mask']
        assert variant['reliability'] and not variant['cross_indicator_context']
        arrays = {'prior_log': np.asarray(payload['prior_log'], dtype='<f4')}
        arrays.update({key: tensor.detach().cpu().numpy().astype('<f4')
                       for key, tensor in payload['state_dict'].items()})
        tensors = {}
        for name, array in arrays.items():
            assert np.isfinite(array).all(), name
            tensors[name] = {'shape': list(array.shape), 'offset': offset, 'length': int(array.size)}
            chunks.append(array.tobytes(order='C'))
            offset += int(array.size)
        members.append({
            'seed': spec.seed, 'checkpoint': spec.filename,
            'checkpoint_sha256': spec.sha256,
            'source_url': f'https://huggingface.co/{MODEL_REPO_ID}/resolve/{MODEL_REVISION}/checkpoints/{spec.filename}',
            'selected_config': payload['selected_config'],
            'selected_epoch': int(payload['selected_epoch']), 'tensors': tensors,
        })
    data = b''.join(chunks)
    manifest = {
        'schema': 'wlcr-sea-browser-model/v1', 'repo_id': MODEL_REPO_ID,
        'revision': MODEL_REVISION, 'variant': MODEL_VARIANT,
        'aggregation': 'arithmetic mean of individually inverse-transformed and floored predictions',
        'weights': {'file': 'weights.bin', 'dtype': 'float32-le', 'byte_length': len(data),
                    'sha256': hashlib.sha256(data).hexdigest()},
        'members': members,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'weights.bin').write_bytes(data)
    # The public demo sample is synthetic; normalize only CSV record separators.
    sample = (ROOT/'demo/examples/synthetic_traffic.csv').read_text(encoding='utf-8')
    (output/'sample.csv').write_text(sample, encoding='utf-8')
    manifest['sample'] = {'file': 'sample.csv', 'source': 'demo/examples/synthetic_traffic.csv',
                          'synthetic': True, 'sha256': hashlib.sha256((output/'sample.csv').read_bytes()).hexdigest()}
    (output / 'model.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(f'Exported five verified PUBLIC checkpoints: {len(data):,} bytes; revision {MODEL_REVISION}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'docs/assets/local-demo')
    args = parser.parse_args()
    export(args.output_dir, args.checkpoint_dir)
