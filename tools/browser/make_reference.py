"""Regenerate synthetic browser-test predictions using the public PyTorch model.

Requires the project torch environment and the checksum-verified public weights.
No operator data or paper aggregate CSVs are read.
"""
from pathlib import Path
from datetime import datetime
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Model.traffic_window_forecasting import read_traffic
from demo.model_loader import load_a6_ensemble, MODEL_REVISION
from experiments import wlcr_sea_model as sea
from experiments.missingness_protocol import global_corruption_mask


def make_reference():
    rows = read_traffic(ROOT/'docs/assets/local-demo/sample.csv')
    raw = np.array([[float(v) for v in row.metrics] for row in rows], dtype=np.float32)[None]
    bundle = load_a6_ensemble()
    manifest = json.loads((ROOT/'docs/assets/local-demo/model.json').read_text())
    cases = []
    definitions = [
        ('complete', 'sample', 'none', 0.0),
        ('mcar', 'sample', 'mcar', 0.3),
        ('block', 'sample', 'block', 0.5),
        ('tail', 'sample', 'recent_tail', 0.5),
        ('async', 'sample', 'asynchronous', 0.5),
        ('inherent_gaps', 'gapped', 'none', 0.0),
        ('zeros', 'zeros', 'none', 0.0),
        ('single_observation', 'sparse', 'none', 0.0),
    ]
    for name, source, mechanism, rate in definitions:
        values = raw.copy(); mask = np.ones_like(values, dtype=bool)
        if source == 'zeros': values[:] = 0
        if source == 'sparse': mask[:, :-1] = False
        if source == 'gapped':
            for h in range(336):
                for q in range(4):
                    if (h+q) % 5 == 0: mask[0,h,q] = False
        end_hour = int((rows[-1].timestamp - datetime(1970,1,1)).total_seconds()//3600)
        extra = global_corruption_mask(np.array([rows[0].cell]),np.array([end_hour]), mechanism=mechanism,requested_rate=rate,seed=2026)
        log = np.where(mask,np.log1p(values),0).astype(np.float32)
        predictions = []
        with torch.inference_mode():
            for member in bundle.members:
                experts = sea.build_expert_batch(log,mask,member.prior_log,additional_missing=extra)
                out = member.model(*[torch.from_numpy(a) for a in (experts.values,experts.availability,experts.reliability,experts.context)])
                predictions.append(sea.prediction_from_log(out['prediction_log'].numpy()[0]))
        cases.append({'name':name,'source':source,'scenario':mechanism,'rate':rate,
                      'observed':int(np.count_nonzero(mask & ~extra)),
                      'prediction':np.mean(predictions,axis=0).astype(np.float32).tolist()})
    result={'schema':'wlcr-sea-browser-reference/v1','model_revision':MODEL_REVISION,
            'weights_sha256':manifest['weights']['sha256'],
            'source':'Only the bundled synthetic sample; expected values calculated by the published PyTorch implementation.',
            'cases':cases}
    p=ROOT/'tests/fixtures/browser_demo_reference.json'
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Wrote',len(cases),'synthetic PyTorch reference cases')

if __name__ == '__main__':
    make_reference()
