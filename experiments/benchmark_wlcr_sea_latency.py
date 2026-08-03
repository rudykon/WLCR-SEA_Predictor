from __future__ import annotations

"""Matched-information batch-one CPU latency for WLCR-SEA and neural baselines."""

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from experiments import train_neural_baselines as neural
from experiments import train_wlcr_sea as runner
from experiments import wlcr_sea_model as sea


def load_sea(source: Path) -> tuple[sea.WLCRSEA, dict[str, object]]:
    path=source/"models"/f"{runner.PRIMARY_VARIANT}_seed42.pt"
    payload=torch.load(path,map_location="cpu")
    variant=sea.VariantConfig(**payload["variant"])
    model=runner.model_from_config(variant,payload["selected_config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model,payload


def load_neural(
    root: Path, model_name: str
) -> tuple[torch.nn.Module, dict[str, object], Path] | None:
    path=root/"models"/f"{model_name}_seed42.pt"
    if not path.is_file(): return None
    payload=torch.load(path,map_location="cpu")
    if payload.get("model") != model_name:
        raise ValueError(f"checkpoint/model mismatch for {path}")
    model=neural.build_model(model_name,payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model,payload,path


def distribution(values: list[float]) -> dict[str,float]:
    array=np.asarray(values,dtype=np.float64)
    return {
        "mean_ms":float(np.mean(array)),
        "sample_sd_ms":float(statistics.stdev(values)) if len(values)>1 else 0.0,
        "p50_ms":float(np.quantile(array,0.50)),
        "p90_ms":float(np.quantile(array,0.90)),
        "p95_ms":float(np.quantile(array,0.95)),
        "p99_ms":float(np.quantile(array,0.99)),
        "min_ms":float(np.min(array)),
        "max_ms":float(np.max(array)),
    }


def benchmark(call: Callable[[int],None], identities: int, warmups: int, measured: int) -> dict[str,float]:
    for index in range(warmups): call(index%identities)
    timings=[]
    for index in range(measured):
        start=time.perf_counter_ns();call(index%identities);end=time.perf_counter_ns()
        timings.append((end-start)/1_000_000.0)
    return distribution(timings)


def run(args: argparse.Namespace) -> int:
    root=runner.project_root()
    source=(root/args.source).resolve(strict=True) if not Path(args.source).is_absolute() else Path(args.source).resolve(strict=True)
    output=(root/args.output).resolve(strict=False) if not Path(args.output).is_absolute() else Path(args.output).resolve(strict=False)
    allowed=(root/runner.OUTPUT_ROOT).resolve(strict=False)
    if not output.is_relative_to(allowed): raise ValueError(f"latency output must remain under {runner.OUTPUT_ROOT}")
    output.parent.mkdir(parents=True,exist_ok=True)
    baseline_root=(root/args.baseline_root).resolve(strict=False) if not Path(args.baseline_root).is_absolute() else Path(args.baseline_root).resolve(strict=False)
    train_path=neural.resolve_train_path();before=neural.sha256_file(train_path)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    os.environ["OMP_NUM_THREADS"]="1";os.environ["MKL_NUM_THREADS"]="1"
    with tempfile.TemporaryDirectory(prefix="wlcr-sea-latency-") as temporary:
        arrays,_=neural.build_window_arrays(neural.read_training_series(train_path))
        neural.write_dataset_cache(Path(temporary),arrays)
        dataset=neural.load_dataset_cache(Path(temporary))
        holdout=dataset.indices_for_dates(neural.HOLDOUT_DATES)
        final_train=np.concatenate((dataset.indices_for_dates(neural.FIT_DATES),dataset.indices_for_dates(neural.INNER_DATES)))
        count=min(args.identities,len(holdout))
        selected=holdout[np.linspace(0,len(holdout)-1,count,dtype=int)]
        x_values=np.asarray(dataset.x_values[selected],dtype=np.float32)
        x_masks=np.asarray(dataset.x_masks[selected],dtype=np.uint8)
        sea_model,sea_payload=load_sea(source)
        prior=np.asarray(sea_payload["prior_log"],dtype=np.float32)
        fixed_model=sea.WLCRSEA(sea.VARIANTS["A0_fixed"],token_dim=16,hidden_dim=32).eval()
        def sea_call(index:int)->None:
            batch=sea.build_expert_batch(x_values[index:index+1],x_masks[index:index+1],prior)
            tensors=runner.batch_to_tensors(batch)
            with torch.no_grad():
                output=sea_model(tensors[0],tensors[1].bool(),tensors[2],tensors[3])
                _=sea.prediction_from_log(output["prediction_log"])
        def fixed_call(index:int)->None:
            batch=sea.build_expert_batch(x_values[index:index+1],x_masks[index:index+1],prior)
            tensors=runner.batch_to_tensors(batch)
            with torch.no_grad():
                output=fixed_model(tensors[0],tensors[1].bool(),tensors[2],tensors[3])
                _=sea.prediction_from_log(output["prediction_log"])
        fixed_asset_path=source/"baselines"/"fixed_seasonal_assets.npz"
        if not fixed_asset_path.is_file():
            raise ValueError(f"missing fixed seasonal asset file: {fixed_asset_path}")
        with np.load(fixed_asset_path,allow_pickle=False) as fixed_assets:
            frozen_scalar_assets=int(sum(np.asarray(fixed_assets[key]).size for key in fixed_assets.files))
        sea_trainable=runner.count_parameters(sea_model)
        results={
            runner.PRIMARY_VARIANT:{
                "latency":benchmark(sea_call,count,args.warmups,args.measured),
                "trainable_parameters":sea_trainable,
                "parameter_count":sea_trainable,
                "frozen_scalar_assets":int(np.asarray(prior).size),
                "serialized_size_bytes":int((source/"models"/f"{runner.PRIMARY_VARIANT}_seed42.pt").stat().st_size),
                "model_size_bytes":int((source/"models"/f"{runner.PRIMARY_VARIANT}_seed42.pt").stat().st_size),
            },
            "A0_fixed":{
                "latency":benchmark(fixed_call,count,args.warmups,args.measured),
                "trainable_parameters":0,
                "parameter_count":0,
                "frozen_scalar_assets":frozen_scalar_assets,
                "serialized_size_bytes":int(fixed_asset_path.stat().st_size),
                "model_size_bytes":int(fixed_asset_path.stat().st_size),
                "serialized_asset_file":str(fixed_asset_path.relative_to(root)),
            },
        }
        for model_name, label in (("dlinear", "DLinear"), ("patchtst", "PatchTST")):
            loaded=load_neural(baseline_root,model_name)
            if loaded is None:
                continue
            model,payload,path=loaded
            normalization=neural.Normalization(**payload["normalization"])
            input_mean=np.asarray(normalization.input_mean,dtype=np.float32)
            input_std=np.asarray(normalization.input_std,dtype=np.float32)
            target_mean=np.asarray(normalization.target_mean,dtype=np.float32)
            target_std=np.asarray(normalization.target_std,dtype=np.float32)
            def neural_call(index:int)->None:
                values=(x_values[index:index+1]-input_mean[None,None,:])/input_std[None,None,:]
                inputs=torch.from_numpy(np.concatenate((values,x_masks[index:index+1].astype(np.float32)),axis=2))
                with torch.no_grad():
                    normalized=model(inputs)
                    prediction=torch.expm1(normalized*torch.from_numpy(target_std)[None,None,:]+torch.from_numpy(target_mean)[None,None,:]).clamp_min(1e-4)
                    _=prediction.numpy()
            trainable=neural.count_parameters(model)
            results[label]={
                "latency":benchmark(neural_call,count,args.warmups,args.measured),
                "trainable_parameters":trainable,
                "parameter_count":trainable,
                "frozen_scalar_assets":int(
                    len(normalization.input_mean)+len(normalization.input_std)
                    +len(normalization.target_mean)+len(normalization.target_std)
                ),
                "serialized_size_bytes":int(path.stat().st_size),
                "model_size_bytes":int(path.stat().st_size),
            }
    after=neural.sha256_file(train_path)
    if before!=after: raise RuntimeError("registered training data changed")
    payload={
        "schema_version":1,
        "protocol":"same machine, one CPU thread, batch size one, preprocessing + model + postprocessing",
        "request_identities":count,
        "warmups":args.warmups,
        "measured_requests":args.measured,
        "matched_information":"target-cell traffic and masks; no weather, parameters, coordinates, or cell ID features",
        "results":results,
        "finals_test_opened":False,
        "input_sha256_before":before,
        "input_sha256_after":after,
    }
    runner.atomic_json(output,payload)
    return 0


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Benchmark batch-one CPU WLCR-SEA latency")
    p.add_argument("--source",default=str(runner.DEFAULT_OUTPUT))
    p.add_argument("--baseline-root",default="artifacts/reproduction/neural_baselines/mixed")
    p.add_argument("--output",default=str(runner.DEFAULT_OUTPUT/"latency.json"))
    p.add_argument("--identities",type=int,default=256)
    p.add_argument("--warmups",type=int,default=64)
    p.add_argument("--measured",type=int,default=1024)
    return p


if __name__=="__main__":
    raise SystemExit(run(parser().parse_args()))
