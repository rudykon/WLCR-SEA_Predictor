from __future__ import annotations

"""Strict cell-disjoint transfer evaluation for the frozen WLCR-SEA design."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments import train_neural_baselines as neural
from experiments import train_wlcr_sea as runner
from experiments import wlcr_sea_model as sea

SCHEMA_VERSION = 1
FOLDS = 5
SEED = 42


def fold_mapping(cells: Sequence[str]) -> dict[str, int]:
    ordered = sorted(set(str(cell) for cell in cells))
    return {cell: index % FOLDS for index, cell in enumerate(ordered)}


def load_source(source: Path) -> dict[str, object]:
    checkpoint = source / "models" / f"{runner.PRIMARY_VARIANT}_seed{SEED}.pt"
    if not checkpoint.is_file():
        raise ValueError(f"missing frozen temporal checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu")
    if payload["variant"]["name"] != runner.PRIMARY_VARIANT:
        raise ValueError("source checkpoint is not the predeclared primary variant")
    return payload


def run_worker(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("unseen worker requires one visible CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dataset = neural.load_dataset_cache(Path(args.dataset_cache).resolve(strict=True))
    fit = dataset.indices_for_dates(neural.FIT_DATES)
    inner = dataset.indices_for_dates(neural.INNER_DATES)
    holdout = dataset.indices_for_dates(neural.HOLDOUT_DATES)
    final_train = np.concatenate((fit, inner))
    mapping = fold_mapping(np.concatenate((dataset.cells[final_train], dataset.cells[holdout])).tolist())
    train = np.asarray([index for index in final_train if mapping[str(dataset.cells[index])] != args.fold], dtype=np.int64)
    evaluate = np.asarray([index for index in holdout if mapping[str(dataset.cells[index])] == args.fold], dtype=np.int64)
    if args.smoke:
        train, evaluate = train[:256], evaluate[:128]
    if not len(train) or not len(evaluate):
        raise ValueError(f"empty unseen fold {args.fold}")
    source_payload = load_source(Path(args.source).resolve(strict=True))
    variant = sea.VariantConfig(**source_payload["variant"])
    config = source_payload["selected_config"]
    epochs = 2 if args.smoke else int(source_payload["selected_epoch"])
    prior = sea.training_prior_log(dataset.targets, dataset.target_masks, train)
    train_tensors, corruption = runner.make_training_tensors(
        dataset, train, prior, variant, seed=SEED + args.fold * 1000
    )
    eval_batch, eval_tensors = runner.make_eval_tensors(dataset, evaluate, prior)
    model, outputs, training = runner.train_final(
        variant=variant,
        config=config,
        seed=SEED + args.fold,
        epochs=epochs,
        train_tensors=train_tensors,
        holdout_tensors=eval_tensors,
        device=device,
        batch_size=args.batch_size,
        include_audit=False,
    )
    actual = np.asarray(dataset.targets[evaluate], dtype=np.float32)
    scales = np.asarray(dataset.mase_scales[evaluate], dtype=np.float32)
    eval_cells = np.asarray(dataset.cells[evaluate])
    thresholds = sea.frozen_low_activity_thresholds(dataset.targets, dataset.target_masks, train)
    fixed = sea.WLCRSEA(sea.VARIANTS["A0_fixed"], token_dim=16, hidden_dim=32)
    fixed_output = runner.predict(
        fixed, eval_tensors, device=torch.device("cpu"), batch_size=args.batch_size
    )
    output = Path(args.output).resolve(strict=False)
    worker_dir = output / "worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = worker_dir / f"fold{args.fold}_predictions.npz"
    np.savez_compressed(
        prediction_file,
        indices=evaluate,
        proposed=outputs["prediction"],
        fixed=fixed_output["prediction"],
    )
    model_file = worker_dir / f"fold{args.fold}_model.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "fold": args.fold,
            "seed": SEED + args.fold,
            "variant": source_payload["variant"],
            "config": config,
            "epochs": epochs,
            "prior_log": prior,
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        },
        model_file,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "fold": args.fold,
        "physical_gpu": args.physical_device,
        "train_windows": len(train),
        "evaluation_windows": len(evaluate),
        "train_cells": len(set(str(dataset.cells[index]) for index in train)),
        "evaluation_cells": len(set(eval_cells.tolist())),
        "cell_overlap": len(set(str(dataset.cells[index]) for index in train).intersection(eval_cells.tolist())),
        "configuration_frozen_from_temporal_inner_layer": True,
        "epochs": epochs,
        "augmentation": corruption,
        "proposed_metrics": sea.forecast_metrics(actual, outputs["prediction"], scales, eval_cells),
        "fixed_metrics": sea.forecast_metrics(actual, fixed_output["prediction"], scales, eval_cells),
        "proposed_threshold_hit_score": sea.threshold_hit_score(actual, outputs["prediction"], thresholds),
        "fixed_threshold_hit_score": sea.threshold_hit_score(actual, fixed_output["prediction"], thresholds),
        "training": training,
        "prediction_file": str(prediction_file.relative_to(output)),
        "model_file": str(model_file.relative_to(output)),
    }
    runner.atomic_json(worker_dir / f"fold{args.fold}.json", report)
    print(json.dumps({"status":"complete","fold":args.fold}))
    return 0


def launch(
    fold: int,
    device: int,
    script: Path,
    cache: Path,
    source: Path,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    command=[
        sys.executable,str(script),"--worker","--fold",str(fold),
        "--physical-device",str(device),"--dataset-cache",str(cache),
        "--source",str(source),"--output",str(output),"--batch-size",str(args.batch_size),
    ]
    if args.smoke: command.append("--smoke")
    environment=os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"]=str(device)
    environment["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    environment["OMP_NUM_THREADS"]="4"
    environment["MKL_NUM_THREADS"]="4"
    completed=subprocess.run(command,cwd=runner.project_root(),env=environment,capture_output=True,text=True)
    log=output/"logs"/f"fold{fold}.log"
    log.parent.mkdir(parents=True,exist_ok=True)
    log.write_text("COMMAND\n"+" ".join(command)+"\n\nSTDOUT\n"+completed.stdout+"\nSTDERR\n"+completed.stderr,encoding="utf-8")
    return {"fold":fold,"device":device,"returncode":completed.returncode,"log":str(log.relative_to(output))}


def aggregate(output: Path, dataset: neural.CachedDataset, holdout: np.ndarray) -> dict[str, object]:
    proposed=np.full(np.asarray(dataset.targets[holdout]).shape,np.nan,dtype=np.float32)
    fixed=np.full_like(proposed,np.nan)
    position={int(index):offset for offset,index in enumerate(holdout.tolist())}
    fold_rows=[]
    for fold in range(FOLDS):
        report=json.loads((output/"worker"/f"fold{fold}.json").read_text(encoding="utf-8"))
        fold_rows.append({
            "fold":fold,
            "train_cells":report["train_cells"],
            "evaluation_cells":report["evaluation_cells"],
            "evaluation_windows":report["evaluation_windows"],
            "cell_overlap":report["cell_overlap"],
            "proposed_macro_wape":report["proposed_metrics"]["macro_indicator"]["wape"],
            "fixed_macro_wape":report["fixed_metrics"]["macro_indicator"]["wape"],
            "macro_wape_gain":report["fixed_metrics"]["macro_indicator"]["wape"]-report["proposed_metrics"]["macro_indicator"]["wape"],
        })
        arrays=np.load(output/str(report["prediction_file"]),allow_pickle=False)
        for local,index in enumerate(arrays["indices"].tolist()):
            offset=position[int(index)]
            proposed[offset]=arrays["proposed"][local]
            fixed[offset]=arrays["fixed"][local]
    if np.any(~np.isfinite(proposed)) or np.any(~np.isfinite(fixed)):
        raise ValueError("unseen folds did not cover every holdout prediction")
    actual=np.asarray(dataset.targets[holdout],dtype=np.float32)
    scales=np.asarray(dataset.mase_scales[holdout],dtype=np.float32)
    cells=np.asarray(dataset.cells[holdout])
    payload={
        "schema_version":SCHEMA_VERSION,
        "protocol":"five deterministic cell-disjoint folds; frozen temporal configuration; no evaluation cell appears in training",
        "evidence_status":"exploratory_redesign_on_existing_trace",
        "folds":fold_rows,
        "positive_gain_folds":int(sum(float(row["macro_wape_gain"])>0.0 for row in fold_rows)),
        "proposed_metrics":sea.forecast_metrics(actual,proposed,scales,cells),
        "fixed_metrics":sea.forecast_metrics(actual,fixed,scales,cells),
        "paired_bootstrap":sea.cell_cluster_bootstrap_wape_delta(actual,proposed,fixed,cells,replicates=5000,seed=42),
    }
    runner.atomic_csv(output/"fold_results.csv",fold_rows)
    runner.atomic_json(output/"summary.json",payload)
    np.save(output/"unseen_proposed.npy",proposed,allow_pickle=False)
    np.save(output/"unseen_fixed.npy",fixed,allow_pickle=False)
    return payload


def run_master(args: argparse.Namespace) -> int:
    source=Path(args.source)
    if not source.is_absolute(): source=runner.project_root()/source
    source=source.resolve(strict=True)
    output=Path(args.output)
    if not output.is_absolute(): output=runner.project_root()/output
    output=output.resolve(strict=False)
    allowed=(runner.project_root()/runner.OUTPUT_ROOT).resolve(strict=False)
    if not output.is_relative_to(allowed): raise ValueError("unseen outputs must remain under artifacts/revision6")
    output.mkdir(parents=True,exist_ok=True)
    source_payload=load_source(source)
    train_path=neural.resolve_train_path()
    before=neural.sha256_file(train_path)
    devices=[int(item) for item in args.gpu_devices.split(",")]
    with tempfile.TemporaryDirectory(prefix="wlcr-sea-unseen-") as temporary:
        cache=Path(temporary)
        arrays,report=neural.build_window_arrays(neural.read_training_series(train_path))
        neural.write_dataset_cache(cache,arrays)
        dataset=neural.load_dataset_cache(cache)
        holdout=dataset.indices_for_dates(neural.HOLDOUT_DATES)
        jobs=[(fold,devices[fold%len(devices)]) for fold in range(1 if args.smoke else FOLDS)]
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            results=list(executor.map(lambda job:launch(job[0],job[1],Path(__file__).resolve(),cache,source,output,args),jobs))
        failures=[item for item in results if item["returncode"]!=0]
        runner.atomic_json(output/"worker_status.json",results)
        if failures: return 1
        if args.smoke:
            runner.atomic_json(output/"smoke_summary.json",json.loads((output/"worker"/"fold0.json").read_text()))
        else:
            aggregate(output,dataset,holdout)
    after=neural.sha256_file(train_path)
    if before!=after: raise RuntimeError("registered training data changed")
    runner.atomic_json(output/"protocol.json",{
        "schema_version":SCHEMA_VERSION,
        "source_checkpoint":str((source/"models"/f"{runner.PRIMARY_VARIANT}_seed42.pt").relative_to(runner.project_root())),
        "source_checkpoint_sha256":runner.sha256_file(source/"models"/f"{runner.PRIMARY_VARIANT}_seed42.pt"),
        "configuration_frozen":source_payload["selected_config"],
        "selected_epoch":source_payload["selected_epoch"],
        "folds":FOLDS,
        "seed":SEED,
        "finals_test_opened":False,
        "input_sha256_before":before,
        "input_sha256_after":after,
        "smoke":args.smoke,
    })
    runner.atomic_json(output/"manifest.json",runner.output_manifest(output))
    return 0


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Run strict cell-disjoint WLCR-SEA transfer evaluation")
    p.add_argument("--source",default=str(runner.DEFAULT_OUTPUT))
    p.add_argument("--output",default=str(runner.DEFAULT_OUTPUT/"unseen"))
    p.add_argument("--gpu-devices",default="0,1,2,3")
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    p.add_argument("--fold",type=int,help=argparse.SUPPRESS)
    p.add_argument("--physical-device",type=int,help=argparse.SUPPRESS)
    p.add_argument("--dataset-cache",help=argparse.SUPPRESS)
    return p


def main() -> int:
    args=parser().parse_args()
    if args.worker:
        try:return run_worker(args)
        except Exception:
            traceback.print_exc();return 1
    return run_master(args)


if __name__=="__main__":
    raise SystemExit(main())
