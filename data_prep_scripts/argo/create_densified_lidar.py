import os
os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing
from argparse import ArgumentParser
from multiprocessing import Pool, current_process
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.structures.sweep import Sweep
from tqdm import tqdm
from bucketed_scene_flow_eval.utils.loaders import save_feather, load_feather


def load_multistep_flow(scene_flow_dir: Path, log_id: str, timestamp: int, step: int) -> np.ndarray:
    path = scene_flow_dir / log_id / f"{timestamp}.feather"
    if not path.exists():
        raise FileNotFoundError(f"Missing scene flow file: {path}")
    df = load_feather(path)
    if step == 1:
        flow = np.stack([df["flow_tx_m"], df["flow_ty_m"], df["flow_tz_m"]], axis=-1)
    else:
        flow = np.stack([
            df[f"flow_tx_m_step{step}"],
            df[f"flow_ty_m_step{step}"],
            df[f"flow_tz_m_step{step}"]
        ], axis=-1)
    return flow

def process_log(
    dataset: AV2SensorDataLoader,
    log_id: str,
    output_dir: Path,
    scene_flow_dir: Path,
    num_scans: int = 3,
    n: Optional[int] = None,
):
    timestamps = dataset.get_ordered_log_lidar_timestamps(log_id)

    for i in tqdm(range(len(timestamps)), leave=False, position=n, desc=f"Log {log_id}"):
        ts0 = timestamps[i]
        pose_i = dataset.get_city_SE3_ego(log_id, ts0)

        densified_points = []

        for rel_idx, j in enumerate(range(max(0, i - num_scans + 1), i + 1)):
            if j == i:
                continue  # Skip the current frame

            ts_j = timestamps[j]
            sweep = Sweep.from_feather(dataset.get_lidar_fpath(log_id, ts_j))
            xyz = sweep.xyz.copy()
            intensity = sweep.intensity.copy()
            laser_number = sweep.laser_number.copy()
            offset_ns = sweep.offset_ns.copy()

            if not (len(intensity) == len(xyz) == len(laser_number) == len(offset_ns)):
                raise ValueError(f"Attribute length mismatch at {ts_j}")

            step = i - j
            flow = load_multistep_flow(scene_flow_dir, log_id, ts_j, step)
            if len(flow) != len(xyz):
                raise ValueError(f"Mismatch in flow and point count at {ts_j} for step={step}")
            xyz += flow  # move points within ego_j frame

            pose_j = dataset.get_city_SE3_ego(log_id, ts_j)
            ego_i_SE3_ego_j = pose_i.inverse().compose(pose_j)
            xyz = ego_i_SE3_ego_j.transform_point_cloud(xyz)

            # densified_points.append(xyz)
            points_j = np.column_stack([xyz, intensity, laser_number, offset_ns])
            densified_points.append(points_j)

        if len(densified_points) == 0:
            continue  # No auxiliary points to save

        all_points = np.concatenate(densified_points, axis=0)

        df = pd.DataFrame({
            "x": all_points[:, 0].astype(np.float16),
            "y": all_points[:, 1].astype(np.float16),
            "z": all_points[:, 2].astype(np.float16),
            "intensity": all_points[:, 3].astype(np.uint8),
            "laser_number": all_points[:, 4].astype(np.uint8),
            "offset_ns": all_points[:, 5].astype(np.int32),
        })

        save_path = output_dir / log_id / "sensors" / f"lidar_densification_{num_scans}_scans" / f"{ts0}.feather"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_feather(save_path, df, verbose=False)

def process_log_wrapper(x):
    current = current_process()
    pos = current._identity[0] if current._identity else 0
    process_log(*x, n=pos)


def process_logs(data_dir: Path, output_dir: Path, scene_flow_dir: Path, nproc: int, num_scans: int):
    if not data_dir.exists():
        print(f"{data_dir} not found")
        return

    dataset = AV2SensorDataLoader(data_dir=data_dir, labels_dir=data_dir)
    logs = dataset.get_log_ids()
    args = [(dataset, log, output_dir, scene_flow_dir, num_scans) for log in logs]

    print(f"Using {nproc} processes")
    if nproc <= 1:
        for x in tqdm(args):
            process_log_wrapper(x)
    else:
        with Pool(processes=nproc) as p:
            list(tqdm(p.imap_unordered(process_log_wrapper, args), total=len(args)))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = ArgumentParser(description="Generate densified LiDAR using multi-step GT scene flow labels.")
    parser.add_argument("--argo_dir", type=Path, required=True, help="Root of Argoverse2 data")
    parser.add_argument("--output_dir", type=Path, required=True, help="Where to save densified LiDAR")
    parser.add_argument("--scene_flow_dir", type=Path, required=True, help="Dir with multi-step scene flow .feather files")
    parser.add_argument("--num_scans", type=int, default=3, help="Number of past scans to include")
    parser.add_argument("--nproc", type=int, default=(multiprocessing.cpu_count() - 1), help="Parallel processes")

    args = parser.parse_args()
    process_logs(args.argo_dir, args.output_dir, args.scene_flow_dir, args.nproc, args.num_scans)
