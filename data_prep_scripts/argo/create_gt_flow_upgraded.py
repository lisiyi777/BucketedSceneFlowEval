import os

os.environ["OMP_NUM_THREADS"] = "1"

import enum
import multiprocessing
import os
from argparse import ArgumentParser
from multiprocessing import Pool, current_process
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.structures.cuboid import Cuboid, CuboidList
from av2.structures.sweep import Sweep
from av2.utils.io import read_feather
from tqdm import tqdm

from bucketed_scene_flow_eval.datasets.argoverse2.argoverse_scene_flow import (
    CATEGORY_MAP_INV,
)
from bucketed_scene_flow_eval.utils.loaders import save_feather
from copy import deepcopy

# lidar_densification_3_scans
class DensifiedAV2SensorDataLoader(AV2SensorDataLoader):
    def get_closest_lidar_fpath(self, log_id: str, cam_timestamp_ns: int) -> Optional[Path]:
        lidar_timestamp_ns = self._sdb.get_closest_lidar_timestamp(cam_timestamp_ns, log_id)
        if lidar_timestamp_ns is None:
            return None
        lidar_fname = f"{lidar_timestamp_ns}.feather"
        lidar_fpath = self._data_dir / log_id / "sensors" / "lidar_densification_3_scans" / lidar_fname
        return lidar_fpath

    def get_lidar_fpath_at_lidar_timestamp(self, log_id: str, lidar_timestamp_ns: int) -> Optional[Path]:
        lidar_fname = f"{lidar_timestamp_ns}.feather"
        lidar_fpath = self._data_dir / log_id / "sensors" / "lidar_densification_3_scans" / lidar_fname
        if not lidar_fpath.exists():
            return None
        return lidar_fpath

    def get_lidar_fpath(self, log_id: str, lidar_timestamp_ns: int) -> Path:
        lidar_fname = f"{lidar_timestamp_ns}.feather"
        lidar_fpath = Path(self._data_dir) / log_id / "sensors" / "lidar_densification_3_scans" / lidar_fname
        return lidar_fpath

def category_to_class_id(category: str) -> int:
    assert isinstance(category, str), f"Expected str, got {type(category)}"
    assert category in CATEGORY_MAP_INV, f"Unknown category: {category}"
    return CATEGORY_MAP_INV[category]


def volume_to_class_id(volume: float) -> int:
    # Determined by clustering.
    if volume < 9.5:
        return 0  # SMALL
    elif volume < 40:
        return 1  # MEDIUM
    return 2  # LARGE


class ClassType(enum.Enum):
    SEMANTIC = "SEMANTIC"
    VOLUME = "VOLUME"


CLASS_TYPE_DICT = {
    ClassType.SEMANTIC.value: category_to_class_id,
    ClassType.VOLUME.value: volume_to_class_id,
}


def get_ids_and_cuboids_at_lidar_timestamps(
    dataset: DensifiedAV2SensorDataLoader, log_id: str, lidar_timestamps_ns: list[int]
) -> list[dict[str, Cuboid]]:
    """Load the sweep annotations at the provided timestamp with unique ids.
    Args:
        log_id: Log unique id.
        lidar_timestamp_ns: Nanosecond timestamp.
    Returns:
        dict mapping ids to cuboids
    """
    annotations_feather_path = dataset._data_dir / log_id / "annotations.feather"

    # Load annotations from disk.
    # NOTE: This file contains annotations for the ENTIRE sequence.
    # The sweep annotations are selected below.
    cuboid_list = CuboidList.from_feather(annotations_feather_path)
    for cuboid in cuboid_list.cuboids:
        # Convert timestamp_ns from string to int if necessary
        if isinstance(cuboid.timestamp_ns, str):
            cuboid.timestamp_ns = int(cuboid.timestamp_ns)

    raw_data = read_feather(annotations_feather_path)
    ids = raw_data.track_uuid.to_numpy()

    cuboid_timestamps = [cuboid.timestamp_ns for cuboid in cuboid_list.cuboids]

    cuboids_and_ids_list = []
    for timestamp_ns in lidar_timestamps_ns:
        cuboids_and_ids = {
            id: cuboid
            for id, cuboid in zip(ids, cuboid_list.cuboids)
            if cuboid.timestamp_ns == timestamp_ns
        }
        cuboids_and_ids_list.append(cuboids_and_ids)

    return cuboids_and_ids_list


def compute_sceneflow(
    dataset: DensifiedAV2SensorDataLoader, log_id: str, timestamps: tuple[int, int], class_type: ClassType, velocity_aware: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sceneflow between the sweeps at the given timestamps.
    Args:
      dataset: Sensor dataset.
      log_id: unique id.
      timestamps: the timestamps of the lidar sweeps to compute flow between
      class_type: ClassType to use for the output classes
    Returns:
      dictionary with fields:
        pcl_0: Nx3 array containing the points at time 0
        pcl_1: Mx3 array containing the points at time 1
        flow_0_1: Nx3 array containing flow from timestamp 0 to 1
        classes_0: Nx1 array containing the class ids for each point in sweep 0
        valid_0: Nx1 array indicating if the returned flow from 0 to 1 is valid (1 for valid, 0 otherwise)
        ego_motion: SE3 motion from sweep 0 to sweep 1
    """

    def compute_flow(sweeps, cuboids, poses):
        ego1_SE3_ego0 = poses[1].inverse().compose(poses[0])
        # Convert to float32s
        ego1_SE3_ego0.rotation = ego1_SE3_ego0.rotation.astype(np.float32)
        ego1_SE3_ego0.translation = ego1_SE3_ego0.translation.astype(np.float32)

        flow_0_1 = np.zeros_like(sweeps[0].xyz, dtype=np.float32)

        valid_0 = np.ones(len(sweeps[0].xyz), dtype=bool)
        classes_0 = np.ones(len(sweeps[0].xyz), dtype=np.int8) * CATEGORY_MAP_INV["BACKGROUND"]

        for id in cuboids[0]:
            c0 = cuboids[0][id]
            c0.length_m += (
                0.2  # the bounding boxes are a little too tight and some points are missed
            )
            c0.width_m += 0.2
            obj_pts, obj_mask = c0.compute_interior_points(sweeps[0].xyz)

            match class_type:
                case ClassType.SEMANTIC:
                    classes_0[obj_mask] = CLASS_TYPE_DICT[class_type.value](c0.category)
                    # CATEGORY_MAP_INV[c0.category]
                case ClassType.VOLUME:
                    classes_0[obj_mask] = CLASS_TYPE_DICT[class_type.value](
                        c0.length_m * c0.width_m * c0.height_m
                    )

            if id in cuboids[1]:
                c1 = cuboids[1][id]
                c1_SE3_c0_ego_frame = ego1_SE3_ego0.inverse().compose(
                    c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
                )
                obj_flow = c1_SE3_c0_ego_frame.transform_point_cloud(obj_pts) - obj_pts
                flow_0_1[obj_mask] = obj_flow.astype(np.float32)
            else:
                valid_0[obj_mask] = 0
        return flow_0_1, classes_0, valid_0, ego1_SE3_ego0

    def compute_velocity_aware_flow(sweeps, cuboids, poses):
        ego1_SE3_ego0 = poses[1].inverse().compose(poses[0])
        # Convert to float32s
        ego1_SE3_ego0.rotation = ego1_SE3_ego0.rotation.astype(np.float32)
        ego1_SE3_ego0.translation = ego1_SE3_ego0.translation.astype(np.float32)

        flow_0_1 = np.zeros_like(sweeps[0].xyz, dtype=np.float32)
        valid_0 = np.ones(len(sweeps[0].xyz), dtype=bool)
        classes_0 = np.ones(len(sweeps[0].xyz), dtype=np.int8) * CATEGORY_MAP_INV["BACKGROUND"]

        for id in cuboids[0]:
            c0 = deepcopy(cuboids[0][id])
            obj_pts_npy, obj_mask_npy = c0.compute_interior_points(sweeps[0].xyz)

            # velocity-aware bbox expansion
            if id in cuboids[1]:
                c1 = cuboids[1][id]
                c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
                rel_obj_flow = c1_SE3_c0.transform_point_cloud(obj_pts_npy) - obj_pts_npy
                delta_move = abs(np.linalg.norm(rel_obj_flow, axis=0).mean())

                if delta_move > 0.04:  # moving faster than 0.4 m/s
                    c0 = cuboids[0][id]
                    c0.length_m += 0.2 + min(delta_move / 2, 2)
                    c0.width_m += 0.2
                    c0.height_m += 0.2
                    obj_pts_npy, obj_mask_npy = c0.compute_interior_points(sweeps[0].xyz)
                    c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
                obj_pts = obj_pts_npy
                obj_mask = obj_mask_npy
            else:
                valid_0[obj_mask_npy] = 0
                continue

            obj_pts = obj_pts.astype(np.float32)

            # Assign class
            match class_type:
                case ClassType.SEMANTIC:
                    classes_0[obj_mask] = CLASS_TYPE_DICT[class_type.value](c0.category)
                case ClassType.VOLUME:
                    classes_0[obj_mask] = CLASS_TYPE_DICT[class_type.value](
                        c0.length_m * c0.width_m * c0.height_m
                    )

            # Apply SE3 transform to get flow
            c1_SE3_c0_ego_frame = ego1_SE3_ego0.inverse().compose(c1_SE3_c0)
            obj_flow = c1_SE3_c0_ego_frame.transform_point_cloud(obj_pts) - obj_pts
            flow_0_1[obj_mask] = obj_flow.astype(np.float32)

        return flow_0_1, classes_0, valid_0, ego1_SE3_ego0



    sweeps = [Sweep.from_feather(dataset.get_lidar_fpath(log_id, ts)) for ts in timestamps]
    cuboids = get_ids_and_cuboids_at_lidar_timestamps(dataset, log_id, timestamps)
    poses = [dataset.get_city_SE3_ego(log_id, ts) for ts in timestamps]
    if velocity_aware:
        flow_0_1, classes_0, valid_0, _ = compute_velocity_aware_flow(sweeps, cuboids, poses)
    else:
        flow_0_1, classes_0, valid_0, _ = compute_flow(sweeps, cuboids, poses)
    return flow_0_1, classes_0, valid_0


def process_log(
    dataset: DensifiedAV2SensorDataLoader,
    log_id: str,
    output_dir: Path,
    class_type: ClassType,
    rollout_horizon: int = 1,
    crop_valid_to_scan0: bool = False,
    velocity_aware: bool = False,
    n: Optional[int] = None,
):
    """Outputs sceneflow and auxillary information for each pair of pointclouds in the
    dataset. Output files have the format <output_dir>/<log_id>_<sweep_1_timestamp>.npz
     Args:
       dataset: Sensor dataset to process.
       log_id: Log unique id.
       output_dir: Output_directory.
       class_type: ClassType to use for the output classes
       n: the position to use for the progress bar
     Returns:
       None
    """
    timestamps = dataset.get_ordered_log_lidar_timestamps(log_id)

    for i in tqdm(range(len(timestamps)), leave=False, position=n, desc=f"Log {log_id}"):
        ts0 = timestamps[i]

        multi_flow = {}
        valid_0 = None
        classes_0 = None

        for step in range(1, rollout_horizon + 1):
            if i + step >= len(timestamps):
                break  # Not enough future frames

            ts1 = timestamps[i + step]
            flow, classes, valid = compute_sceneflow(dataset, log_id, (ts0, ts1), class_type, velocity_aware)

            if step == 1:
                classes_0 = classes
                valid_0 = valid
                multi_flow["flow_tx_m"] = flow[:, 0].astype(np.float32)
                multi_flow["flow_ty_m"] = flow[:, 1].astype(np.float32)
                multi_flow["flow_tz_m"] = flow[:, 2].astype(np.float32)
            else:
                multi_flow[f"flow_tx_m_step{step}"] = flow[:, 0].astype(np.float32)
                multi_flow[f"flow_ty_m_step{step}"] = flow[:, 1].astype(np.float32)
                multi_flow[f"flow_tz_m_step{step}"] = flow[:, 2].astype(np.float32)

        if not multi_flow:
            continue

        df = pd.DataFrame(multi_flow)
        df["is_valid"] = valid_0
        df["classes_0"] = classes_0

        save_feather(output_dir / log_id / f"{ts0}.feather", df, verbose=False)


def process_log_wrapper(x, ignore_current_process=False):
    if not ignore_current_process:
        current = current_process()
        pos = current._identity[0]
    else:
        pos = 1
    process_log(*x, n=pos)


def process_logs(data_dir: Path, output_dir: Path, nproc: int, class_type: ClassType, rollout_horizon: int = 1, crop_valid_to_scan0: bool = False, velocity_aware: bool = False):
    """Compute sceneflow for all logs in the dataset. Logs are processed in parallel.
    Args:
      data_dir: Argoverse 2.0 directory
      output_dir: Output directory.
    """

    if not data_dir.exists():
        print(f"{data_dir} not found")
        return

    split_output_dir = output_dir
    split_output_dir.mkdir(exist_ok=True, parents=True)

    dataset = DensifiedAV2SensorDataLoader(data_dir=data_dir, labels_dir=data_dir)
    logs = dataset.get_log_ids()
    args = sorted([(dataset, log, split_output_dir, class_type, rollout_horizon, crop_valid_to_scan0, velocity_aware) for log in logs])

    print(f"Using {nproc} processes")
    if nproc <= 1:
        for x in tqdm(args):
            process_log_wrapper(x, ignore_current_process=True)
    else:
        with Pool(processes=nproc) as p:
            res = list(tqdm(p.imap_unordered(process_log_wrapper, args), total=len(logs)))

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = ArgumentParser(
        prog="create",
        description="Create a LiDAR sceneflow dataset from Argoveser 2.0 Sensor",
    )
    parser.add_argument(
        "--argo_dir",
        type=Path,
        required=True,
        help="The top level directory contating the input dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="The location to output the sceneflow files to",
    )
    parser.add_argument("--rollout_horizon", type=int, default=1)
    parser.add_argument("--nproc", type=int, default=(multiprocessing.cpu_count() - 1))
    parser.add_argument(
        "--class_type",
        type=str,
        default=ClassType.SEMANTIC.value,
        choices=[ClassType.SEMANTIC.value, ClassType.VOLUME.value],
    )
    parser.add_argument(
        "--crop_valid_to_scan0",
        action="store_true",
        help="Only mark original scan_0 points as valid, all densified points are set to invalid",
    )
    parser.add_argument(
        "--velocity_aware",
        action="store_true",
    )
    args = parser.parse_args()
    data_root = Path(args.argo_dir)
    output_dir = Path(args.output_dir)
    class_type = ClassType[args.class_type]
    crop_valid_to_scan0 = args.crop_valid_to_scan0
    velocity_aware = args.velocity_aware

    process_logs(data_root, output_dir, args.nproc, class_type, args.rollout_horizon, crop_valid_to_scan0, velocity_aware)
