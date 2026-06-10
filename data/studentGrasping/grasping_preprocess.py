import argparse
import trimesh
import pathlib
import os
import numpy as np
import mujoco
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import pybullet


def preprocess_sample(args):
    scale_dir, manifold_bin, simplify_bin, max_num_vertices, center_by_com = args

    # Pid-based unique tmp files.
    pid = os.getpid()
    tmp_dir = pathlib.Path("/tmp/preprocess_student_grasp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_mani_path = tmp_dir / f"mani_{pid}.obj"
    tmp_simp_path = tmp_dir / f"simp_{pid}.obj"
    tmp_mani_path.unlink(missing_ok=True)
    tmp_simp_path.unlink(missing_ok=True)
    try:
        # Ensure mesh is watertight.
        subprocess.run(
            [manifold_bin, str(scale_dir / "mesh.obj"), str(tmp_mani_path)],
            capture_output=True,
        )
        if not tmp_mani_path.is_file():
            return None
        # Simplify mesh.
        subprocess.run(
            [
                simplify_bin,
                "-i",
                str(tmp_mani_path),
                "-o",
                str(tmp_simp_path),
                "-m",
                "-f",
                f"{max_num_vertices}",
            ],
            capture_output=True,
        )
        if not tmp_simp_path.is_file():
            return None

        # Center mesh.
        mesh = trimesh.load(tmp_simp_path)
        object_center_pos = (
            mesh.center_mass if center_by_com else mesh.bounding_box.centroid
        )
        mesh.apply_translation(-object_center_pos)

        # Transform grasps into hand frame.
        data = np.load(scale_dir / "recording.npz")
        sorted_idx = np.argsort(data["scores"])[::-1]
        scores = data["scores"][sorted_idx]
        grasp_data = data["grasps"]
        joint_angles = grasp_data[sorted_idx, 7:]
        hand_pos = grasp_data[sorted_idx, :3]
        hand_rot = Rotation.from_quat(grasp_data[sorted_idx, 3:7], scalar_first=False)
        object_pos = hand_rot.inv().apply(object_center_pos - hand_pos)
        object_rot = hand_rot.inv()

        return (
            mesh,
            joint_angles,
            object_pos,
            object_rot.as_quat(scalar_first=False),
            scores,
        )

    except Exception as e:
        print(f"Error processing {scale_dir}: {e}")
        return None


def preprocess_dataset(
    base_dir, output_dir, manifold_dir, max_num_vertices, center_by_com=False, max_meshes=None
):
    pathlib.Path.mkdir(output_dir, exist_ok=True)
    manifold_bin = str(manifold_dir / "manifold")
    simplify_bin = str(manifold_dir / "simplify")

    # Collect all meshes to process.
    scale_dirs = []
    for category_dir in [p for p in base_dir.iterdir() if p.is_dir()]:
        for object_dir in [p for p in category_dir.iterdir() if p.is_dir()]:
            for scale_dir in [p for p in object_dir.iterdir() if p.is_dir()]:
                scale_dirs.append(scale_dir)
    if max_meshes is not None:
        scale_dirs = scale_dirs[:max_meshes]

    # Preprocess all meshes.
    args_list = [
        (scale_dir, manifold_bin, simplify_bin, max_num_vertices, center_by_com)
        for scale_dir in scale_dirs
    ]
    num_meshes = 0
    joint_angles_list = []
    object_pos_list = []
    object_rot_list = []
    scores_list = []
    with Pool(processes=os.cpu_count() - 1) as pool:
        results = pool.imap_unordered(preprocess_sample, args_list)
        for result in tqdm(results, total=len(args_list)):
            if result is None:
                continue
            # Output.
            mesh, joint_angles, object_pos, object_rot, scores = result
            mesh.export(output_dir / f"{num_meshes:07d}.obj")
            joint_angles_list.append(joint_angles)
            object_pos_list.append(object_pos)
            object_rot_list.append(object_rot)
            scores_list.append(scores)
            num_meshes += 1
    np.savez(
        output_dir / "dataset.npz",
        joint_angles=np.stack(joint_angles_list),
        object_pos=np.stack(object_pos_list),
        object_rot=np.stack(object_rot_list),
        scores=np.stack(scores_list),
    )
    print(f"Saved {num_meshes} meshes.")


def visualize_dataset(dataset_dir, urdf_path):
    # Load dataset.
    dataset = np.load(dataset_dir / "dataset.npz")
    joint_angles_all = dataset["joint_angles"]
    object_pos_all = dataset["object_pos"]
    object_rot_all = dataset["object_rot"]
    scores_all = dataset["scores"]

    # Load hand.
    pybullet.connect(pybullet.GUI)
    hand_id = pybullet.loadURDF(
        str(urdf_path),
        globalScaling=1,
        basePosition=[0, 0, 0],
        baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
        flags=pybullet.URDF_MAINTAIN_LINK_ORDER,
    )
    object_id = None
    while input("q = exit: ") != "q":
        # Remove old object.
        if object_id is not None:
            pybullet.removeBody(object_id)
        # Sample object and grasp.
        object_idx = np.random.randint(low=0, high=joint_angles_all.shape[0])
        grasp_idx = np.random.randint(low=0, high=joint_angles_all.shape[1])
        # Load object
        visualShapeId = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=str(dataset_dir / f"{object_idx:07d}.obj"),
            rgbaColor=[1, 1, 1, 1],
            specularColor=[0.4, 0.4, 0],
            visualFramePosition=[0, 0, 0],
            meshScale=1,
        )
        object_id = pybullet.createMultiBody(
            baseMass=1,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=visualShapeId,
            baseCollisionShapeIndex=visualShapeId,
            basePosition=object_pos_all[object_idx, grasp_idx],
            baseOrientation=object_rot_all[object_idx, grasp_idx],
        )
        # Set joint angles
        grasp = joint_angles_all[object_idx, grasp_idx]
        for k, j in enumerate([1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21]):
            pybullet.resetJointState(
                hand_id, jointIndex=j, targetValue=grasp[k], targetVelocity=0
            )
            # Set coupled joint
            if j in [3, 9, 15, 21]:
                pybullet.resetJointState(
                    hand_id,
                    jointIndex=j + 1,
                    targetValue=grasp[k],
                    targetVelocity=0,
                )
        print(f"Grasping score: {scores_all[object_idx, grasp_idx]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess student grasp dataset.")
    parser.add_argument(
        "base_dir", type=pathlib.Path, help="Root directory of the raw dataset."
    )
    parser.add_argument(
        "output_dir", type=pathlib.Path, help="Directory to write preprocessed data."
    )
    parser.add_argument(
        "manifold_dir",
        type=pathlib.Path,
        help="Directory containing the manifold and simplify binaries.",
    )
    parser.add_argument(
        "urdf_path",
        type=pathlib.Path,
        help="Path to the hand URDF file.",
    )
    parser.add_argument(
        "--max-num-vertices",
        type=int,
        default=2048,
        help="Maximum number of vertices after mesh simplification.",
    )
    parser.add_argument(
        "--center-by-com",
        action="store_true",
        help="Center meshes by center of mass instead of bounding box centroid.",
    )
    parser.add_argument(
        "--max-meshes",
        type=int,
        default=None,
        help="Maximum number of meshes to preprocess.",
    )
    args = parser.parse_args()

    preprocess_dataset(
        base_dir=pathlib.Path(args.base_dir),
        output_dir=pathlib.Path(args.output_dir),
        manifold_dir=pathlib.Path(args.manifold_dir),
        max_num_vertices=args.max_num_vertices,
        center_by_com=args.center_by_com,
        max_meshes=args.max_meshes,
    )

    visualize_dataset(
        dataset_dir=pathlib.Path(args.output_dir),
        urdf_path=pathlib.Path(args.urdf_path),
    )
