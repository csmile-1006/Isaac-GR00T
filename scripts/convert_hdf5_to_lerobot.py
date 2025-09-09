import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

import av
import h5py
import jsonlines
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def convert_robocasa_state_to_gr00t(robocasa_state: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert RoboCasa state format to GR00T format."""
    STATE_KEY_MAPPING = {
        "robot0_base_pos": "base_position",
        "robot0_base_quat": "base_rotation",
        "robot0_eef_pos": "end_effector_position_absolute",
        "robot0_base_to_eef_pos": "end_effector_position_relative",
        "robot0_eef_quat": "end_effector_rotation_absolute",
        "robot0_base_to_eef_quat": "end_effector_rotation_relative",
        "robot0_gripper_qpos": "gripper_qpos",
        "robot0_gripper_qvel": "gripper_qvel",
        "robot0_joint_pos": "joint_position",
        "robot0_joint_pos_cos": "joint_position_cos",
        "robot0_joint_pos_sin": "joint_position_sin",
        "robot0_joint_vel": "joint_velocity",
    }

    states = []
    obs_len = len(robocasa_state[list(STATE_KEY_MAPPING.keys())[0]])

    for i in range(obs_len):
        state = np.concatenate([robocasa_state[key][i] for key in STATE_KEY_MAPPING.keys()], axis=0)
        states.append(state)

    return np.stack(states, axis=0)


def convert_robocasa_action_to_gr00t(robocasa_action: np.ndarray) -> np.ndarray:
    """Convert RoboCasa action format to GR00T format."""
    # Split action components
    ee_pos = robocasa_action[:, 0:3]
    ee_rot = robocasa_action[:, 3:6]
    gripper = robocasa_action[:, 6:7]
    base_motion = robocasa_action[:, 7:11]
    control_mode = robocasa_action[:, 11:12]

    # Normalize gripper and control mode to binary values
    gripper_binary = np.where(gripper < 0, 0, 1)
    control_mode_binary = np.where(control_mode < 0, 0, 1)

    # Combine components in GR00T order
    return np.concatenate([base_motion, control_mode_binary, ee_pos, ee_rot, gripper_binary], axis=1)


def append_jsonlines(data: dict, filepath: Path) -> None:
    """Append data to a JSONL file."""
    filepath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(filepath, "a") as writer:
        writer.write(data)


def write_episode(episode: dict, output_dir: Path) -> None:
    """Write episode data to episodes.jsonl."""
    append_jsonlines(episode, output_dir / "meta" / "episodes.jsonl")


def write_json(data: dict, filepath: Path) -> None:
    """Write data to a JSON file."""
    filepath.parent.mkdir(exist_ok=True, parents=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_info(info: dict, output_dir: Path) -> None:
    """Write info data to info.json."""
    write_json(info, output_dir / "meta" / "info.json")


def write_modality_json(modality_config: dict, output_dir: Path) -> None:
    """Write modality data to modality.json."""
    write_json(modality_config, output_dir / "meta" / "modality.json")


def encode_video_worker(
    args: Tuple[np.ndarray, str, int, str, str, Optional[int], Optional[int], int, Optional[int], bool],
) -> Tuple[str, bool]:
    """
    Worker function for video encoding that can be used with multiprocessing.

    Args:
        args: Tuple containing (frames, output_path, fps, vcodec, pix_fmt, g, crf, fast_decode, log_level, overwrite)

    Returns:
        Tuple of (output_path, success)
    """
    frames, output_path, fps, vcodec, pix_fmt, g, crf, fast_decode, log_level, overwrite = args

    try:
        encode_video_frames(
            frames=frames,
            output_path=output_path,
            fps=fps,
            vcodec=vcodec,
            pix_fmt=pix_fmt,
            g=g,
            crf=crf,
            fast_decode=fast_decode,
            log_level=log_level,
            overwrite=overwrite,
        )
        return str(output_path), True
    except OSError as e:
        logging.error("Failed to encode video %s: %s", output_path, e)
        return str(output_path), False


def encode_video_frames(
    frames: np.ndarray,
    output_path: Union[Path, str],
    fps: int,
    vcodec: str = "h264",
    pix_fmt: str = "yuv420p",
    g: Optional[int] = 4,
    crf: Optional[int] = 23,
    fast_decode: int = 0,
    log_level: Optional[int] = av.logging.ERROR,
    overwrite: bool = False,
) -> None:
    """
    Encode video frames using ffmpeg.

    Args:
        frames: Array of video frames
        output_path: Path to save encoded video
        fps: Frames per second
        vcodec: Video codec (h264, hevc, or libsvtav1)
        pix_fmt: Pixel format
        g: GOP size
        crf: Constant Rate Factor (quality)
        fast_decode: Enable fast decoding
        log_level: Logging level
        overwrite: Whether to overwrite existing file
    """
    SUPPORTED_CODECS = ["h264", "hevc", "libsvtav1"]
    if vcodec not in SUPPORTED_CODECS:
        raise ValueError(f"Unsupported codec: {vcodec}. Must be one of {SUPPORTED_CODECS}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=overwrite)

    # Handle codec/format incompatibilities
    if (vcodec in ["libsvtav1", "hevc"]) and pix_fmt == "yuv444p":
        logging.warning("Format 'yuv444p' incompatible with %s, using 'yuv420p'", vcodec)
        pix_fmt = "yuv420p"

    if frames is None or len(frames) == 0:
        raise ValueError("No frames provided")

    # Get dimensions from first frame
    frame = Image.fromarray(frames[0])
    width, height = frame.size

    # Configure encoder
    video_options = {}
    if g is not None:
        video_options["g"] = str(g)
    if crf is not None:
        video_options["crf"] = str(crf)
    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    # Set logging
    if log_level is not None:
        logging.getLogger("libav").setLevel(log_level)

    # Encode frames
    with av.open(str(output_path), "w") as output:
        stream = output.add_stream(vcodec, fps, options=video_options)
        stream.pix_fmt = pix_fmt
        stream.width = width
        stream.height = height

        for frame_data in frames:
            frame_img = Image.fromarray(frame_data).convert("RGB")
            frame = av.VideoFrame.from_image(frame_img)
            packet = stream.encode(frame)
            if packet:
                output.mux(packet)

        # Flush encoder
        packet = stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging
    if log_level is not None:
        av.logging.restore_default_callback()

    if not output_path.exists():
        raise OSError(f"Video encoding failed - file not found: {output_path}")


class LeRobotFormatter:
    """Converts RoboCasa HDF5 format to RoboLab format."""

    def __init__(
        self,
        root: str,
        task_name: str,
        chunks_size: int,
        src_hdf5_path: str,
        meta_path: str,
        num_video_workers: int = 4,
    ):
        self.root = Path(root)
        self.task_name = task_name
        self.chunks_size = chunks_size
        self.num_video_workers = num_video_workers
        self.tasks = {}
        self.task_to_task_index = {}
        self.episodes = {}
        self.info = {}
        self.total_frames = 0
        self.episode_index = 0

        # Video encoding management
        self.video_encoding_tasks = []
        self.video_encoding_results = {}
        self.video_encoding_lock = threading.Lock()

        self.hf_features = [
            "observation.state",
            "action",
            "timestamp",
            "annotation.human.action.task_description",
            "task_index",
            "annotation.human.action.task_name",
            "annotation.human.validity",
            "episode_index",
            "index",
            "next.reward",
            "next.done",
        ]

        self.video_key_mapping = {
            "robot0_agentview_left_image": "observation.images.left_view",
            "robot0_agentview_right_image": "observation.images.right_view",
            "robot0_eye_in_hand_image": "observation.images.wrist_view",
        }
        self.reverse_video_mapping = {v: k for k, v in self.video_key_mapping.items()}

        # Set up HDF5 source
        self.src_hdf5_path = Path(src_hdf5_path)
        self.src_hdf5_file = h5py.File(src_hdf5_path, "r")
        self.src_data = self.src_hdf5_file["data"]

        # Create output directory structure
        self._setup_output_dirs()
        self.meta_path = Path(meta_path)
        self._copy_meta_files()
        self._init_paths()

    def _setup_output_dirs(self):
        """Create output directory structure."""
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in ["meta", "data", "videos"]:
            (self.root / folder).mkdir(exist_ok=True)

    def _copy_meta_files(self):
        """Copy metadata files."""
        shutil.copy(self.meta_path / "info.json", self.root / "meta" / "info.json")
        with open(self.meta_path / "modality.json", "r", encoding="utf-8") as f:
            modality_config = json.load(f)
        # update modality config w/ rewards and done keys
        modality_config["reward"] = {"next.reward": {}}
        modality_config["done"] = {"next.done": {}}
        modality_config["next_state"] = modality_config["state"]
        modality_config["next_video"] = modality_config["video"]
        write_modality_json(modality_config, self.root)

        self.load_info_file()

    def _init_paths(self):
        """Initialize data and video paths."""
        self.data_path = self.info["data_path"]
        self.video_path = self.info["video_path"]

    @property
    def video_keys(self) -> List[str]:
        """Get list of video feature keys."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def image_keys(self) -> List[str]:
        """Get list of image feature keys."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def fps(self) -> int:
        """Get frames per second."""
        return self.info["fps"]

    @property
    def features(self) -> dict:
        """Get feature definitions."""
        return self.info["features"]

    @property
    def num_episodes(self) -> int:
        """Get number of episodes."""
        return len(self.episodes) if self.episodes is not None else 0

    def create_episode_buffer(self, episode_index: Optional[int] = None) -> dict:
        """Create buffer for episode data."""
        buffer = {
            "size": 0,
            "task": [],
        }
        for key in self.info["features"]:
            buffer[key] = episode_index if key == "episode_index" else []
        return buffer

    def load_info_file(self):
        """Load and initialize info file."""
        with open(self.root / "meta" / "info.json", "r", encoding="utf-8") as f:
            self.info = json.load(f)

        # Initialize counters
        for counter in ["episodes", "frames", "chunks", "tasks"]:
            self.info[f"total_{counter}"] = 0

    def get_timestamp(self, episode_length: int) -> np.ndarray:
        """Generate timestamps for episode frames."""
        return np.array([i / self.fps for i in range(episode_length)])

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get chunk index for episode."""
        return ep_index // self.chunks_size

    def get_data_file_path(self, ep_index: int) -> Path:
        """Get path for episode data file."""
        chunk = self.get_episode_chunk(ep_index)
        return Path(self.data_path.format(episode_chunk=chunk, episode_index=ep_index))

    def get_video_file_path(self, ep_index: int, video_key: str) -> Path:
        """Get path for episode video file."""
        chunk = self.get_episode_chunk(ep_index)
        return Path(self.video_path.format(episode_chunk=chunk, episode_index=ep_index, video_key=video_key))

    def get_task_index(self, task: str) -> Optional[int]:
        """Get index for task."""
        return self.task_to_task_index.get(task)

    def add_task(self, task: str):
        """Add new task to task list."""
        if task in self.task_to_task_index:
            raise ValueError(f"Task '{task}' already exists")

        task_index = self.info["total_tasks"]
        self.task_to_task_index[task] = task_index
        self.tasks[task_index] = task
        self.info["total_tasks"] += 1

        append_jsonlines(
            {
                "task_index": task_index,
                "task": task,
            },
            self.root / "meta" / "tasks.jsonl",
        )

    def convert_episode_to_lerobot_format(self, demo: dict) -> dict:
        """Convert a single episode from RoboCasa to RoboLab format."""
        # Get task instruction
        task_inst = json.loads(demo.attrs["ep_meta"])["lang"]

        # Initialize episode buffer
        buffer = self.create_episode_buffer(self.episode_index)
        episode_length = len(demo["actions"])
        buffer["size"] = episode_length

        # Create task arrays
        tasks = [task_inst] * buffer["size"]
        task_names = [self.task_name] * buffer["size"]
        validities = ["Valid"] * buffer["size"]

        # Convert state and actions
        buffer["observation.state"] = convert_robocasa_state_to_gr00t(demo["obs"])
        buffer["action"] = convert_robocasa_action_to_gr00t(demo["actions"])
        buffer["timestamp"] = self.get_timestamp(len(demo["actions"]))

        # Add indices
        buffer["index"] = np.arange(self.total_frames, self.total_frames + len(demo["actions"]))
        buffer["episode_index"] = np.full(buffer["size"], self.episode_index)

        # Add rewards and dones
        buffer["next.done"] = demo["dones"][:]
        buffer["next.reward"] = demo["rewards"][:]

        # Process tasks
        for task in set(tasks + task_names + validities):
            if self.get_task_index(task) is None:
                self.add_task(task)

        # Add task indices
        buffer["task_index"] = np.array([self.get_task_index(task) for task in tasks])
        buffer["annotation.human.action.task_description"] = buffer["task_index"]
        buffer["annotation.human.action.task_name"] = np.array([self.get_task_index(name) for name in task_names])
        buffer["annotation.human.validity"] = np.array([self.get_task_index(validity) for validity in validities])

        # Handle videos if present
        if self.video_keys:
            video_paths = self.queue_episode_videos(demo, self.episode_index)
            buffer.update(video_paths)

        # Save episode data
        self._save_episode_table(buffer, self.episode_index)
        self._write_episode_metadata(task_inst, episode_length)
        self._update_episode_info(buffer["size"])

        # Verify files
        self._verify_files()

        return buffer

    def _save_episode_table(self, buffer: dict, episode_index: int):
        """Save episode data to parquet file."""
        episode_dict = {key: buffer[key].tolist() for key in self.hf_features}
        df = pd.DataFrame.from_dict(episode_dict)
        path = self.root / self.get_data_file_path(episode_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    def _update_episode_info(self, episode_size: int):
        """Update episode counters."""
        chunk = self.get_episode_chunk(self.episode_index)
        if chunk >= self.info["total_chunks"]:
            self.info["total_chunks"] += 1

        self.episode_index += 1
        self.total_frames += episode_size
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_size

        write_info(self.info, self.root)

    def _write_episode_metadata(self, task_inst: str, episode_length: int):
        """Write episode metadata."""
        episode_dict = {
            "episode_index": self.episode_index,
            "episode_tasks": [task_inst, self.task_name, "Valid"],
            "length": episode_length,
        }
        self.episodes[self.episode_index] = episode_dict
        write_episode(episode_dict, self.root)

    def _verify_files(self):
        """Verify all expected files exist."""
        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

    def queue_episode_videos(self, demo: dict, episode_index: int) -> Dict[str, str]:
        """Queue episode videos for encoding."""
        video_paths = {}
        for key in self.video_keys:
            video_info = self.features[key]["video_info"]
            video_path = self.root / self.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)

            if video_path.is_file():
                continue

            images = demo["obs"][self.reverse_video_mapping[key]]

            # Queue video encoding task
            task_args = (
                images,
                str(video_path),
                video_info["video.fps"],
                video_info["video.codec"],
                video_info["video.pix_fmt"],
                4,  # g
                23,  # crf
                0,  # fast_decode
                av.logging.ERROR,  # log_level
                True,  # overwrite
            )

            with self.video_encoding_lock:
                self.video_encoding_tasks.append((episode_index, key, task_args))

        return video_paths

    def process_video_encoding_queue(self):
        """Process all queued video encoding tasks using multiprocessing."""
        if not self.video_encoding_tasks:
            return

        logging.info(
            "Processing %d video encoding tasks with %d workers", len(self.video_encoding_tasks), self.num_video_workers
        )

        # Create a mapping from task args to episode/key for result tracking
        task_mapping = {}
        task_args_list = []

        for episode_index, key, task_args in self.video_encoding_tasks:
            task_mapping[task_args] = (episode_index, key)
            task_args_list.append(task_args)

        # Process videos in parallel
        with ProcessPoolExecutor(max_workers=self.num_video_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(encode_video_worker, task_args): task_args for task_args in task_args_list
            }

            # Collect results with progress bar
            with tqdm(total=len(future_to_task), desc="Encoding videos") as pbar:
                for future in as_completed(future_to_task):
                    task_args = future_to_task[future]
                    episode_index, key = task_mapping[task_args]

                    try:
                        output_path, success = future.result()
                        with self.video_encoding_lock:
                            self.video_encoding_results[(episode_index, key)] = (output_path, success)

                        if not success:
                            logging.error("Failed to encode video for episode %d, key %s", episode_index, key)

                    except Exception as e:
                        logging.error("Video encoding task failed: %s", e)
                        with self.video_encoding_lock:
                            self.video_encoding_results[(episode_index, key)] = (None, False)

                    pbar.update(1)

        # Clear the queue
        self.video_encoding_tasks.clear()

    def convert_hdf5_to_lerobot_format(self, num_episodes: Optional[int] = None):
        """Convert all episodes from HDF5 to RoboLab format."""
        demo_list = sorted(self.src_data.keys(), key=lambda x: int(x.split("_")[-1]))

        # Phase 1: Process all episodes (non-video data) sequentially
        logging.info("Phase 1: Processing episodes (state, actions, metadata)...")
        for demo_idx in tqdm(demo_list[:num_episodes], desc="Processing episodes"):
            demo = self.src_data[demo_idx]
            self.convert_episode_to_lerobot_format(demo)

        # Phase 2: Process all video encoding tasks in parallel
        if self.video_keys:
            logging.info("Phase 2: Encoding videos in parallel...")
            self.process_video_encoding_queue()

            # Verify all videos were encoded successfully
            failed_videos = []
            for (episode_index, key), (output_path, success) in self.video_encoding_results.items():
                if not success:
                    failed_videos.append((episode_index, key, output_path))

            if failed_videos:
                logging.error("Failed to encode %d videos:", len(failed_videos))
                for episode_index, key, output_path in failed_videos:
                    logging.error("  Episode %d, key %s: %s", episode_index, key, output_path)
                raise RuntimeError("Some videos failed to encode")

            logging.info("All videos encoded successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_hdf5_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--chunks_size", type=int, default=300)
    parser.add_argument("--meta_path", type=str, default="meta/info.json")
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument(
        "--num_video_workers", type=int, default=4, help="Number of parallel workers for video encoding"
    )
    args = parser.parse_args()

    formatter = LeRobotFormatter(
        root=args.output_path,
        task_name=args.task_name,
        chunks_size=args.chunks_size,
        src_hdf5_path=args.src_hdf5_path,
        meta_path=args.meta_path,
        num_video_workers=args.num_video_workers,
    )
    formatter.convert_hdf5_to_lerobot_format(args.num_episodes)
