from typing import Any, Dict, List

import numpy as np


def loads(string: str):
    buffer: List[str] = string.splitlines()
    buffer.reverse()

    skeleton_sequence = {}
    skeleton_sequence["numFrames"] = int(buffer.pop())
    skeleton_sequence["frames"] = []
    for _ in range(skeleton_sequence["numFrames"]):
        frame_info = {}
        frame_info["numBodies"] = int(buffer.pop())
        frame_info["bodies"] = []

        for _ in range(frame_info["numBodies"]):
            body_info = {}
            body_info_key = [
                "bodyID",
                "clipedEdges",
                "handLeftConfidence",
                "handLeftState",
                "handRightConfidence",
                "handRightState",
                "isResticted",
                "leanX",
                "leanY",
                "trackingState",
            ]
            body_info = {}
            for key, value in zip(body_info_key, buffer.pop().split()):
                if key == "bodyID":
                    continue

                if key in ["leanX", "leanY"]:
                    body_info[key] = float(value)
                else:
                    body_info[key] = int(value)

            body_info["numJoint"] = int(buffer.pop())
            body_info["joints"] = []
            for _ in range(body_info["numJoint"]):
                joint_info_key = [
                    "cameraX",
                    "cameraY",
                    "cameraZ",
                    "depthX",
                    "depthY",
                    "colorX",
                    "colorY",
                    "orientationW",
                    "orientationX",
                    "orientationY",
                    "orientationZ",
                    "trackingState",
                ]
                joint_info: Dict[str, float] = {}
                for key, value in zip(joint_info_key, buffer.pop().split()):
                    if key in "trackingState":
                        joint_info[key] = int(value)
                    else:
                        joint_info[key] = float(value)

                body_info["joints"].append(joint_info)

            frame_info["bodies"].append(body_info)

        skeleton_sequence["frames"].append(frame_info)

    return skeleton_sequence


def as_numpy(
    skeleton_sequence: Dict[str, Any],
    joint_keys: List[str] = ["cameraX", "cameraY", "cameraZ"],
):
    num_frames: int = skeleton_sequence["numFrames"]

    max_bodies: int = 0
    for frame in skeleton_sequence["frames"]:
        max_bodies: int = max(frame["numBodies"], max_bodies)

    num_joints: int = 25

    buffer = np.zeros(
        (max_bodies, num_frames, num_joints, len(joint_keys)), dtype=np.float32
    )

    for t, frame in enumerate(skeleton_sequence["frames"]):
        for m, body in enumerate(frame["bodies"]):
            joints = []
            for joint in body["joints"]:
                joints.append([joint[key] for key in joint_keys])

            assert not np.isnan(joints).any(), print(joints, np.isnan(joints).any())
            assert np.sum(np.abs(joints)) != 0

            buffer[m, t] = joints

    return buffer
