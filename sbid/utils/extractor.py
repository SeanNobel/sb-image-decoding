import sys
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from termcolor import cprint
from datetime import timedelta
from typing import Optional

from nd.utils.vector2d import Vector2d


class FaceExtractor:
    def __init__(self, args, input_size: np.ndarray, quadrant: str, fps: int) -> None:
        if args.mode == "mesh":
            self.mp_solution = mp.solutions.face_mesh
            self.mp = self.mp_solution.FaceMesh(
                static_image_mode=False,  # NOTE: important to set this False
                max_num_faces=2,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            self.run = self.bbox_from_mesh

        elif args.mode == "detection":
            self.mp_solution = mp.solutions.face_detection
            self.mp = self.mp_solution.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5,
            )
            self.run = self.detection

        else:
            raise ValueError()

        self.quadrant = quadrant
        self.fps = fps

        self.input_height, self.input_width = input_size
        if not self.quadrant == "center":
            self.input_height //= 2
            self.input_width //= 2

        self.output_size = tuple([args.output_size] * 2)

        self.eyes_dist_mult = args.eyes_dist_mult
        self.movement_alert_thresh = args.movement_alert_thresh
        self.center_prev = np.zeros(2)

    def bbox_from_mesh(self, t: int, frame: np.ndarray) -> Optional[np.ndarray]:
        frame = self.crop_quadrant(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp.process(frame)

        if results.multi_face_landmarks:
            video_size = np.array([self.input_height, self.input_width])

            left_iris = self.filter_landmarks(
                results.multi_face_landmarks[0], self.mp_solution.FACEMESH_LEFT_IRIS
            ).mean(axis=0)
            right_iris = self.filter_landmarks(
                results.multi_face_landmarks[0], self.mp_solution.FACEMESH_RIGHT_IRIS
            ).mean(axis=0)
            # ( 2, ) x, y

            left_iris *= np.flip(video_size)
            right_iris *= np.flip(video_size)

            center = np.flip((left_iris + right_iris) / 2).astype(int)
            self.movement_alert(t, center=center)

            vectors = Vector2d((right_iris - left_iris)[np.newaxis])
            self.movement_alert_thresh = vectors.norm[0]
            # cprint(f"Angle: {vectors.angle[0]} | Norm: {vectors.norm[0]}", "yellow")

            # FIXME: These few lines could be used for rotation correction with some work
            # M = cv2.getRotationMatrix2D(tuple(center), vectors.angle[0], 1.0)
            # rotated_frame = cv2.warpAffine(frame, M, video_size.astype(int))
            # rotated_frame = rotated_frame.transpose(1, 0, 2)
            # print(frame.shape, rotated_frame.shape)

            crop_half = int(vectors.norm[0]) * self.eyes_dist_mult // 2
            frame = frame[
                center[0] - crop_half : center[0] + crop_half,
                center[1] - crop_half : center[1] + crop_half,
            ]

            try:
                frame = cv2.resize(frame, dsize=self.output_size)
            except:
                cprint("!ssize.empty() in function 'resize'", "yellow")
                return None

            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def detection(self, t: int, frame: np.ndarray) -> Optional[np.ndarray]:
        frame = self.crop_quadrant(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp.process(frame)

        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            self.movement_alert(t, bbox=bbox)

            frame = frame[
                int(bbox.ymin * self.input_height) : int(
                    (bbox.ymin + bbox.height) * self.input_height
                ),
                int(bbox.xmin * self.input_width) : int(
                    (bbox.xmin + bbox.width) * self.input_width
                ),
            ]

            frame = cv2.resize(frame, dsize=self.output_size)

            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def movement_alert(self, t, bbox=None, center=None) -> None:
        if center is None:
            assert bbox is not None, "Please pass either bbox or center to alert func."
            center = np.array(
                [
                    int((bbox.xmin + bbox.width / 2) * self.input_width),
                    int((bbox.ymin + bbox.height / 2) * self.input_height),
                ]
            )

        dist = np.linalg.norm(center - self.center_prev)
        # print(dist)

        if t > 0 and dist > self.movement_alert_thresh:
            cprint(
                f"Bounding box center moved more than {self.movement_alert_thresh} px from previous frame at frame {t}! Please go check around {timedelta(seconds=t//self.fps)}"
            )

        self.center_prev = center

    def crop_quadrant(self, frame):
        if self.quadrant == "center":
            pass
        elif self.quadrant == "left_up":
            frame = frame[: self.input_height, : self.input_width]
        elif self.quadrant == "right_up":
            frame = frame[self.input_height :, : self.input_width]
        elif self.quadrant == "left_down":
            frame = frame[: self.input_height, self.input_width :]
        elif self.quadrant == "right_down":
            frame = frame[self.input_height :, self.input_width :]
        else:
            raise ValueError("Please set the directory names as in README.md")

        return frame

    @staticmethod
    def filter_landmarks(landmarks_list, feature_filter) -> np.ndarray:
        filtered_landmarks = []

        landmark_num = len(landmarks_list.landmark)
        filtered_landmarks = [
            landmarks_list.landmark[i]
            for i in range(landmark_num)
            if i in list(sum(feature_filter, ()))
        ]
        filtered_landmarks_pb = landmark_pb2.NormalizedLandmarkList(
            landmark=filtered_landmarks
        )

        landmark_coordinates = [
            [landmark.x, landmark.y, landmark.z]
            for landmark in filtered_landmarks_pb.landmark
        ]

        return np.array(landmark_coordinates)[:, :2]
