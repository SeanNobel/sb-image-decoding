import sys
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from tqdm import tqdm
from termcolor import cprint
import matplotlib.pyplot as plt
from typing import Optional

from utils.tilt_correction import Vector2d


# FIXME: use distance & tilt invariant method for eyes extraction
def face_preproc(args, video_path, video_times_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    video_size = np.array(
        [cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
    )
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    y_times = np.loadtxt(video_times_path, delimiter=",")  # ( 109800, )

    y_list = []
    no_face_idxs = []
    i = 0
    first_frame = None

    pbar = tqdm(total=num_frames)
    while cap.isOpened():
        ret, _frame = cap.read()

        if not ret:
            cprint("Not ret", color="yellow")
            break

        # frame = extract_eyes_then_grad(
        #     face_mesh, mp_face_mesh, _frame, video_size, args.image_size
        # )
        frame = bbox_from_mesh(
            face_mesh, mp_face_mesh, _frame, video_size, args.image_size
        )

        if frame is not None:
            y_list.append(frame)

            if first_frame is None:
                first_frame = _frame.copy()

        else:
            if i < len(y_times):
                no_face_idxs.append(i)

        # DEBUG
        # y_list.append(np.random.rand(32, 32))

        i += 1
        pbar.update(1)

    cap.release()

    segment_len = args.seq_len * fps  # 90

    y = np.stack(y_list)  # ( 109797, 256, 256 )

    y = y[: -(y.shape[0] % segment_len)]  # ( 109710, 32, 32 )
    y = y.reshape(-1, segment_len, 1, *y.shape[-2:])  # ( 1219, 90, 1, 32, 32 )

    y_times = np.delete(y_times, no_face_idxs)
    y_times = y_times[::segment_len][: y.shape[0]]  # ( 1219, )

    assert len(y) == len(y_times)

    return y, y_times, first_frame


def bbox_from_mesh(
    face_mesh,
    mp_face_mesh,
    eyes_dist_mult: int,
    frame: np.ndarray,
    video_size: np.ndarray,
    output_size: tuple,
) -> Optional[np.ndarray]:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(frame)

    if results.multi_face_landmarks:
        left_iris = filter_landmarks(
            results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_LEFT_IRIS
        ).mean(axis=0)
        right_iris = filter_landmarks(
            results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_RIGHT_IRIS
        ).mean(axis=0)
        # ( 2, ) x, y

        left_iris *= video_size
        right_iris *= video_size

        center = np.flip((left_iris + right_iris) / 2).astype(int)
        # self.movement_alert(t, center=center)

        vectors = Vector2d((right_iris - left_iris)[np.newaxis])
        # self.movement_alert_thresh = vectors.norm[0]
        # cprint(f"Angle: {vectors.angle[0]} | Norm: {vectors.norm[0]}", "yellow")

        # FIXME: These few lines could be used for rotation correction with some work
        # M = cv2.getRotationMatrix2D(tuple(center), vectors.angle[0], 1.0)
        # rotated_frame = cv2.warpAffine(frame, M, video_size.astype(int))
        # rotated_frame = rotated_frame.transpose(1, 0, 2)
        # print(frame.shape, rotated_frame.shape)

        crop_half = int(vectors.norm[0]) * eyes_dist_mult // 2
        frame = frame[
            center[0] - crop_half : center[0] + crop_half,
            center[1] - crop_half : center[1] + crop_half,
        ]

        frame = cv2.resize(frame, dsize=output_size)

        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def extract_eyes_then_grad(face_mesh, mp_face_mesh, frame, video_size, image_size):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(frame)

    if results.multi_face_landmarks:
        right_eye = get_eye(frame, results, video_size, mp_face_mesh.FACEMESH_RIGHT_IRIS)
        left_eye = get_eye(frame, results, video_size, mp_face_mesh.FACEMESH_LEFT_IRIS)

        try:
            right_eye = cv2.resize(right_eye, dsize=(image_size // 2, image_size))
        except:
            print(right_eye)
            sys.exit()
        left_eye = cv2.resize(left_eye, dsize=(image_size // 2, image_size))

        frame = np.concatenate((right_eye, left_eye), axis=1)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        dx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(dx**2 + dy**2).astype(np.uint8)

        return grad


def get_eye(frame, results, video_size, mesh):
    iris = filter_landmarks(results.multi_face_landmarks[0], mesh)
    # ( 4, 2 ) NOTE 0: right, 1: down, 2: left, 3: up

    center = iris.mean(axis=0) * video_size  # .astype(int)

    vectors = Vector2d(np.array([iris[0] - iris[2], iris[1] - iris[3]]))

    M = cv2.getRotationMatrix2D(tuple(center), 180 - vectors.angle[0], 1.0)
    rotated_frame = cv2.warpAffine(frame, M, video_size.astype(int))

    # NOTE: 2 is actually a hyperparameter
    crop_half = int(vectors.norm[0] * video_size[0]) * 2

    center = center.astype(int)
    eye = rotated_frame[
        center[1] - crop_half : center[1] + crop_half,
        center[0] - crop_half : center[0] + crop_half,
    ]

    return eye


# From Elie
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


if __name__ == "__main__":
    from configs.args import args
