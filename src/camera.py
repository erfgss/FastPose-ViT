# This file contains the camera parameters for the datasets. In our case we just have SPEED, so we only define one camera.

import torch


class Camera:
    """
    Data class based on UrsoNet's implementation:
    https://github.com/pedropro/UrsoNet/blob/8e59d9b81dd3805aba1d773bd9b44f1a33745b05/speed.py
    Parameters verified with the original SPEED dataset paper:
    https://arxiv.org/pdf/1906.09868
    SPEED+ uses the dataset-local camera.json calibration.
    """

    # Focal length[m]
    fwx = 0.0176
    fwy = 0.0176
    # Image size [pixels]
    width = 1920
    height = 1200
    # Size of the pixels [m / pixel]
    ppx = 5.86e-6
    ppy = ppx

    # Focal length[pixels]
    fx = fwx / ppx
    fy = fwy / ppy

    # Intrinsics matrix
    K = torch.tensor([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])
    # Inverse of the intrinsics matrix
    K_inv = torch.inverse(K)

    # Centered intrinsics matrix
    K_c = torch.tensor([[fx, 0, 0], [0, fy, 0], [0, 0, 1]])
    # Inverse of the centered intrinsics matrix
    K_c_inv = torch.inverse(K_c)

    # Camera aspect ratio
    aspect_ratio = width / height

    # Pre-computed constants for performance optimization
    width_height_product = width * height
    width_height_over_aspect = width_height_product / aspect_ratio

    # Cache commonly used values for translation computations
    cx = width / 2  # Image center x
    cy = height / 2  # Image center y


    @classmethod
    def from_json(cls, path):
        """Load SPEED+ pixel intrinsics; retain distortion metadata without resampling."""
        import json
        import math
        from pathlib import Path
        data = json.loads(Path(path).read_text())
        camera = cls()
        camera.width, camera.height = int(data["Nu"]), int(data["Nv"])
        camera.K = torch.tensor(data["cameraMatrix"], dtype=torch.float32)
        if camera.K.shape != (3, 3) or not torch.isfinite(camera.K).all():
            raise ValueError("Invalid cameraMatrix in calibration")
        camera.fx = float(data["cameraMatrix"][0][0])
        camera.fy = float(data["cameraMatrix"][1][1])
        camera.cx = float(data["cameraMatrix"][0][2])
        camera.cy = float(data["cameraMatrix"][1][2])
        if min(camera.fx, camera.fy, camera.width, camera.height) <= 0:
            raise ValueError("Camera focal lengths and dimensions must be positive")
        camera.ppx, camera.ppy = float(data["ppx"]), float(data["ppy"])
        camera.fwx, camera.fwy = camera.fx * camera.ppx, camera.fy * camera.ppy
        camera.K_inv = torch.inverse(camera.K)
        camera.K_c = torch.diag(torch.tensor([camera.fx, camera.fy, 1.]))
        camera.K_c_inv = torch.inverse(camera.K_c)
        camera.aspect_ratio = camera.width / camera.height
        camera.width_height_product = camera.width * camera.height
        camera.width_height_over_aspect = camera.width_height_product / camera.aspect_ratio
        camera.dist_coeffs = list(data.get("distCoeffs", []))
        if not all(math.isfinite(x) for x in camera.dist_coeffs):
            raise ValueError("Non-finite distortion coefficients")
        return camera
