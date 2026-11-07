# SPEED+ dataset preparation — 2026-11-07

Usable dataset: /data/datasets/SPEED_PLUS_FIXED
Original extraction: /data/datasets/speedplusv2
Original archive retained: /data/lost+found/speedplusv2.zip
Dataset selector: SPEED_PLUS (not SPEED).

## Layout

```text
SPEED_PLUS_FIXED/
  train.json                         # 47,966 synthetic samples
  val.json                           # 11,994 synthetic samples
  test.json                          # 6,740 lightbox + 2,791 sunlamp
  images/train/
  images/val/
  images/test/
  speed_plus_bbox_annotations.json    # 69,491 repository bbox entries
  camera.json                        # official calibration
  LICENSE.md
  image_sources.json                 # prepared filename -> original relative path
  preparation_report.json
  loader_validation.json
```

The original synthetic train/validation partition is retained. Test images come only
from the original lightbox and sunlamp test annotations. Real-image filenames receive
_lightbox or _sunlamp suffixes, matching the repository bbox keys and avoiding name
collisions. Images are hard-linked to the original extraction: no second image copy.
Editing an image in either directory changes the same underlying file; treat both as
read-only data. JSON files are independent. The original archive is retained.

## Validation

All archive members were extracted with ZIP CRC validation. All 69,491 images passed
PIL header verification and have size 1920x1200. Every labeled image has a valid bbox,
finite translation and approximately unit quaternion, with no duplicate prepared
filenames. All 69,491 bbox entries match the prepared dataset.

Sampled four examples from each split, including both real domains, through the
actual SPEEDDataset loader. Images are [4,3,224,224]; rotation labels [4,3,3]; geometry
is finite; relative/absolute translation round-trip passed. The geometry dual-token
model produced finite [4,9] outputs on CPU for every sampled batch. This validates
loading and interface compatibility, not pose accuracy. CPU was used to avoid
competing with the existing NPU training job.

The code previously used the SPEED camera for all datasets. SPEED+ now loads pixel
intrinsics from dataset-local camera.json: fx=2988.5795163815555 and
fy=2988.3401159176124. Both training-label conversion and output restoration use the
same camera instance. The SPEED default is unchanged. Distortion coefficients are
retained as metadata; the existing pinhole pipeline does not undistort images. No
image resampling or bbox adjustment was introduced during preparation.

## Usage

For the existing training/evaluation/prediction entry points set:

```bash
--dataset SPEED_PLUS --dataset_root_dir /data/datasets/SPEED_PLUS_FIXED
```

For direct Python loading:

```python
from src.datasets import SPEEDDataset
root = '/data/datasets/SPEED_PLUS_FIXED'
dataset = SPEEDDataset(root, split='train', dataset_name='SPEED_PLUS',
    rotation_format='matrix',
    bbox_json_path=root + '/speed_plus_bbox_annotations.json')
```

Keep the dual-token model options and checkpoint options from the model report.
The current background training job continues on SPEED_FIXED; preparing SPEED+ does
not switch or restart it. The two-stage shell recipe currently hardcodes --dataset
SPEED, so do not change only its DATASET_ROOT for SPEED+: invoke src.train with the
correct dataset selector, or update both settings together.

## Evidence and storage

Preparation script: scripts/prepare_speedplus_hardlinks.py (fresh destinations only).
Validation script: scripts/validate_speedplus_dataset.py.
Logs: mig_docs/working/speedplus_preparation/prepare.log and validate.log.
Source backup for camera loader correction:
mig_docs/working/speedplus_preparation/camera_source_before.tar.gz.

Extracted bytes: 17,341,210,131. Disk free after preparation: approximately 1.69 GB.
No archives, original datasets, model weights or existing experiments were deleted.
