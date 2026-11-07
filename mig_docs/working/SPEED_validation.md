# SPEED real-data NPU validation — 2026-11-07

## Scope and dataset
Validated /data/FastPose-ViT on Ascend 910B2C using .venv-npu (torch 2.10.0, torch_npu 2.10.0). FASTPOSE_DEVICE=npu:0.
Archive: /data/lost+found/speed.zip, 4925443892 bytes, SHA256 2869433b585cab03a9b7e02b28f2de7bd94da90fc863840ccefbf3b153d7ac97.
Extracted data: /data/datasets/speed. Prepared split: /data/datasets/SPEED_FIXED.
Original labeled training set (12000 images) split with sklearn train_test_split: 20% holdout, then half validation/half test, random_state=42 for both calls. Train/validation/test = 9600/1200/1200. Split images are hardlinks. Repository bbox annotations copied into the dataset root.
This is an internal split, not the official blind benchmark. Supplied checkpoint training provenance is not established; training overlap cannot be excluded. Scores demonstrate executable adaptation and are not independent generalization claims.
SPEED+ image data is absent; its metadata folder is insufficient for validation.

## Completed evaluations
All three official evaluation commands completed successfully on 1200 images.
Pose evaluation uses repository bounding-box annotations, not detector-predicted crops.

| Pose checkpoint | Translation error (m) | Rotation error (rad) | Relative translation | SPEED score |
|---|---:|---:|---:|---:|
| vit_b_16.pth, 224 | 0.0231 | 0.0078 | 0.0021 | 0.0100 |
| vit_b_16_384.pth, 384 | 0.0221 | 0.0074 | 0.0020 | 0.0093 |

Detector small.pth: AP@0.50=0.9992, AP@0.75=0.9958, AP@[0.50:0.95]=0.9519, as calculated by the repository evaluator. Test Loss (IoU)=0.1926 (a loss, not mean IoU). 1200 predictions/1200 targets. Detector and pose are evaluated separately; detector-to-pose chained accuracy has not been measured.

Real-image numerical check (32 test images, 224 model): CPU FP32 versus NPU FP32 maximum absolute output difference 0.000463009, mean 0.0000868492; versus NPU FP16 maximum 0.000968933, mean 0.000197721. FP32 assert_close passed with atol=rtol=0.005. All outputs finite.
Real training smoke: 12 training images, 3 batches of 4, original losses/augmentation, AMP FP16, GradScaler initial scale 256, AdamW lr=1e-6. Backward gradients finite and optimizer steps passed. Losses: 0.0191661, 0.0210655, 0.0155862. This is not convergence testing. Existing checkpoints were not overwritten.

## Reproduction
```bash
cd /data/FastPose-ViT
export OMP_NUM_THREADS=4 FASTPOSE_DEVICE=npu:0
for model in vit_b_16 vit_b_16_384; do
 .venv-npu/bin/python -m src.evaluate --dataset SPEED --dataset_root_dir /data/datasets/SPEED_FIXED --vit_model ${model}_no_weights --model_weights weights/${model}.pth --rotation_format matrix --merge_outputs --no_mlp --batch_size 8 --num_workers 2 --skip_tensorboard --log_dir mig_docs/working/speed_validation/${model} --output_json mig_docs/working/speed_validation/${model}.json
done
.venv-npu/bin/python -m object_detector.evaluate --model_weights weights/small.pth --model_variant small --dataset SPEED --dataset_root_dir /data/datasets/SPEED_FIXED --batch_size 4 --num_workers 2 --output_dir mig_docs/working/speed_validation/detector --log_dir mig_docs/working/speed_validation/detector_logs
.venv-npu/bin/python scripts/validate_speed_npu.py
```

## Evidence
Logs and predictions are in mig_docs/working/speed_validation/: vit_b_16.log/json, vit_b_16_384.log/json, detector.log, detector/predictions_small.json, golden_train.log/json.
Remaining unverified: SPEED+, independent official test accuracy, long training convergence, multi-card training, detector-to-pose chained evaluation and GPU speedup.
