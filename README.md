# OMT-SAM

## Script, clip_variant, log and output directory mapping

| clip_variant | Script file                | First Round (.err/.log) (48h)                              | Second Round (48h)                  | Output directory                                              |
|:------------:|:--------------------------|:-----------------------------------------------|:-------------------------------------|:------------------------------------------------------------|
| biomedclip   | train_one_gpu.py          | clip_train.err / clip_train.log                | clip_train_2nd.err / clip_train_2nd.log | work_dir/MedSAM-ViT-B_MSFalse_oneneckFalse_use_clip_20250616-0933/ |
| bioclip      | train_one_gpu_bioclip.py   | clip_train_bioclip.err / clip_train_bioclip.log|                                      | work_dir/MedSAM-ViT-B_MSFalse_oneneckFalse_use_clip_20250617-1643/ |
| clip         | train_one_gpu_clip.py      | true_clip_train.err / true_clip_train.log      |                                      | work_dir/MedSAM-ViT-B_MSFalse_oneneckFalse_use_clip_20250617-1449/ |


