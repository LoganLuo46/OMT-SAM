# id ↔ name 按你截图确定
ORGANS = {
     1: "liver",
     2: "right_kidney",
     3: "spleen",
     4: "pancreas",
     5: "aorta",
     6: "ivc",
     7: "right_adrenal_gland",
     8: "left_adrenal_gland",
     9: "gallbladder",
    10: "esophagus",
    11: "stomach",
    13: "left_kidney",
}


import os, glob, random, numpy as np, torch
from os.path import join, basename
from skimage.transform import resize
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import nibabel as nib


class OrganSliceDataset(Dataset):
    """
    读取指定目录下的 .nii.gz 文件，根据 type 选择 img 或 gt。
    保持原有的数据处理方式和输出格式。
    输出: (img_tensor, gt_tensor, bbox_tensor, 文件名, text_ids)
    """
    def __init__(self, nii_dir, organ_id=1, bbox_shift=20, img_size=1024, gt_size=256, type="gt"):
        super().__init__()
        assert type in ["img", "gt"], "type must be 'img' or 'gt'"
        self.type = type
        self.nii_dir = nii_dir
        self.organ_id = organ_id
        self.bbox_shift = bbox_shift
        self.img_size = img_size
        self.gt_size = gt_size
        self.files_img = sorted([f for f in os.listdir(nii_dir) if f.endswith('_img.nii.gz')])
        self.files_gt = sorted([f for f in os.listdir(nii_dir) if f.endswith('_gt.nii.gz')])
        assert len(self.files_img) == len(self.files_gt), "img/gt数量不一致"
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        self.slice_map = list(range(len(self.files_img)))
        if not self.slice_map:
            raise RuntimeError(f"No nii.gz files found in {nii_dir}")
        # print(f"Found {len(self.files_img)} img/gt pairs in {nii_dir}")

    @staticmethod
    def _resize(arr, tgt, order):
        return resize(arr, (tgt, tgt), order=order,
                      preserve_range=True, mode="constant",
                      anti_aliasing=(order != 0)).astype(arr.dtype)

    def __len__(self):
        return len(self.slice_map)

    def __getitem__(self, idx):
        img_fname = self.files_img[idx]
        gt_fname = self.files_gt[idx]
        img_path = os.path.join(self.nii_dir, img_fname)
        gt_path = os.path.join(self.nii_dir, gt_fname)
        # 读取 nii.gz
        img = nib.load(img_path).get_fdata()  # (H,W)
        gt = nib.load(gt_path).get_fdata()    # (H,W)
        # resize
        if img.shape[0] != self.img_size:
            img = self._resize(img, self.img_size, order=3)
        if gt.shape[0] != self.gt_size:
            gt = self._resize(gt, self.gt_size, order=0)
        img = np.repeat(img[:, :, None], 3, axis=-1).astype("float32") / 255.
        img = np.transpose(img, (2,0,1))            # (3,H,W)
        gt = self._resize(gt, self.img_size, order=0)
        gt2D = (gt == self.organ_id).astype("uint8")# (H,W)
        # bbox
        y_idx, x_idx = np.where(gt2D > 0)
        x_min, x_max = np.min(x_idx), np.max(x_idx)
        y_min, y_max = np.min(y_idx), np.max(y_idx)
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bbox  = np.array([x_min, y_min, x_max, y_max], dtype="float32")
        text_ids = self.tokenizer(
            str(self.organ_id),
            truncation=True, padding="max_length",
            max_length=77, return_tensors="pt"
        )["input_ids"].squeeze(0)
        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(gt2D[None], dtype=torch.long),
            torch.tensor(bbox, dtype=torch.float32),
            img_fname,
            text_ids
        )

import torch, random, numpy as np
from torch.utils.data import DataLoader

def build_dataloaders():
    root_dir = "data/npy/CT_Abd"
    batch_size, num_workers = 2, 0
    dataloaders = {}
    for oid, name in ORGANS.items():
        try:
            ds = OrganSliceDataset(root_dir, type="img")
            dl = DataLoader(ds, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True)
            dataloaders[oid] = dl
        except RuntimeError as e:
            print(f"[{name}] skipped:", e)
    return dataloaders

def main():
    dls = build_dataloaders()
    imgs, gts, bboxes, names = next(iter(dls[4]))   # 4 = pancreas
    print(imgs.shape, gts.shape, names[:3])

if __name__ == "__main__":
    main()
