# MSMamba

Official PyTorch implementation of **MSMamba: A Multi-Semantic Mamba Framework for Referring Remote Sensing Image Segmentation**.

MSMamba is designed for **Referring Remote Sensing Image Segmentation (RRSIS)**, which aims to segment the target object in a remote sensing image according to a natural-language expression.

## Overview

MSMamba combines a VMamba-based visual backbone with language guidance from BERT. The model introduces multi-semantic visual-language interaction, including sentence-level guidance, local pixel-word correlation, and attribute-word guidance, to improve fine-grained referring segmentation.

## Requirements

Install PyTorch according to your CUDA version. For example:

```bash
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Then install other dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The selective scan CUDA operators required by VMamba should also be installed correctly.


## Pretrained Weights

Please place the VMamba pretrained checkpoint under:

```text
pretrain/vssm_base_0229_ckpt_epoch_237.pth
```

## Training

```bash
python main.py \
  --model MSMamba \
  --data-set rrsisd \
  --data-path ./datasets \
  --input-size 480 \
  --batch-size 8 \
  --epochs 50 \
  --output-dir ./outputs/msmamba
```

## Evaluation

```bash
python main.py \
  --model MSMamba \
  --data-set rrsisd \
  --data-path ./datasets \
  --input-size 480 \
  --eval \
  --resume ./outputs/msmamba/best_checkpoint.pth
```

The evaluation reports mIoU, oIoU, and Pr@50–Pr@90.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zhang2026msmamba,
  title={MSMamba: A Multi-Semantic Mamba Framework for Referring Remote Sensing Image Segmentation},
  author={Zhang, Tianxiang and Li, Junbai and Feng, Yanqiang and Wen, Zhaokun and Liu, Li and Li, Jiangyun},
  journal={Remote Sensing},
  volume={18},
  number={12},
  pages={1949},
  year={2026},
  publisher={MDPI}
}
```

## Acknowledgement

This project is built upon PyTorch, timm, Transformers, spaCy, and VMamba.

## License

This code is released for academic research purposes.
