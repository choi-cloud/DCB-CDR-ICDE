
# DCB-CDR: Dual Diffusion with Coarse-to-Fine Guidance and Bias-aware Integration

Official implementation of DCB-CDR for cross-domain recommendation.

## Requirements

- Python with PyTorch and CUDA support
- Dependencies listed in `requirements.txt`
- Amazon 5-core review datasets: `Books`, `Movies_and_TV`, and `CDs_and_Vinyl`

## Data Preparation

Create the following directories:

```bash
mkdir -p data/raw data/mid data/ready
```

Download the Amazon 5-core review datasets and place the following files in `data/raw/`:

- `reviews_Books_5.json.gz`
- `reviews_Movies_and_TV_5.json.gz`
- `reviews_CDs_and_Vinyl_5.json.gz`

Set `"root": "./data/"` in `config.json`.

Run preprocessing:

```bash
python entry.py --root ./ --process_data_mid 1
python entry.py --root ./ --process_data_ready 1
```

## Task Definitions

| Task | Source Domain | Target Domain |
|---|---|---|
| 1 | Movies and TV | CDs and Vinyl |
| 2 | Books | Movies and TV |
| 3 | Books | CDs and Vinyl |

## Running DCB-CDR

The following example runs Task 3 with 20% training users and 80% testing users.

### Step 1. MF Pretraining

```bash
mkdir -p checkpoints

python -u entry.py --root ./ --exp_part None_CDR --base_model MF \
  --experiment mf_task3 --save_path ./checkpoints/pretrain \
  --gpu 0 --task 3 --ratio '[0.2, 0.8]' --epoch 10 --seed 1
```

The pretrained MF checkpoint is saved to `checkpoints/`.

### Step 2. DCB-CDR Training and Evaluation

```bash
python -u entry.py --root ./ --exp_part dcb_CDR --base_model MF \
  --experiment dcb_task3 --save_path ./checkpoints/pretrain \
  --gpu 0 --task 3 --ratio '[0.2, 0.8]' --epoch 10 --seed 1 \
  --diff_lr 0.01 --mapping_lambda 0.01 --uniformity_loss 0.1 \
  --recon_loss 0.01 --pretrain_epochs 50
```

To run other tasks or training ratios, modify `--task` and `--ratio`.

## Outputs

- `results/`: Evaluation results
- `logs/`: Training and evaluation logs
- `checkpoints/`: MF pretrained checkpoints
- `data/rqvae_ckpt2/`: Cached RQ-VAE checkpoints
- `data/stylecache/`: Cached user and item rating statistics

Cached RQ-VAE checkpoints and rating statistics are automatically reused when available. If the pretrained MF embeddings or experimental settings change, regenerate the corresponding caches.
