# DCB-CDR: Dual Diffusion with Coarse-to-Fine Guidance and Bias-aware Integration

Research implementation for cross-domain rating prediction. This release candidate retains the model and evaluation paths used in the `exp058` single-run verification. **The diffusion reconstruction objective in this package is MSE.**

## Requirements

- Python with PyTorch and CUDA support for GPU training. See `requirements.txt` for a snapshot of packages in the development environment. The list is an environment snapshot, **not a guaranteed cross-platform pip lockfile** (notably the `+cu121` builds).
- Amazon review 5-core data for `Books`, `Movies_and_TV`, and `CDs_and_Vinyl`. Follow the dataset provider's access and licensing terms; data files are not redistributed here.

## Data preparation

Commands below assume the shell working directory is this repository root. The **CLI `--root`** points to the directory containing `config.json` (default `./`), whereas **`config.json`'s `root` field** points to the dataset directory. They are different.

1. Create `data/raw/`, `data/mid/` and `data/ready/`.
2. Place the three gzipped 5-core review files in `data/raw/` as `reviews_Books_5.json.gz`, `reviews_Movies_and_TV_5.json.gz`, and `reviews_CDs_and_Vinyl_5.json.gz`.
3. Edit `config.json` and set `"root": "./data/"` (or a full absolute path ending in `/`). The package default is deliberately not an environment-specific path.
4. Run preprocessing:

```bash
python entry.py --root ./ --process_data_mid 1
python entry.py --root ./ --process_data_ready 1
```

This prepares all three tasks and the 20/80, 50/50, 80/20 train/test ratios. **Reproducibility caution:** the original preprocessing assigns integer IDs using Python `set` iteration, whose order can vary across environments. To compare results exactly, preserve the generated `data/ready/` files and their ID mapping; a fresh preprocessing run is not guaranteed to produce byte-identical indices. The test user split uses its own fixed `seed=2020` in `preprocessing.py`.

## Task definitions

| Task | Source domain | Target domain |
| --- | --- | --- |
| 1 | Movies and TV | CDs and Vinyl |
| 2 | Books | Movies and TV |
| 3 | Books | CDs and Vinyl |

## Run Task 3 (20% training / 80% testing)

Create a writable `checkpoints/` directory:

```bash
mkdir -p checkpoints
python -u entry.py --root ./ --exp_part None_CDR --base_model MF \
  --experiment mf_task3 --save_path ./checkpoints/pretrain \
  --gpu 0 --task 3 --ratio '[0.2, 0.8]' --epoch 10 --seed 1
```

The MF checkpoint is saved as `checkpoints/pretrain_3_[0.2, 0.8].pth`. **Then** run DCB-CDR with the same checkpoint prefix:

```bash
python -u entry.py --root ./ --exp_part dcb_CDR --base_model MF \
  --experiment dcb_task3 --save_path ./checkpoints/pretrain \
  --gpu 0 --task 3 --ratio '[0.2, 0.8]' --epoch 10 --seed 1 \
  --diff_lr 0.01 --mapping_lambda 0.01 --uniformity_loss 0.1 \
  --recon_loss 0.01 --pretrain_epochs 50
```

Results are written to `results/` and logs to `logs/` relative to the working directory. These settings match the command used in the `exp058` *single-seed* test, not necessarily every reported paper run. Additional cached RQ-VAE checkpoints and user/item rating statistics are written to `data/rqvae_ckpt2/` and `data/stylecache/`. The existing RQ cache key does **not** encode an MF checkpoint fingerprint: delete/invalidate the cache if the MF weights or related settings change.

### Verification scope and limitations

The `exp058` run (Task 3, 20/80, seed 1) completed with MAE **0.6934586**, RMSE **0.9544179** using the MSE diffusion loss. The manuscript's corresponding Table III values, **0.6602/0.9444**, are five-run averages from an earlier implementation; the authors have **not** established five-run metric equivalence for this MSE release candidate.

The training code evaluates the test split after each epoch and reports best MAE and RMSE independently, following its inherited experimental protocol. These values may correspond to different epochs; this should be considered when interpreting reported results. Users should not treat these as validation-selected estimates.

This is a research implementation with legacy baseline imports and preprocessing utilities retained to preserve the original execution paths. Please verify the final paper against the released source before submission.

## Files

- `entry.py`: CLI, config loading, preprocessing, execution.
- `run.py`: training, evaluation, local-context aggregation, checkpoint and style caching.
- `models.py`, `DiffModel.py`, `rqvae.py`, `dpm_solver_pytorch.py`: model components and diffusion samplers.
- `preprocessing.py`: raw review conversion and task splitting.
- `utils.py`: auxiliary utilities and style statistics.
- `sscdr_model.py`, `lacdr_model.py`: preserved baseline import dependencies.

No raw datasets, checkpoints, cache files, model logs, results, credentials or local experimental wrapper scripts are included.
