import argparse
import logging
import logging
import time

import torch
torch.set_default_dtype(torch.float32)
#from pytorch_lightning import seed_everything
#seed_everything(1234)

from einops import rearrange




from model import model_generator
from physical import physical_generator
from datamodule import dataset_generator
from tools.utils import joinc, torch_ergas, torch_psnr, torch_rmse, torch_ssim,torch_sam
import scipy.io as sio
import numpy as np

from visual.visual_per_bands import visual_x2
from tools.config_util import recursive_load_config



from torch.utils.data import default_collate
def collate_warp(batch_size):
    def collate_fn(batch):
        now_batch_num = len(batch)
        if now_batch_num < batch_size:
            # Calculate the number of samples to supplement
            num_required = batch_size - len(batch)
            # Randomly sample num_required samples
            additional = [random.choice(batch) for _ in range(num_required)]
            # Add supplemental samples to the batch
            batch += additional
        return default_collate(batch)
    return collate_fn

def setup_data(opt):
    """Load the physical model and dataset only once, returning reusable (physical_model, dm)."""
    logging.info(f"Loading physical model: {opt.forward_model_path}")
    physical_model = physical_generator(opt)
    logging.info(f"Physical model loaded: {type(physical_model)}")

    logging.info("Loading dataset...")
    dm = dataset_generator(
        opt, collate_fn=collate_warp(opt.batch_size), physical_model=physical_model
    )
    dm.setup(stage=None)
    logging.info(f"Train loader: {len(dm.train_dataloader())} batches")
    logging.info(f"Val loader: {len(dm.val_dataloader())} batches")
    return physical_model, dm


def infer_one_model(opt, physical_model, dm, save_visual=True):
    """Load a single model and evaluate, returning (psnr_mean, ssim_mean, sam_mean, per_sample_psnr, per_sample_ssim, per_sample_sam)."""
    logging.info(f"Loading reconstruction model: {opt.load_params_path}")
    model = model_generator(opt, physical_model)
    if hasattr(opt, "load_params_path"):
        if ".pt" in opt.load_params_path:
            model.load_state_dict(torch.load(opt.load_params_path, map_location="cpu"))
        else:
            ckpt = torch.load(opt.load_params_path, map_location="cpu")
            model_dict = {k.replace("restoreModel.", ""): v
                        for k, v in ckpt["state_dict"].items()
                        if k.startswith("restoreModel.")}
            model.load_state_dict(model_dict)
            print(f"Loaded checkpoint from {opt.load_params_path} with keys: {list(model_dict.keys())[:5]} ...")
    else:
        raise Exception("No checkpoint specified for inference.")
    model.cuda()
    model.eval()
    model.set_test_mode()
    logging.info("Starting inference...")

    psnr_list, ssim_list,rmse_list, ergas_list, sam_list = [], [], [], [], []
    for idx, data in enumerate(dm.test_dataloader()):
        gt, input_meas = data['gt'], data['meas']
        gt, input_meas = gt.cuda(), input_meas.cuda()

        with torch.no_grad():
            tic = time.time()
            model_out = model(input_meas, (physical_model.cassi_model.test_phi, physical_model.cassi_model.test_phis))
            print(f"time cost:{time.time() - tic}")
        print("model_out", model_out.shape)

        for k in range(gt.shape[0]):
            gt_max = gt[k, :, :, :].max()
            pred_norm = model_out[k, :, :, :]
            pred_norm = torch.clip(pred_norm, 0, 1)
            gt_norm = gt[k, :, :, :] / gt_max

            # Calculate metrics
            psnr_val = torch_psnr(pred_norm, gt_norm)
            ssim_val = torch_ssim(pred_norm, gt_norm)
            rmse_val = torch_rmse(pred_norm, gt_norm)
            ergas_val = torch_ergas(pred_norm, gt_norm)
            sam_val = torch_sam(pred_norm, gt_norm)

            psnr_list.append(psnr_val.detach())
            ssim_list.append(ssim_val.detach())
            rmse_list.append(rmse_val.detach())
            ergas_list.append(ergas_val.detach())
            sam_list.append(sam_val.detach())

        if save_visual:
            out_np = model_out.detach().cpu().numpy()
            out_np = rearrange(out_np, "b c h w -> b h w c")
            gt_np = gt.detach().cpu().numpy()
            gt_np = rearrange(gt_np, "b c h w -> b h w c")
            out_np[out_np < 0] = 0
            visual_x2(out_np[0], np.array(opt.target_bands), joinc(opt.outf, "visual2", f"{idx}.png"))
            visual_x2(gt_np[0], np.array(opt.target_bands), joinc(opt.outf, "visual2gt", f"{idx}.png"))
    per_sample_psnr = torch.stack([x for x in psnr_list]).detach().cpu().numpy()
    per_sample_ssim = torch.stack([x for x in ssim_list]).detach().cpu().numpy()
    per_sample_rmse = torch.stack([x for x in rmse_list]).detach().cpu().numpy()
    per_sample_ergas = torch.stack([x for x in ergas_list]).detach().cpu().numpy()
    per_sample_sam = torch.stack([x for x in sam_list]).detach().cpu().numpy()
    psnr_mean = float(per_sample_psnr.mean())
    ssim_mean = float(per_sample_ssim.mean())
    rmse_mean = float(per_sample_rmse.mean())
    ergas_mean = float(per_sample_ergas.mean())
    sam_mean = float(per_sample_sam.mean())
    
    print(f"[{opt.name}] psnr_mean: {psnr_mean:.4f}, ssim_mean: {ssim_mean:.4f}, sam_mean: {sam_mean:.4f}")
    return psnr_mean, ssim_mean, sam_mean, per_sample_psnr, per_sample_ssim, per_sample_sam




def _export_per_sample_xlsx(per_sample, out_path):
    """Export per-sample metrics to xlsx, one metric per sheet.

    per_sample: {model_name: {"psnr": np.array, "ssim": np.array, "sam": np.array}}
    Table structure: rows = sample index, columns = method name
    """
    import pandas as pd

    model_names = sorted(per_sample.keys())
    num_samples = len(next(iter(per_sample.values()))["psnr"])
    index = [f"sample_{i}" for i in range(num_samples)]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for metric in ["psnr", "ssim", "sam"]:
            data = {name: per_sample[name][metric] for name in model_names}
            df = pd.DataFrame(data, index=index)
            df.to_excel(writer, sheet_name=metric.upper(), float_format="%.4f")

    print(f"Per-sample table exported to: {out_path}")


if __name__ == '__main__':
    import glob
    import os

    parser = argparse.ArgumentParser(
            description=""
        )
    parser.add_argument(
        "--config",
        type=str,
        default="config/infer/",
        help="path to a single yaml config, or a folder containing many yaml configs.",
    )
    parser.add_argument(
        "--export_excel",
        type=str,
        default="metrics.xlsx",
        help="if set, export per-sample metrics to this xlsx file (e.g. results/comparison.xlsx).",
    )
    parser.add_argument(
        "--no_visual",
        action="store_true",
        default=False,
        help="skip saving visualization PNGs to speed up batch evaluation.",
    )
    args = parser.parse_args()

    if os.path.isdir(args.config):
        yaml_files = sorted(glob.glob(os.path.join(args.config, "**/*.yaml"), recursive=True))
        yaml_files += sorted(glob.glob(os.path.join(args.config, "**/*.yml"), recursive=True))
        if not yaml_files:
            raise FileNotFoundError(f"No yaml files found in: {args.config}")
        print(f"Found {len(yaml_files)} config(s) in {args.config}:")
        for f in yaml_files:
            print(f"  - {os.path.basename(f)}")
    else:
        yaml_files = [args.config]

    all_results = {}
    per_sample = {}  # per_sample[model_name] = {"psnr": np.array, "ssim": np.array}
    physical_model = None
    dm = None

    for i, yaml_path in enumerate(yaml_files):
        name = os.path.splitext(os.path.basename(yaml_path))[0]
        cfg = recursive_load_config(yaml_path)
        name = cfg.get("train_data_mode", "") + "_" + name  # Use name from config if available
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}  ({i+1}/{len(yaml_files)})")
        print(f"{'='*60}")

        

        if physical_model is None:
            physical_model, dm = setup_data(cfg)

        psnr_mean, ssim_mean, sam_mean, per_psnr, per_ssim, per_sam = infer_one_model(
            cfg, physical_model, dm, save_visual=not args.no_visual
        )
        all_results[name] = {"psnr": psnr_mean, "ssim": ssim_mean, "sam": sam_mean}
        per_sample[name] = {"psnr": per_psnr, "ssim": per_ssim, "sam": per_sam}

    # if len(all_results) > 1:
    #     print(f"\n{'='*60}")
    #     print("Summary: All Model Results")
    #     print(f"{'='*60}")
    #     print(f"{'Model':<20} {'PSNR':>10} {'SSIM':>10} {'SAM':>10}")
    #     print("-" * 52)
    #     for name, metrics in all_results.items():
    #         print(f"{name:<20} {metrics['psnr']:>10.4f} {metrics['ssim']:>10.4f} {metrics['sam']:>10.4f}")

    #     best_psnr = max(all_results, key=lambda k: all_results[k]['psnr'])
    #     best_ssim = max(all_results, key=lambda k: all_results[k]['ssim'])
    #     best_sam = max(all_results, key=lambda k: all_results[k]['sam'])
    #     print(f"\nBest PSNR: {best_psnr} = {all_results[best_psnr]['psnr']:.4f}")
    #     print(f"Best SSIM: {best_ssim} = {all_results[best_ssim]['ssim']:.4f}")
    #     print(f"Best SAM: {best_sam} = {all_results[best_sam]['sam']:.4f}")

    if args.export_excel:
        _export_per_sample_xlsx(per_sample, args.export_excel)