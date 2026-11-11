This repo includes:  

- Specification of dependencies.
- Evaluation code.
- Pre-trained models.
- Dataset


The current public release focuses on evaluation and reproducibility. Additional components, including the complete dataset and training pipeline, will be released in a future update.



## 1. Create Environment:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- [PyTorch >= 1.3](https://pytorch.org/)

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

  ```shell
  pip install -r requirements.txt
  ```
## 2. Prepare Data:


All required files (mini dataset of rSCI, pretrained models for all five training modes, and physical mask) are available for download:

| Platform      | Link         |
|---------------|--------------|
| Baidu Netdisk | [Link](https://pan.baidu.com/s/1kiqzc374xdg8CZbqAlWqXA)    (Access Code: `lpfm`) |
<!--| Google Drive  | [Link](https://drive.google.com/drive/folders/1sh2ppIPPAR9ImqWa6CPJQpL-1xV8DOhS?usp=drive_link)    |-->




## 3. Set Up Data and Model Paths
After downloading, place the data into 'data_root' folder. The structure of the 'data_root' folder should look like this:
```plaintext
  |-- csv
  |   |-- mini.xlsx
  |
  |-- forward_model
  |   |-- physicalModel.mat
  |
  |-- pretrain
  |   |-- m1
  |   |   |-- cst_l.pt
  |   |   |-- cst_m.pt
  |   |   |-- cst.pt
  |   |   |-- dpu_3stg.pt
  |   |   |-- dpu_5stg.pt
  |   |   |-- dpu.pt
  |   |   |-- dssp.pt
  |   |   |-- mst_l.pt
  |   |   |-- mstpp.pt
  |   |   |-- mst.pt
  |   |   |-- rdluf_3stg.pt
  |   |   |-- rdluf_5stg.pt
  |   |   |-- rdluf.pt
  |   |   |-- ssr_3stg.pt
  |   |   |-- ssr_5stg.pt
  |   |   |-- ssr.pt
  |   |
  |   |-- m2
  |   |   |-- ... (same as m1)
  |   |
  |   |-- m3
  |   |   |-- ... (same as m1)
  |   |
  |   |-- m4
  |   |   |-- ... (same as m1)
  |   |
  |   |-- m5
  |       |-- ... (same as m1)
  |
  |-- mini_real_data/          (mini test data)
  │   ├── gt/
  │   │   ├── 20250731090903.mat
  │   │   └── 20250804141225.mat
  │   └── meas/
  │       ├── Scene20250731090903.mat
  │       └── Scene20250804141225.mat
  │
  │
  |-- real_data/               (full data)
      ├── gt/
      │   ├── 20250724154431.mat
      │   ├── 20250731090903.mat
      │   ├── ...
      │   └── 20250804141225.mat
      |   
      └── meas/
          ├── Scene20250724154431.mat
          ├── Scene20250731090903.mat
          │   ├── ...
          └── Scene20250804141225.mat
```
update the root paths in the configuration file  config/basic.yaml:
```yaml
# config/basic.yaml
data_root: path to all the data
_infer_data_root: choose "full data" or  "mini test data"
```

---

## 4. Evaluation

We provide evaluation scripts for five training modes corresponding to different data strategies.

### Run All Models at Once

```bash
python inference.py --no_visual --config=config/infer
```

### Run Individual Modes

| Mode | Training Data | Command |
|------|--------------|---------|
| 1 | Models trained on SFM data | `python inference.py --no_visual --config=config/infer/m1` |
| 2 | Models trained on LPFM data | `python inference.py --no_visual --config=config/infer/m2` |
| 3 | Models trained on R (real) data | `python inference.py --no_visual --config=config/infer/m3` |
| 4 | Models trained on R + SFM data | `python inference.py --no_visual --config=config/infer/m4` |
| 5 | Models trained on R + LPFM data | `python inference.py --no_visual --config=config/infer/m5` |

Or run a specific model directly:

```bash
python inference.py --no_visual --config=config/infer/m5/dpu_5stg.yaml

python inference.py --no_visual --config=config/infer/m5/ssr_5stg.yaml

...
```

---
#### Notes
- --no_visual disables visualization during evaluation.
- All evaluation results will be automatically saved to the configured output directory.
- Ensure that the dataset paths and checkpoint paths are correctly configured before running inference.




