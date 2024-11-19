# SpikingNeRF: Making Bio-inspired Neural Networks See through the Real

Code implementation for SpikingNeRF-D, which is described in the main text of the paper.  We provide the codes for anything you like. 

## Quick setup

One can follow the DVGO official setup tutorials.  Or, follow our implemented setup procedures:

1. Install PyTorch.  Here, we show the setup on our A100 GPU CentOS server.
   ```
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
   ```

2. Install MMCV:
   ```
   pip install -U openmim
   mim install mmcv==1.7.0
   ```

3. Install torch_scatter.  Here, we show our practice.  First, download torch_scatter-2.1.1+pt20cu117-cp38-cp38-linux_x86_64.whl.  Then:
   ```
   pip install torch_scatter-2.1.1+pt20cu117-cp38-cp38-linux_x86_64.whl
   ```

4. Install Miscs: 

   ```
   cd SpikingNeRF-D
   pip install -r requirements.txt
   ```

5. Prepare datasets.  Download the datasets and place them as described in "Directory structure for the datasets" of the official DVGO repository .

Afther the envronment of DVGO is set up, install SpikingJelly to support spiking neural networks, i.e, SpikingNeRF:

```
pip install spikingjelly
```

## Training

Here is the python command for training a TCP-based SpikingNeRF-D model:

```
python run.py --config configs/nerf_main_condensing/lego.py --render_test
```

In the `configs/nerf_main_condensing`file, besides `lego.py`, we store the configurations for all the other scenes of Synthetic-NeRF.

SpikingJelly sometimes fails to detect CuPy correctly.  In this case, one should switch the backend to torch: In `lib/spiking_dvgo.py`, replace all `backend='cupy'`by `backend='torch'`.

Additionally, the training command for a TP-based SpikingNeRF-D:

```
python run.py --config configs/nerf_main_not_condensing/lego.py --render_test
```

## Evaluation

Here is the python command for evaluating a TCP-based SpikingNeRF-D model:

```
python run.py --config configs/nerf_main_condensing/lego.py --render_test --eval_ssim --render_only --op_count
```

Results of SpikingNeRF-D on this scene will be printed, including PSNR, SSIM, FLOPs, and SOPs.

Correspondingly, TP-based evaluation command:

```
python run.py --config configs/nerf_main_not_condensing/lego.py --render_test --eval_ssim --render_only --op_count
```

## What's more

One can easily re-write the config file (e.g. configs/nerf_main_condensing/lego.py) to reproduce different ablation results, e.g. temporal flip. 
SpikingNeRF-T is the TensoRF based spikingnerf. Just check and understant how to use autorun.py in Spikingnerf-t.
and `python autorun.py`.

## Declaration 

This project will be updated irregularly until the full version is contained. I run this project all by my own and the critical idea was proposed by the other first author. 

