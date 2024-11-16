# SpikingNeRF: Making Bio-inspired Neural Networks See through the Real

Code implementation for SpikingNeRF-D, which is described in the main text of the paper.  We provide the codes for anything you like. I also have some words in the Declaration part.

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

## Declaration & Some Nonsignificant Personal Complaints （I just failed holding it back）

This project will be updated irregularly until the full version is contained. I run this project all by my own. I would take responsibility for what I said. **Do not relate to my affiliation.**

I was really pissed by the lying reviewer of AAAI25 I unluckily encountered.  This MF wrote a  bunch of comments that were full of fake evidence and fabricated arguments. In the very beginning, this paper was poorly written and the experiments are not complete. But, after a year of perfection. I got 7/6/3/2 scores in AAAI25. the reviewers who gave me 7/6 recognized the contibution and deemed this work as "solid". The reviewer who gave me 3 had its own insistence on the hardware part (its concept of hardware was wrong and it appeared not familiar with hardware at all). Fine, I respect it although the guy was unprofessional and low-level. At least, he had his opinion and recognized my originality.

The most ridiculous part came. The guy who gave me 2 points wrote a lot of comments but every words is full of fake evidence, injustice, and FKing **LIES**. Swear to GOD, I would love to expose the comments and this MF, if I could. This guy first blame me for **NOT** Citing A Paper That Appeared **LATER** than the AAAI Submission Deadline. BTW, that paper reports worse results than mine. Then, it claimed I used the similar methods of some irrelevant works. However, **none** of the tricks of those works would even possibly apply to mine. Totally different works. And, I had clearly written in the paper and codes that my models were all trained **end2end**. But, this MF accused me of using the "**ANN2SNN** conversion" training, which"should not be my origianlity". **WTF?!!**. It also questioned my experiemts were insufficient, but the expermetal results I gave were more than any of those works it mentioned. I put so many results that the main text could not contain. There were so many other **LIES**. I am not gonna list them all. I will post the full rebuttal if I could someday. 

I am totally open to negative comments. They can help perfect my work. But such **toxic lies** are driving me FKing crazy. It exactly fits the old Chinese saying "欲加之罪何患无辞", just fabricate some crimes and put me in the jail. 

I was so so so pissed.  Why would I suffer such persecution? 

Now, I clearly understand why influential works like Mamba, PACT, would be rejected. Some vicious peers.  Nevertheless, I have to let it go. I am just tired. Need to move on. But,  wont go any further with this project. 11-16,2024. 

