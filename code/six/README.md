# The six-object pipeline

FruitNinja releases six reconstructions. This directory takes all six from the released `.ply` to
the DreamSim table and the comparison figure, in four steps that run in order:

    python3 six/objects.py          # 1. write method/objects/<name>.conf for each object
    python3 six/prep.py             # 2. put every reference on white, repoint the confs
    bash    six/train.sh            # 3. train all six, one per GPU
    bash    six/eval.sh             # 4. render both arms and score, one batch
    python3 six/figure.py           # 5. the comparison figure

Steps 1, 2 and 5 run on `/usr/bin/python3` (3.12, where the analysis packages are). Steps 3 and 4
launch the render venv themselves. Never run 1, 2 or 5 under `/workspace/fn_remote/venv/bin/python`.

## Why step 2 exists

Section supervision finds the reference's slice by thresholding against white. Eight of the released
references were shot on a dark or textured backdrop, so the whole frame read as the slice: the fitted
radius was the frame's rather than the slice's, inflated by 1.25x to 1.82x on the cake. `prep.py`
floods the backdrop to white and repoints the affected confs at `refs_white/`. Nothing errors without
it; the training log prints `dark 66.56%` and the run proceeds.

## What step 4 measures, and what it does not

Every arm and every baseline is rendered in one batch at the same plane indices, because DreamSim
values drift by up to 0.006 between batches.

`HELDOUT_BAND=0.30,0.70` is not a region training never reaches. The depth walk takes `H_STEPS=24`
steps, `H_LO:H_HI` supervises centres 4..19, and at `JITTER=0.5` each slot's window is exactly one
step wide, so the windows tile and training covers f in [0.146, 0.813] continuously. What is held
out is the particular plane, not the depth. The only genuinely unseen quantity in this project is
the left-out photograph of the leave-one-out folds (`mkfolds.py`).

## Results, as of the batch this directory produces

| object | this work | FruitNinja released | photographs to each other |
|---|---|---|---|
| orange | **0.0700** | 0.1290 | 0.0424 |
| watermelon | **0.1783** | 0.2328 | 0.0670 |
| apple | **0.2951** | 0.3823 | 0.1653 |
| pomegranate | **0.2508** | 0.2985 | 0.1324 |
| bread | **0.3685** | 0.4949 | 0.1873 |
| cake | 0.2816 | **0.2669** | 0.2677 |

Five of six, by 1.19x to 1.84x. The cake loses by 0.015, against a baseline that on that object
sits at the photographs' own agreement with each other.
