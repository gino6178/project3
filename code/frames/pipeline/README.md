# Adding a generated object

One line in `objects.json`, then `bash local_object.sh <object>` until it says the object is queued.
Every step skips itself when its output exists, so the same command resumes after any stop.

```
objects.json        name -> long / trans / exterior prompts, kind (revolve | box), curved (true | false)
local_object.sh     1 generate -> 2 review -> 3 shape -> 4 prepare -> 5 submit           (this machine)
remote_object.sh    carrier (code/run.sh) | priors -> lifts -> held-out DreamSim -> cut animation  (GPU machine)
sched_ai.sh         one worker per GPU draining ~/ov/queue.txt through remote_object.sh
gen.py, sheet.py    Gemini 2.5 Flash Image generation; the review contact sheet
```

## The steps

1. **generate** — nine lengthwise and nine transverse cut faces and four exteriors, each image a separate
   request and so a separate specimen, as the method assumes. Prompts ask for an orthographic face on white.
2. **review** — `aiobj/<o>/sheet.png`. Reject in `aiobj/REJECT.txt` anything that is not one orthographic cut
   face of the stated family: two pieces side by side, perspective, the wrong cut, text. Lengthwise faces
   photographed lying down are fine (they are stood up). Choose the exterior: `echo K > aiobj/<o>/REVIEWED`.
3. **shape** — TRELLIS.2 on `p3-fast` lifts the exterior to a mesh; `glb2grid.py` fills it (slice-wise, so an
   open mesh is filled too), scales its height to the lengthwise photographs' aspect, and gives a thin object
   (aspect > 2) a 256^3 grid.
4. **prepare** — a curved object has its bend turned into the theta=0 lengthwise plane and its centreline read
   off the shape (`orient_bend.py`); `prep_ai.py` frames every photograph to the grid's own slice, centres every
   cross-section, unwraps the polar strips, and for a curved object straightens the lengthwise photographs.
5. **submit** — uploaded to `p3-train-4g` and queued; the next free GPU takes it.

## What `remote_object.sh` runs

The paper's program, nothing else: the O-Voxel carrier by `code/run.sh` (the object enters as a coloured
Gaussian set, exterior coloured, interior one mean colour), `sd3d_train.py` and `polar_train.py`, Algorithm 1
on the cylinder and on the Cartesian lattice, on both curvilinear lattices for a curved object, the paper's
held-out DreamSim (`dsscore.py`, `dsrun.py`) and, for a curved object, the same cut along its centreline
(`eval_curved.py`), then the lift baked into the O-Voxel and cut by `figures/draw_cuts.sh`.

## Mistakes already made, now prevented

| what went wrong | what the pipeline does |
|---|---|
| the carrier converted at 256^3: every other interior voxel empty, a white interior | `voxelize_ov.py` always at 128^3, resampled to the object's grid by `ov2grid.py` |
| cross-sections framed off the grid axis: polar strips with almost no object in them | cross-sections centred in the frame |
| a curved object's bend outside the theta=0 plane its lengthwise photographs are framed against | `orient_bend.py` turns the grid |
| a centreline written for 128^3 only, half of a 256^3 banana held constant | `centerline_from_grid.py` covers the grid's whole height |
| a generated mesh taller than the cut faces, photographs stretched | `ASPECT` from the lengthwise photographs |
| an open generated mesh filled only at its top | slice-wise filling in `glb2grid.py` |
| a generated mesh's up running opposite to a lengthwise face's rows: every photograph fitted upside down | `glb2grid.py` mirrors the mesh in y |
| a job "still running" forever: `pgrep -f` matching its own command line | a pid file |
| three priors on one GPU at once, each twice as slow; objects pinned to GPUs | one prior at a time per GPU, a shared queue |
| a batch of 18 objects filled the 291 GB disk with carrier checkpoints | checkpoints deleted once the carrier is trained |
| a generated honeycomb reused a synthetic honeycomb's trained carrier (same run name) | generated objects run remotely as `ai_<name>` |

Results land in `~/ov/ai_<o>/`: `score.log` (flat protocol), `curved_scores.json` (curved objects),
`pipeline.log` (stages and times); the cut animation in `~/show/<o>_<arm>.gif`.
