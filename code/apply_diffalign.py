"""Add the differentiable-alignment arm to the trainer, behind DIFF_ALIGN.

Three edits, each anchored on a unique line so a moved file fails loudly rather than silently
patching the wrong place:

  the import          beside the other section modules
  the two targets     the longitudinal branch and the transverse one; when the flag is on the
                      target becomes a differentiable warp of the same canonical reference,
                      initialised at the moment fit so the arm starts where the method starts
  the backward        `_bw` already collects the priors and calls backward, so the alignment
                      step goes immediately after it
"""
import sys

p = "/workspace/rebuild/project3/code/src/train_voxel.py"
s = open(p).read()
n = 0


def sub(old, new, count=1):
    global s, n
    assert s.count(old) == count, (old[:70], s.count(old))
    s = s.replace(old, new)
    n += count


# 1. the import
sub("from section_match import section_target\n",
    "from section_match import section_target\nimport diffalign\n")

# 2. the two target sites
old_t = """ground_truth_tensor = (section_target(rendering, np.asarray(ref), alpha_r)
                                       if SECTION_MATCH else transform(ref).to(device))"""
new_t = """ground_truth_tensor = (section_target(rendering, np.asarray(ref), alpha_r)
                                       if SECTION_MATCH else transform(ref).to(device))
                if diffalign.ON:
                    # The same reference, placed by three learnable parameters instead of by
                    # moments, and left attached to the graph so the loss can move them.
                    ground_truth_tensor = diffalign.target(
                        f"v{i}", transform(ref).to(device), rendering)"""
sub(old_t, new_t)

old_h = """            ground_truth_tensor = (section_target(rendering, np.asarray(ref), alpha_r)
                                   if SECTION_MATCH else transform(ref).to(device))"""
new_h = """            ground_truth_tensor = (section_target(rendering, np.asarray(ref), alpha_r)
                                   if SECTION_MATCH else transform(ref).to(device))
            if diffalign.ON:
                ground_truth_tensor = diffalign.target(
                    f"h{i}", transform(ref).to(device), rendering)"""
sub(old_h, new_h)

# 3. the alignment step, wherever the loop backpropagates
sub("        loss.backward()\n", "        loss.backward()\n        diffalign.step()\n")

open(p, "w").write(s)
print(f"  {n} edits applied to {p}")
for k in ("import diffalign", "diffalign.target", "diffalign.step()"):
    print(f"    {k}: {s.count(k)}")
