# Paper specification

Binding on `index.html`, `ovoxel.html` and `paper/ours/main.tex`. A violation is a defect, not a
matter of taste.

## 1. Output form
- No greeting, no preamble, no closing summary, no meta-announcement.
- No unsolicited follow-up proposals inside the paper body.

## 2. Minimalism
- Occam: one sentence where one will do; short clauses over compound subordination.
- No filler ("It is worth noting that", "In order to achieve this goal").
- Every sentence carries a physical fact, a geometric definition, an algorithmic step or a number.
- Prefer formulae, bullets and tables to paragraphs.

## 3. Claims and scope
- The operator is **planar slicing/cutting**. "General virtual dissection", "arbitrary tearing",
  "fracture" are forbidden: not demonstrated.
- No subjective modifiers: *superior, extremely accurate, novel, seamless, groundbreaking*.
  Replace with the measured quantity (machine precision $\epsilon$, 0.14% mass error,
  0.07% unpainted).
- State failure boundaries. Ghosting when a specimen's non-rigid distortion exceeds the lattice
  tolerance. No claim on absolute Young's modulus.

## 4. Formulation
- One symbol table for the whole paper: cell $C_i$, feature $\mathbf{f}_i \in \mathbb{R}^d$,
  plane $\Pi_k = (\mathbf{n}_k, d_k)$. No collisions, no undefined reference.
- No invented unconstrained parameter, no loss without a measurement behind it, no undefined
  variable.
- Terms must meet their own preconditions. No "information bottleneck" without a rate-distortion
  term; no "stereological spectral invariance" on a non-stationary field. Use 2-manifold, chordal
  intersection, Euler characteristic where they apply.
- Pre-cut alignment is a global geometric problem over intersection constraints,
  $\min_{\pi,\{\delta_k\}}$, not "sort by eye and greedily match".
- Plane jitter is a Monte Carlo expectation over a continuous plane distribution, and that is what
  gives blind-spot coverage.
- LaTeX for every formula, with domains.

## 5. Figures
- Anything involving spatial intersection, cell state transition, chordal alignment or polyhedral
  decomposition is drawn, not described.
- **No high-level flowcharts.** Box-and-arrow diagrams do not substitute for the data's state.
- Step-by-step state decomposition, subfigures (a)-(d): initial state, local intersection,
  topological decomposition, final output. Each labelled with the symbols of the text
  ($\Pi_k$, $e_i$, $\mathbf{v}_j$).
- Captions self-contained: subject, what each of (a)-(d) shows, the key number.
- Cross-method comparisons share viewpoint, plane, lighting and resolution. Defects in inset boxes
  of one line width, annotated with the number. Scalar fields carry a perceptually uniform
  colorbar with its range.

## 6. Evaluation
- Pre-empt the rebuttal: argue in Related Work or Discussion why classical 3D texture synthesis and
  neural implicit fields were not used, on measured geometric cost.
- Ablations isolate one variable each, with the marginal gain quantified.
- Report the full cost matrix: alignment (ms), peak optimiser memory (MiB), per-cut latency (ms),
  triangles generated.
- Keep leave-one-out and held-out protocols. A model that has seen the slice is not a baseline
  for it.
- Topology and complexity as propositions: equal-size subdivision admits no hanging edge;
  connected-component extraction on the integer lattice is $\mathcal{O}(N_{\text{crossed}})$.
- No "code coming soon", no dead link. Give every hyperparameter, the lattice resolution $h$ and
  its level count, and the feature dimension $d$.
