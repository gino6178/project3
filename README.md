# Cube Interior + O-Voxel Cut Surface

Project page. The site is served from the repository root; `index.html` and `assets/` are the
whole of it.

The hidden interior of a cuttable 3D asset is stored as a coarse structured cube grid holding
occupancy and an appearance latent, rather than as opaque Gaussian primitives. Cells are
subdivided only where a cut passes, and only the newly exposed cross-section becomes an
O-Voxel surface.

Numbers on the page are measured under one protocol on one machine and include the results
that did not work.
