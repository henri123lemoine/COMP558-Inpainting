# Plots

- navier-stokes/
  - good/
    - equivalent-to-el-but-750x-faster.png -> Shows navier-stokes doing as well as the other methods, but 750x faster
    - gradients.png -> Shows navier-stokes doing well on gradients
    - building.png -> Shows navier-stokes doing well on removing text
  - bad/
    - large-checkerboard-heatmap-1000-iter.png -> shows navier stokes doing poorly on checkboard patterns
    - checkerboard-brush-heatmap.png -> same as above, different mask and slightly smaller board.
    - side-mask.png -> Shows navier stokes doing poorly for masks that reach the sides, probably due to our having implemented the boundary conditions incorrectly
  - iter/
    - metrics_by_category.pdf -> Shows metrics for 1 vs 10 vs 100 vs 1000 iterations
    - diminishing-returns.png -> Shows the diminishing returns of increasing the number of iterations
    - gradient-iter-1.png & gradient-iter-1000.png -> Shows the difference between 1 and 1000 iterations on the gradient example
    - the-scream-iter-1.png & the-scream-iter-1000.png -> Same but for The Scream by Edvard Munch
- patchmatch/
  - good.png -> Shows good performance of PatchMatch on the hall example (due to it being mostly structures and textures)
  - bad.png -> shows it being bad at gradients
- old/ -> ignore these
