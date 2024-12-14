# COMP558-Inpainting

by Colin Taylor, Nelsin Martin-Burnett, Henri Lemoine

Our COMP558 final project on image inpainting. We investigate classical inpainting methods with a special focus on Navier-Stokes, and compare them together.

## Installation

This repository uses `uv`. Install [here](https://docs.astral.sh/uv/getting-started/installation/).

Install the repo with `git clone https://github.com/henri123lemoine/COMP558-Inpainting`.

## Usage

### Running Individual Algorithms

You can run any of the implemented algorithms directly on test images. Each algorithm accepts parameters like image path, mask path, and algorithm-specific settings.

```bash
# Efros-Leung
uv run -m src.algorithms.classical.efros_leung --image-path path/to/image.png --mask-path path/to/mask.png --window-size 13 --error-threshold 0.15

# Navier-Stokes
uv run -m src.algorithms.classical.navier_stokes --image-path path/to/image.png --mask-path path/to/mask.png --max-iterations 1000 --dt 0.02

# PatchMatch
uv run -m src.algorithms.classical.patch_match --image-path path/to/image.png --mask-path path/to/mask.png --patch-size 13 --num-iterations 5
```

Common parameters for all algorithms:
- `--image-path`: Path to input image
- `--mask-path`: Path to mask image
- `--scale-factor`: Scale factor for input image (default: 1.0)
- `--save-output`: Whether to save results (default: True)
- `--greyscale`: Process as greyscale image (default: False)

### Running Benchmarks

To run the benchmarks comparing all algorithms:
```bash
uv run -m src.experiments.benchmark
```

This will:
1. Generate synthetic test cases (lines, shapes, textures)
2. Load real test images from the dataset
3. Run all algorithms on each test case
4. Generate comparison visualizations and metrics
5. Save results to `data/benchmark_results/`

## Papers

### General

`https://www.math.ucla.edu/~lvese/PAPERS/01211536.pdf`
`https://jiaya.me/file/all_final_papers/imgrep_final_cvpr03.pdf`
`https://www.math.ucla.edu/~bertozzi/papers/cvpr01.pdf`
`https://www.researchgate.net/publication/220720382_Image_inpainting`

### Exemplar-based

`https://ieeexplore.ieee.org/document/4337762`
`https://ieeexplore.ieee.org/document/1323101`

### Textures

`https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-iccv99.pdf` (see `efros_leung.py`)
`https://graphics.stanford.edu/papers/texture-synthesis-sig00/texture.pdf`

### Structures

`https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf`

### Deep Learning

TODO: Add DL papers (Edit: we didn't end up implementing any successful DL methods)
