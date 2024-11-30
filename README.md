# COMP558-Inpainting

by Colin Taylor, Nelsin Martin-Burnett, Henri Lemoine

Our COMP558 final project on image inpainting. We investigate classical inpainintg methods and compare them with Deep Learning (DL) inpainting methods, focusing on the tradeoff between accuracy and computational efficiency.

## Installation

This repository uses `uv`. Install [here](https://docs.astral.sh/uv/getting-started/installation/).

Install the repo with `git clone https://github.com/henri123lemoine/COMP558-Inpainting`.

## Usage

To generate up-to-date results for all experiments, run `uv run main.py`. Note that this will take a long time.
```bash
uv run main.py
```
To specify a specific experiment, run `uv run main.py --experiment <experiment_name>`.

Run
```bash
uv run -m src.experiments.utils.benchmark
```
To run the benchmarks. Results will be saved in `data/benchmark_results/`.

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

TODO: Add DL papers
