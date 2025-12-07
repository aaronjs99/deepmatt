# Paper Assets

This folder contains figures and results used in the paper.

## Files

- `hybrid_benchmark.png` - Benchmark comparison showing Transformer+PPO hybrid approach vs baseline PPO

## Usage

When updating results:
1. Run result generation scripts in `src/deepmatt/validation/`
2. Copy updated figures from `results/` to `paper/assets/`
3. Recompile paper with `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

## Note

All figures in this folder are referenced directly in `main.tex` using relative paths like `assets/filename.png`.
