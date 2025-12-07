# Paper Directory

## Main Files
- `main.tex` - AIAA-formatted paper (Natural Manifold Formation Flying in Cislunar Space)
- `main.pdf` - Compiled paper (25 pages)
- `references.bib` - BibTeX bibliography (43 entries)

## Style Files
- `new-aiaa.cls` - AIAA journal class file
- `new-aiaa.bst` - AIAA bibliography style

## Assets
- `assets/` - Figures and results used in the paper (see assets/README.md)

## Compilation
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use the shorthand:
```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Paper Content Summary
- **Title**: Natural Manifold Formation Flying in Cislunar Space: Theory, Control, and Statistical Validation
- **Length**: 25 pages (double-spaced AIAA format)
- **Sections**: 12 (including Introduction, Related Work, System Model, Guidance, Navigation, Control, Learning-Based Control, Results, Discussion, Conclusion)
- **Theorems**: 3 formal theorems with proofs
- **Tables**: 5 (Monte Carlo results, controller comparison, failure modes, computational benchmark, mission budget)
- **Figures**: 1 (hybrid benchmark comparison)
- **References**: 43 citations (2019-2024, emphasis on 2023-2025)
