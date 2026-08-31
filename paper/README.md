# LateAct NeurIPS 2026 workshop paper

`main.tex` uses the official NeurIPS 2026 workshop style in double-blind mode.
The checked-in paper figures are sufficient to compile the PDF:

```bash
make TECTONIC=tectonic
```

To regenerate the response-curve figure from frozen JSON summaries, first set
`LATEACT_ARTIFACT_ROOT` to the directory containing the LateAct experiment
outputs, then run:

```bash
make figures
```

The submission PDF is `LateAct_NeurIPS2026_Workshop.pdf`. The main paper is six
pages, with references beginning on page 7, followed by the appendix and the
required NeurIPS checklist.
