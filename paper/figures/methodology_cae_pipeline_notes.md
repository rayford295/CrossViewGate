# Methodology Figure Notes

This figure is designed as a top-conference-style method schematic for the
short paper, not as a decorative workflow diagram. Its job is to make the paper
argument easy to repeat:

> Conflict-Aware Evaluation (CAE) tests whether cross-view fusion resolves the
> cases where ground-only and overhead-only evidence disagree.

## Reading Order

1. **Paired evidence** establishes the input object: a paired disaster triage
   sample with a ground view, an overhead patch, and a binary damage label.
   The figure also names the spatial observation regime because the paper's
   explanation depends on target alignment, not only on disaster type.

2. **Comparable predictors** shows the controlled comparison. Street-only,
   remote-only, and cross-view models share the same backbone and split. The
   cross-view branch uses late fusion, but the visual emphasis stays on
   evaluation rather than model novelty.

3. **Conflict gate** is the methodological center. The conflict subset
   `C_tau` keeps cases where the two single-view probabilities differ by more
   than a threshold. This makes CAE a stress test rather than another average
   performance table.

4. **CAE readout** turns the stress test into paper claims: overall F1,
   conflict F1, `Delta_view`, and directional arbitration. The directional
   bars explain when the dominant view switches across spatial observation
   regimes.

## Suggested Caption

```latex
\caption{Conflict-Aware Evaluation (CAE) protocol. CAE starts from paired
ground and overhead observations, trains comparable street-only, remote-only,
and cross-view predictors, isolates examples where the single-view predictors
disagree, and reports whether cross-view fusion improves conflict-subset F1
over the stronger single-view baseline. Directional arbitration summarizes
which view corrects the other inside the conflict subset, linking model behavior
to spatial observation regime.}
```

## LaTeX Insert Snippet

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{../figures/methodology_cae_pipeline.pdf}
  \Description{Methodology figure showing paired evidence, comparable
  single-view and cross-view predictors, a conflict gate based on single-view
  probability disagreement, and a CAE scorecard with overall F1, conflict F1,
  cross-view conflict gain, and directional arbitration.}
  \caption{Conflict-Aware Evaluation (CAE) protocol.}
  \label{fig:methodology}
\end{figure*}
```

## Design Principle

The figure avoids presenting CAE as a new architecture. Reviewers should see a
clean evaluation contribution: the same paired-view model family is judged on
the subset where its value is most falsifiable.
