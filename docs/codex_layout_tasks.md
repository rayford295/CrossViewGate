# Codex Layout Tasks — LaTeX Formatting Fixes

> Written for Codex. Fix all 5 issues below in `paper/sigspatial2026_draft.tex`.
> No new content needed — these are purely formatting/layout changes.
> After each fix, recompile with `pdflatex` and verify visually.

---

## Fix 1 — Remove "Draft updated" timestamp from page headers

**Problem:** "Draft updated: 2026-04-24 22:35 CDT" appears in the running header
on every odd-numbered page, which is unprofessional and incorrect for a conference submission.

**Root cause:** A `\date{Draft updated: ...}` or `\subtitle{...}` command in the
preamble. ACM `sigconf` puts `\date{}` content into the odd-page running header.

**Action:** Find the line containing `\date{` in the preamble and change it to:

```latex
\date{}
```

If no `\date{}` line exists, check for `\subtitle{}` and remove it entirely.

---

## Fix 2 — Replace "Conference'17, July 2017, Washington, DC, USA" placeholder

**Problem:** Every page header/footer shows the ACM template default placeholder
"Conference'17, July 2017, Washington, DC, USA" instead of the actual venue.

**Root cause:** `\acmConference{}` is not defined in the preamble, so the class
falls back to its built-in placeholder.

**Action:** Add the following four lines to the preamble, immediately after the
`\setcopyright{none}` line:

```latex
\acmConference[SIGSPATIAL'26]{ACM SIGSPATIAL GIS}{November 2026}{Atlanta, GA, USA}
\copyrightyear{2026}
\acmYear{2026}
\acmISBN{}
```

---

## Fix 3 — Make \pagestyle{plain} actually take effect

**Problem:** Despite `\pagestyle{plain}` in the preamble, page headers are still
showing conference name and author list.

**Root cause:** ACM's `\maketitle` resets the page style internally, overriding
whatever was set in the preamble.

**Action:** Add `\pagestyle{plain}` on the line immediately after `\maketitle`:

```latex
\maketitle
\pagestyle{plain}   % must repeat here — \maketitle resets the page style
```

---

## Fix 4 — Fix Table 4 (adaptive cascade) overflowing into Discussion column

**Problem:** On page 8, the adaptive cascade results table has 5 columns
(Split, δ, Stage-2 rate, Full CV F1, Cascade F1) and is placed in a single-column
`table` environment. It is too wide for one column and its content overflows into
the Discussion section on the right.

**Action:** Change the table environment from single-column to full-width:

```latex
% BEFORE
\begin{table}[t]
  \caption{Adaptive inference cascade results. ...}
  \label{tab:cascade}
  ...
\end{table}

% AFTER
\begin{table*}[t]
  \caption{Adaptive inference cascade results. ...}
  \label{tab:cascade}
  ...
\end{table*}
```

If the table content itself is still too wide after this change, add `\small`
immediately after `\begin{table*}[t]`:

```latex
\begin{table*}[t]
\small
  \caption{...}
```

---

## Fix 5 — Eliminate large blank space on page 9 (right column)

**Problem:** The Conclusion section is pushed into the left column of page 9,
leaving the right column almost entirely empty. This happens because multiple
`figure*` (full-width) floats accumulate and push text to the wrong position.

**Action:** Add `\clearpage` on the line immediately before `\section{Conclusion}`:

```latex
\clearpage
\section{Conclusion}
```

`\clearpage` forces LaTeX to flush all pending floats before starting the new
section, preventing the blank-column problem.

---

## Verification Checklist

After making all five changes and recompiling:

- [ ] No "Draft updated" text appears anywhere in page headers
- [ ] "Conference'17, July 2017, Washington, DC, USA" is replaced with SIGSPATIAL info
- [ ] Pages show only page numbers in the footer (no author/conference in headers)
- [ ] Table 4 (adaptive cascade) renders cleanly without overlapping the Discussion text
- [ ] Page 9 right column is not mostly blank; Conclusion fills both columns normally
- [ ] Total page count is still ≤ 10 pages + references (ACM SIGSPATIAL limit)

---

## Compile Command

```bash
cd paper/
pdflatex sigspatial2026_draft.tex
bibtex sigspatial2026_draft
pdflatex sigspatial2026_draft.tex
pdflatex sigspatial2026_draft.tex
```

Run `pdflatex` three times after `bibtex` to ensure all cross-references and
floats settle correctly.
