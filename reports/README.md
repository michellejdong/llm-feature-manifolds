# Research reports

[From Hidden States to Causal Tests](calendar-study-walkthrough.pdf) is a visual walkthrough of how the calendar study ranks unlabeled geometry, retrains a dictionary, and tests activation edits with controls. Its figures use the measured tables in the repository. The PDF includes page links and links to the underlying data, notebooks, and papers.

To rebuild the PDF from the repository root, install ReportLab and run the source.

```bash
python -m pip install reportlab
python reports/build_calendar_report.py
```

[Calendar results and data](../RESULTS.md) · [Notebook directory](../notebooks/)
