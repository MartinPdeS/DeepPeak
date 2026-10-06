Analyze experimental traces
===========================

Load an exported trace, remove its DC component, optionally filter it, and
inspect detected peaks:

.. code-block:: python

   from DeepPeak.io import CsvTrace

   trace = CsvTrace("measurement.csv")
   trace.remove_dc()
   trace.low_pass_filter(bandlimit=1_000)
   positions, amplitudes, widths = trace.find_peaks(height="5sigma")
   figure = trace.plot_overview(end_idx=50_000, nbins=80)
   figure.savefig("trace-overview.svg", bbox_inches="tight")

Before batch analysis, verify the time unit, sampling interval, header names,
and polarity on several files. Fit thresholds on a development subset and
report performance on held-out traces to avoid optimistic estimates.
