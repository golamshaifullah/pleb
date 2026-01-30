Overview
========

The EPTA Data Combination Pipeline ("pleb") is a Python toolkit for combining
pulsar timing data across observing backends and generating diagnostics that
highlight timing-model quality, residual behavior, and data-set consistency.
It wraps standard timing workflows (via tempo2 outputs) and adds a structured
reporting layer aimed at PTA-style data sets, where long baselines and multiple
instruments are the norm. [Edwards2006]_ [Hobbs2006]_

If you are new to pulsar timing, the key object is the **timing residual**:
the difference between an observed pulse time-of-arrival (TOA) and the model
prediction. Residuals summarize how well the timing model explains the data,
and they are the primary input to PTA diagnostics and noise modeling. [Lorimer2005]_ [Stairs2003]_

The pipeline is designed to help you:

- Compare timing residuals and summary statistics across git branches or data
  combinations.
- Generate residual plots, covariance heatmaps, and change reports to track
  the impact of model or data updates.
- Run optional quality-control (QC) stages that detect outliers and transient
  behavior in residuals.
- Produce per-pulsar reports suitable for review by analysts new to the data set.

Pulsar timing context
---------------------

Pulsar timing models predict TOAs using a mixture of deterministic and
stochastic terms. Deterministic terms include spin-down, astrometry, and
binary motion; stochastic terms include red timing noise and dispersion-measure
(DM) variations. The pipeline itself does not perform full stochastic modeling,
but it surfaces residual patterns that commonly motivate those models. [Coles2011]_ [Keith2013]_

For example, residuals can be summarized by a reduced chi-square statistic:

.. math::

   \\chi^2_\\nu = \\frac{1}{N - p} \\sum_{i=1}^{N} \\frac{r_i^2}{\\sigma_i^2},

where :math:`r_i` are residuals, :math:`\\sigma_i` are TOA uncertainties, and
:math:`p` is the number of fitted parameters. Deviations in
:math:`\\chi^2_\\nu` or structure in residuals versus time, frequency, or
backend are indicators that the model is incomplete or the data contain
systematic effects. [Edwards2006]_ [Hobbs2006]_

Radio telescopes and backends
-----------------------------

PTA data sets combine observations from multiple radio telescopes and signal
processing backends. Backends can differ in bandwidth, channelization, and
time-tagging, which in turn affects residual scatter and systematic offsets.
This pipeline helps compare those backend-dependent behaviors by enforcing a
consistent metadata schema and plotting residuals grouped by backend. [Manchester2005]_

Where to start
--------------

If you are installing the package for the first time, begin with
:doc:`installation` and :doc:`quickstart`. For the timing concepts that appear
throughout the documentation, see :doc:`concepts`. For end-to-end workflows,
see :doc:`examples`.
