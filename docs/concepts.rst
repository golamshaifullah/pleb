Concepts
========

This section introduces core timing concepts that appear throughout the
pipeline outputs. The goal is to make the diagnostics readable for users who
are new to PTA data analysis.

Timing residuals
----------------

Timing residuals are the differences between observed TOAs and model
predictions. For a timing model with parameters :math:`\\theta`, the residuals
are

.. math::

   r_i = t^{\\mathrm{obs}}_i - t^{\\mathrm{model}}_i(\\theta).

Residuals near zero with no obvious structure indicate that the timing model
captures the dominant deterministic behavior. Systematic patterns can point to
missing physics (e.g., binary derivatives) or unmodeled propagation effects.
[Edwards2006]_ [Hobbs2006]_

Reduced chi-square
------------------

The pipeline reports reduced chi-square to summarize goodness of fit:

.. math::

   \\chi^2_\\nu = \\frac{1}{N - p} \\sum_{i=1}^{N} \\frac{r_i^2}{\\sigma_i^2},

where :math:`\\sigma_i` are TOA uncertainties and :math:`p` is the number of
fitted parameters. Values significantly greater than 1 often indicate that
noise is underestimated or that the model is incomplete; values much smaller
than 1 can indicate overestimated uncertainties or absorbed systematics.
[Edwards2006]_ [Hobbs2006]_

Dispersion measure (DM) variations
----------------------------------

Propagation through the ionized interstellar medium introduces a frequency-
dependent delay:

.. math::

   \\Delta t_{\\mathrm{DM}} = K_{\\mathrm{DM}} \\frac{\\mathrm{DM}}{f^2},

where :math:`K_{\\mathrm{DM}} \\approx 4.148808\\times 10^3\\,\\mathrm{MHz}^2\\,
\\mathrm{pc}^{-1}\\,\\mathrm{cm}^3\\,\\mathrm{s}` and :math:`f` is observing
frequency in MHz. Time-variable DM produces chromatic residuals and is a
dominant red-noise process in PTA data. [Lorimer2005]_ [Keith2013]_ [You2007]_

The pipeline’s QC outputs can highlight residual structure versus frequency,
which is often a signature of DM variations or scattering changes.
[Cordes2016]_

Red noise and correlated residuals
----------------------------------

PTA data commonly show correlated ("red") noise from spin irregularities,
interstellar propagation, or unmodeled systematics. A common phenomenological
model is a power-law power spectral density:

.. math::

   P(f) = A^2 \\left(\\frac{f}{f_0}\\right)^{-\\gamma},

with amplitude :math:`A`, spectral index :math:`\\gamma`, and reference
frequency :math:`f_0`. The pipeline does not fit these models directly, but
residual plots and outlier diagnostics can guide when such modeling is needed.
[Coles2011]_ [Lentati2014]_

Backend offsets and jumps
-------------------------

When combining radio backends, relative offsets are often modeled as "JUMPs".
If backend identifiers are inconsistent or missing in the metadata, the
pipeline's dataset-fix stage can normalize flags and report mismatches. This
helps avoid spurious steps in residuals and improves cross-backend
comparisons. [Hobbs2006]_ [Manchester2005]_

Pulsar timing in context
------------------------

Pulsar timing arrays leverage precisely timed millisecond pulsars to detect
gravitational waves and probe interstellar effects. The pipeline’s reporting
layer is intended to support those analyses by making data-quality and model
changes visible early in the workflow. [Hobbs2010]_ [Lentati2016]_
