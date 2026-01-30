FAQ
===

What inputs does the pipeline expect?
-------------------------------------

The pipeline expects tempo2-compatible ``.par`` and ``.tim`` files organized
under a data root directory. Many users point ``home_dir`` at a repository
containing pulsar subdirectories with timing files and metadata. [Edwards2006]_

Why are my residuals structured versus frequency?
-------------------------------------------------

Frequency-dependent residuals often indicate DM variations or scattering
changes in the interstellar medium. These effects are chromatic and can be
diagnosed by plotting residuals versus observing frequency. [Keith2013]_
[Cordes2016]_ [You2007]_

What does a high reduced chi-square mean?
----------------------------------------

A reduced :math:`\\chi^2` significantly above 1 can indicate underestimated
TOA uncertainties, unmodeled noise processes, or missing timing parameters.
The pipeline surfaces this to guide further modeling or data inspection.
[Edwards2006]_ [Hobbs2006]_

Does the pipeline model red noise directly?
-------------------------------------------

No. The pipeline focuses on diagnostic reporting and QC. If you need explicit
red-noise modeling, use dedicated PTA analysis tools and treat the pipeline's
plots as an exploratory step. [Coles2011]_ [Lentati2014]_

How should I interpret backend-dependent offsets?
-------------------------------------------------

Backend offsets are commonly modeled as ``JUMP`` parameters. If you see
discontinuities between backend groups, check that backend metadata and jump
flags are consistent. [Hobbs2006]_

Where can I learn more about pulsar timing?
-------------------------------------------

Introductory references include the Handbook of Pulsar Astronomy and
Living Reviews articles on timing and relativistic tests. [Lorimer2005]_
[Stairs2003]_ [Taylor1992]_
