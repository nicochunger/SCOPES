.. scopes documentation master file, created by
   sphinx-quickstart on Thu Aug  8 16:33:10 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SCOPES's documentation!
==================================

SCOPES is a Python package designed to automate and optimize the scheduling of astronomical observations for ground-based telescopes equipped with dedicated instruments like cameras or spectrographs. It helps allocate shared telescope time among various observational programs, each with unique scientific objectives and priorities, into a single night. SCOPES ensures that the operation of the telescope is both effective and efficient by maximizing the use of available time while adhering to observational and scientific constraints.

Features
--------

- **Multi-Program Scheduling:** SCOPES can handle scheduling for multiple observational programs, ensuring fair distribution of telescope time based on the scientific priorities of each program.
- **Multi-Instrument Support:** SCOPES allows for scheduling across different instruments on the same telescope, accommodating complex observational setups with ease.
- **Fairness, Sensibility, and Efficiency:** Built around a robust framework, SCOPES optimizes schedules by balancing fairness (equitable time distribution), sensibility (observing conditions and constraints), and efficiency (optimal timing for high data quality).
- **Customizable Merits:** Users can define custom merit functions that influence how targets are selected based on criteria such as airmass, altitude, or custom-defined parameters.
- **Flexible Overheads Management:** SCOPES allows users to define and customize overheads such as telescope slew time, instrument changes, and other operational constraints to better reflect the actual observing conditions of your setup.
- **Comprehensive Visualization:** SCOPES provides detailed visualization tools to plot schedules, offering insights into altitude vs. time, polar plots of telescope movement, and azimuth plots to ensure an efficient schedule is achieved.


Intended Users
--------------

SCOPES is intended for telescope managers, administrators, and astronomers who need to optimize the use of telescope time for multiple programs. While it can be used by individual observers, the setup may be more demanding if used for only a few isolated nights.

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   installation
   changelog
   
.. toctree::
   :maxdepth: 2
   :caption: API reference

   scopes

.. TOC trees for example notebooks

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples