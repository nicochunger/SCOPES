![logo](https://raw.githubusercontent.com/nicochunger/SCOPES/main/logo.png)

# SCOPES

## **S**ystem for **C**oordinating **O**bservational **P**lanning and **E**fficient **S**cheduling

SCOPES is a Python package designed to automate and optimize the scheduling of astronomical observations for ground-based telescopes equipped with dedicated instruments like cameras or spectrographs. It helps allocate shared telescope time among various observational programs, each with unique scientific objectives and priorities, into a single night. SCOPES ensures that the operation of the telescope is both effective and efficient by maximizing the use of available time while adhering to observational and scientific constraints.

## Features
- **Multi-Program Scheduling:** SCOPES can handle scheduling for multiple observational programs, ensuring fair distribution of telescope time based on the scientific priorities of each program.
- **Multi-Instrument Support:** SCOPES allows for scheduling across different instruments on the same telescope, accommodating complex observational setups with ease.
- **Fairness, Sensibility, and Efficiency:** Built around a robust framework, SCOPES optimizes schedules by balancing fairness (equitable time distribution), sensibility (observing conditions and constraints), and efficiency (optimal timing for high data quality).
- **Customizable Merits:** Users can define custom merit functions that influence how targets are selected based on criteria such as airmass, altitude, or custom-defined parameters.
- **Flexible Overheads Management:** SCOPES allows users to define and customize overheads such as telescope slew time, instrument changes, and other operational constraints to better reflect the actual observing conditions of your setup.
- **Comprehensive Visualization:** SCOPES provides detailed visualization tools to plot schedules, offering insights into altitude vs. time, polar plots of telescope movement, and azimuth plots to ensure an efficient schedule is achieved.


## Intended Users
SCOPES is intended for telescope managers, administrators, and astronomers who need to optimize the use of telescope time for multiple programs. While it can be used by individual observers, the setup may be more demanding if used for only a few isolated nights.


## Installation

To install SCOPES, open your terminal and run the following command:

`pip install scopes-astro`

<small>(just `scopes` was unfortunately already taken in PyPI)</small>

This command will install SCOPES along with its necessary dependencies, including `numpy`, `pandas`, `matplotlib`, `astropy`, `astroplan`, `tqdm`, `pytz`, and `timezonefinder`. 

Ensure you have Python 3.8 or later installed.

## Getting Started

To get started, import the package into your Python code:

`import scopes`

### Examples

You can find an example Jupyter notebook that details how to use SCOPES from a simple setup to a full night of observations: [`scopes_example.ipynb`](https://github.com/nicochunger/SCOPES/blob/main/scopes_example.ipynb)

## Documentation

For complete documentation, including detailed explanations of the scheduling strategy, merit functions, and advanced usage, visit the [SCOPES Documentation](https://github.com/nicochunger/SCOPES/blob/main/SCOPES_documentation.pdf).

## Acknowledgements

SCOPES is based on the scheduling framework outlined by [van Rooyen et al. 2018](https://doi.org/10.1117/12.2311839) and builds upon various open-source packages, including `astroplan` and `astropy`.

## Citation

If you use SCOPES in your research, please cite it appropriately. Publication coming soon. In the meantime just link to this GitHub repository.

## Frequently Asked Questions

1. **How can I prioritize certain targets over others in my observation schedule?**
    - You can prioritize targets by assigning them a priority level (0: Top priority, 1: High, 2: Normal, 3: Low) within their respective programs. SCOPES uses a combined priority system that considers both program and target priorities to influence the scheduling.

2. **What should I do if my observing program has used more or less time than allocated?**
    - Use the `TimeShare` merit, which accounts for the percent difference between allocated and used time, to adjust the schedule accordingly. This merit only works when used over more than one night. It will balance time distribution over several nights to ensure fair time allocation among programs.

3. **How can I ensure that my observations are scheduled only during the night?**
    - SCOPES provides an `AtNight` veto merit that ensures observations are only scheduled during nighttime as defined by civil, nautical, or astronomical twilight limits.

4. **What is the difference between veto merits and efficiency merits?**
    - Veto merits (sensibility merits) are constraints that must be met for a target to be considered for observation; if any veto merit evaluates to zero, the observation is discarded. Efficiency merits optimize the timing of observations but do not prevent them if they evaluate to zero.

5. **Can I define my custom merit functions in SCOPES?**
    - Yes, SCOPES allows you to create custom merits by defining functions that operate on available attributes in the `Observation` class. These custom merits can be integrated into the scheduling process alongside pre-defined merits.

6. **How can I manage telescope overheads like slew time or instrument changes?**
    - SCOPES includes an `Overheads` class where you can define telescope-specific overheads, including slew rates for azimuth and altitude. You can also add custom overhead functions if your telescope has unique requirements.

7. **What should I do if my target needs to be observed at a specific phase of one of its planet's orbit?**
    - You can use the `PhaseSpecific` efficiency merit, which allows scheduling based on the phase of a periodic event, such as a planet's orbit. Adjust the merit's parameters to match the desired observational phase.

8. **Can I schedule observations that must be completed before a specific time?**
    - Yes, SCOPES offers an `EndTime` veto merit, which ensures that observations do not extend beyond a specified time limit. This is useful for ensuring that time-sensitive observations are completed within a required timeframe.

9. **How can I visualize the final schedule generated by SCOPES?**
    - After the schedule is created, it will return an instance of the `Plan` class which has several methods to visualize the schedule in various formats, including altitude vs. time plots, polar plots, and azimuth plots. These visualizations help assess the efficiency of the schedule.

10. **What if I need to create a custom scheduler for my unique requirements?**
    - SCOPES allows the creation of custom schedulers by extending the base `Scheduler` class. You can implement your own scheduling logic and integrate it with SCOPES's framework to meet specific needs.
