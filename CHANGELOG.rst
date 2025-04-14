0.4.0 - TBD
-----------

- Changed ``exposure_time`` parameter to ``duration`` to be more agnostic to the type of observation.
- Removed the ``TargetAlwaysUp`` warning from astroplan. This is caused when looking for the rise and set times
  of a target that is always up, but this is not a problem for the scheduler and might be confusing for the user.
- Fixed a small bug in the import of plotly. The idea is that it is an optional dependency and should not be
  imported if not needed. It is only imported when the ``plot_interactive()`` method is called.
- Migrated build system from ``setup.py`` to ``pyproject.toml``
- Added example plots of a plan to the README file.
- Removed ``Instrument`` class. The instrument is now an optional parameter of the ``Observation`` class.
  - This is to allow the cases where a Program might use more than one instrument. And to make it more flexible.


0.3.8 - 26.08.2024
------------------

- Changed the URL of the Changelog to be the one of the ReadTheDocs page.
- Fixed the readthedocs autodocs section with full requirements for compilation.
- Fixed bugs in a few tests that were failing.

0.3.7 - 12.08.2024
------------------

- New readthedocs documentation with installation instructions, changelog, API reference and examples.
- Cleanup of some docstrings and comments.

0.3.6 - 08.08.2024
------------------

- Versions 0.3.4, 0.3.5, and 0.3.6 were improvements in the documentation and the README file.

0.3.3 - 07.08.2024
------------------

- Added tests for all classes from ``scheduler_components``.
- Cleaner README with updated links.
- Removed the ``__eq__`` methods for ``Program`` as it wasn't actually needed and was causing problems.

0.3.2 - 06.08.2024
------------------

- Added matplotlib to the requirements and removed plotly from the top imports of scheduler_components.py.
  This way plotly will only be imported when the plot_interactive() method is called.
- Added some ``__repr__`` and ``__eq__`` methods to the ``Program`` and ``Target`` classes.
- Fixed a bug in the Instrument class where the ``__repr__`` was referencing an old attribute name.
- Get the logo from the GitHub URL instead of the local file of the repo. This way the logo can be
  displayed in PyPI.

0.3.1 - 05.08.2024
------------------

- Added a ``night_duration`` attribute to the night object. A ``datetime.timedelta`` object that represents 
  the duration of the night.

0.3.0 - 06.05.2024
------------------

- Implemented a Local Search Heuristic that optimizes the overall overhead time of a plan.
- The same for optimizing the score after the overheads are minimized.
- In the Overheads object, the slew can now take into account cable wrap limits.
- Simulation module was updated and works again.
- Added a new phase-specific merit which is an improved version of the periodic gaussian merit.
  In this merit the sigma is actually the standard deviation of the gaussian in phase space.
- Added ``to_df()`` and ``to_csv()`` methods to Plan which create a pandas DataFrame of the Plan and saves
  it to a CSV file, respectively.
- Removed the ``print_plan()`` function and replaced it by printing the string representation of the 
  dataframe created by ``to_df()``.

0.2.0 - 07.03.2024
------------------

- Added a bunch of features, like the big overhaul of the Observation object and moving many of
  the tasks to the Scheduler.
- Added option to set an end time for a schedule (to only schedule part of the night).
- Many other things I don't remember (I just created this file to keep track of the changes).

0.1.0 - January 2024
--------------------

- Initial release