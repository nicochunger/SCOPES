from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from uuid import uuid4

import astropy.units as u
import numpy as np
import pandas as pd
from astroplan import AirmassConstraint, AltitudeConstraint, AtNightConstraint, is_observable
from astropy.coordinates import SkyCoord
from astropy.time import Time
from tqdm.auto import tqdm

import merits
import scheduler

# from scheduler import generateQ
from class_definitions import Merit, Night, Observation, Plan, Program, Target

# from helper_functions import build_observations, load_program


class Simulation:
    def __init__(self, start_date, end_date, observer, night_within, scheduler_algorithm):
        self.start_date = start_date
        self.end_date = end_date
        self.observer = observer
        self.night_within = night_within
        self.unique_id = uuid4()
        self.scheduler = scheduler_algorithm
        self.save_folder = f"simulation_results/sim_{self.unique_id.hex}"
        # Create the directory for saving the results
        Path(self.save_folder).mkdir(parents=True, exist_ok=True)
        self.programs = {}
        # Define default merits
        self.default_merits = [
            Merit("Airmass", merits.airmass, merit_type="veto", parameters={"max": 1.8}),
            Merit("Altitude", merits.altitude, merit_type="veto"),
            Merit("AtNight", merits.at_night, merit_type="veto"),
            # Merit("Culmination", merits.culmination, merit_type="efficiency"),
            Merit("CulMapping", merits.culmination_mapping, merit_type="efficiency"),
            # Merit("Egress", merits.egress, merit_type="efficiency"),
            Merit("TimeShare", merits.time_share, merit_type="fairness"),
        ]
        self.nights = []

    def build_nights(self):
        """
        Builds a list of nights from the start date to the end date.
        """
        self.nights = []
        for i in range((self.end_date - self.start_date).days + 1):
            night = Night(self.start_date + timedelta(days=i), self.night_within, self.observer)
            self.nights.append(night)
        return self.nights

    def add_program(self, prog, file):
        """
        Builds the list of targets from a file.
        """
        print(f"Adding program: {prog.progID}{prog.instrument}")
        targets = []
        # Load the program targets
        cat = pd.read_csv(file)
        for _, tar in cat.iterrows():
            merit_list = self.default_merits.copy()
            skycoord = SkyCoord(tar["ra"], tar["dec"], unit=(u.hourangle, u.deg))
            target = Target(
                tar["catalog_name"],
                prog,
                coords=skycoord,
                priority=tar["priority"],
                exposure_time=tar["texp"] / 86400,
            )

            # Airmass merit
            if "airmass" in tar.index:
                # If the target has a custom airmass limit, update the airmass merit
                merit_list[0].parameters["max"] = tar["airmass"]

            # Determine the chosen merits for the target
            if "period" in tar.index:
                # Remove the culmination and egress merit
                merit_list.pop(3)
                # merit_list.pop(4)
                # Add the PhaseSpecific merit
                phases = []
                if tar["phase1"] > 0:
                    phases.append(tar["phase1"])
                if tar["phase2"] > 0:
                    phases.append(tar["phase2"])

                merit_list.append(
                    Merit(
                        "PhaseSpecific",
                        merits.periodic_gaussian,
                        merit_type="efficiency",
                        parameters={
                            "epoch": tar["epoch"],
                            "period": tar["period"],
                            "sigma": 0.1,
                            "phases": phases,
                        },
                    )
                )

            for merit in merit_list:
                target.add_merit(merit)

            # Add target to list
            targets.append(target)

        # Add to the programs dictionary
        self.programs[prog] = {"targets": targets, "file": file, "df": cat}

    def determine_observability(self, night):
        """
        Determines the observability of all targets for the chosen night. It then filters
        the targets based on the cadence constraints.
        And then returns the list of observable targets.
        """
        print("Determining observability of targets...")
        constraints = [
            # AltitudeConstraint(20 * u.deg, 97 * u.deg),
            AirmassConstraint(1.8, 1.0),
            # AtNightConstraint.twilight_nautical(),
        ]
        targets = []
        time_range = Time([night.obs_within_limits[0], night.obs_within_limits[1]], format="jd")
        for prog in self.programs:
            tar_coords = []
            for target in self.programs[prog]["targets"]:
                # Get the coords of the target
                tar_coords.append(target.coords)
            # Determine the observability of the targets
            keep = is_observable(constraints, night.observer, tar_coords, time_range=time_range)

            # Check if targets are due to be observed (cadence constraints)
            cat = self.programs[prog]["df"]
            if "cadence" in cat.columns:
                # Get last observation of all targets
                last_observation_dates = cat["catalog_name"].apply(
                    lambda x: self.observation_history.loc[
                        self.observation_history["target"] == x
                    ]["obs_time"].max()
                    if x in self.observation_history["target"].unique()
                    else 0
                )
                cat_mod = (
                    cat.assign(last_obs=last_observation_dates)
                    .assign(
                        pct_overdue=lambda df: (
                            night.obs_within_limits[0] - (df["last_obs"] + df["cadence"])
                        )
                        / df["cadence"]
                    )
                    .assign(score=lambda df: df["priority"] / df["pct_overdue"])
                )
                # Filter cat_mod for observable targets
                observable_cat_mod = cat_mod[keep]

                # Filter for pct_overdue > 0 and sort by score in ascending order
                filtered_sorted = observable_cat_mod[
                    observable_cat_mod["pct_overdue"] > 0
                ].sort_values(by="score", ascending=True)

                # Take the top 50 targets
                top_50 = filtered_sorted.head(200).sample(min([50, len(filtered_sorted)]))

                # Create a boolean series for marking top 50 targets
                top_50_marks = cat_mod.index.isin(top_50.index)

                # Update keep array
                keep = top_50_marks

            # Get observable targets
            obs_targets = list(np.array(self.programs[prog]["targets"])[keep])
            targets += obs_targets
        return targets

    def build_observations(self, targets: List[Target], night: Night) -> List[Observation]:
        """
        Builds a list of observations for a given list of targets, exposure time, and file name.
        """
        print("Building observations...")
        observations = [
            Observation(target, night.obs_within_limits[0], target.exposure_time, night)
            for target in tqdm(targets)
        ]
        return observations

    def build_plan(self, observations: List[Observation]) -> Plan:
        """
        Builds a plan from a list of observations.
        """
        print("Building plan...")
        scheduler_instance = self.scheduler(observations[0].night.obs_within_limits[0])

        # Create the plan
        plan = scheduler_instance.run(observations, max_plan_length=None, K=1)
        return plan

    def update_tracking_tables(self, plan: Plan):
        """
        Updates the tracking tables for the observations and nights.
        """
        # Update the observation history
        new_observation_history = pd.DataFrame(
            {
                "target": [obs.target.name for obs in plan.observations],
                "program": [
                    f"{obs.target.program.progID}{obs.target.program.instrument}"
                    for obs in plan.observations
                ],
                "obs_time": [obs.start_time for obs in plan.observations],
                "texp": [obs.exposure_time for obs in plan.observations],
                "night": [obs.night.night_date for obs in plan.observations],
                "score": [obs.score for obs in plan.observations],
            }
        )
        self.observation_history = pd.concat(
            [self.observation_history, new_observation_history]
        ).reset_index(drop=True)

        total_prog_time = self.observation_history.groupby("program")["texp"].sum()
        tonights_obs_time = plan.observation_time.total_seconds() / 86400
        if len(self.night_history) == 0:
            tot_obs_time_d = tonights_obs_time
        else:
            tot_obs_time_d = (
                self.night_history["observation_time"].sum().total_seconds() / 86400
            ) + tonights_obs_time
        relative_used_time = total_prog_time / tot_obs_time_d
        # Update the night history
        new_night_history = pd.DataFrame(
            {
                "night": [plan.observations[0].night.night_date],
                "plan_score": [plan.score],
                "plan_length": [len(plan.observations)],
                "observation_time": [plan.observation_time],
                "overhead_time": [plan.overhead_time],
            }
        )
        for prog in self.programs:
            new_night_history[f"{prog.progID}{prog.instrument}_allocated"] = [
                prog.time_share_allocated
            ]
            try:
                prog_time_used = [relative_used_time[f"{prog.progID}{prog.instrument}"]]
            except KeyError:
                # In case this program was not observed yet
                prog_time_used = [0]

            new_night_history[f"{prog.progID}{prog.instrument}_used"] = prog_time_used
            # Update the program's time share
            prog.update_time_share(prog_time_used)

        self.night_history = pd.concat([self.night_history, new_night_history]).reset_index(
            drop=True
        )

    def run(self):
        """
        Runs the simulation.
        """
        # Build the nights
        self.build_nights()

        if self.programs == {}:
            print("No programs have been added to the simulation.")
            return

        # Initialize the tracking tables for the observations and nights
        self.observation_history = pd.DataFrame(
            columns=["target", "program", "obs_time", "texp", "night", "score"]
        )
        night_history_cols = [
            "night",
            "plan_score",
            "plan_length",
            "observation_time",
            "overhead_time",
        ]
        for prog in self.programs:
            night_history_cols.append(f"{prog.progID}{prog.instrument}_allocated")
            night_history_cols.append(f"{prog.progID}{prog.instrument}_used")
        self.night_history = pd.DataFrame(columns=night_history_cols)

        plans = []
        # Looping over the nights
        for night in self.nights:
            print(f"\nRunning night: {night.night_date}")
            # For each program determine the observable targets
            self.observable_targets = self.determine_observability(night)

            # Build the list of observations
            self.observations = self.build_observations(self.observable_targets, night)

            # Build the plan
            plan = self.build_plan(self.observations)

            plans.append(plan)

            print("Updating tracking tables and saving plan...")
            # Update the tracking tables
            self.update_tracking_tables(plan)

            # Update timeshare between programs
            # Count the total time used by each program from the observation history

            # Save the plan (plots, tables, etc.)
            plan.print_plan(save=True, path=f"{self.save_folder}/plan_{night.night_date}.txt")
            plan.plot(
                display=False, save=True, path=f"{self.save_folder}/plot_{night.night_date}.png"
            )

        # Save the tracking tables to csv
        self.observation_history.to_csv(f"{self.save_folder}/observation_history.csv", index=False)
        self.night_history.to_csv(f"{self.save_folder}/night_history.csv", index=False)

        return plans, self.observation_history, self.night_history
