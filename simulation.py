from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from uuid import uuid4

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astroplan import AirmassConstraint, is_observable
from astropy.coordinates import SkyCoord
from astropy.time import Time
from tqdm.auto import tqdm

import merits

# from scheduler import generateQ
from class_definitions import Merit, Night, Observation, Plan, Target


class Simulation:
    def __init__(
        self, start_date, end_date, observer, night_within, scheduler_algorithm, keep_top_n=50
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.observer = observer
        self.night_within = night_within
        self.unique_id = uuid4()
        self.scheduler = scheduler_algorithm
        self.keep_top_n = keep_top_n
        self.save_folder = f"simulation_results/sim_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        # Create the directory for saving the results
        Path(f"{self.save_folder}/plans").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_folder}/night_plots").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_folder}/time_share").mkdir(parents=True, exist_ok=True)
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
        cat_keep = np.ones(len(cat), dtype=bool)
        for i, tar in cat.iterrows():
            # If the target doesn't have a priority assigned, it is ignored and not observed
            if np.isnan(tar["priority"]):
                # Update keep array to keep track of which targets are ignored
                cat_keep[i] = False
                continue

            # Start building the Target object
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
                # Check that all four columns are present
                if not np.isnan(tar[["period", "epoch", "phase1"]].values.astype(float)).any():
                    # Remove the culmination merit
                    merit_list.pop(3)
                    if tar["phase1"] > 1 or tar["phase1"] < 0:
                        raise ValueError("phase1 must be between 0 and 1")
                    # Add the PhaseSpecific merit
                    phases = [tar["phase1"]]
                    # Add phase2 if it is present
                    if not np.isnan(tar["phase2"]):
                        if tar["phase2"] > 0 or tar["phase2"] < 1:
                            phases.append(tar["phase2"])
                        else:
                            raise ValueError("phase2 must be between 0 and 1")

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
        self.programs[prog] = {"targets": targets, "file": file, "df": cat, "keep": cat_keep}

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
            tar_coords = [target.coords for target in self.programs[prog]["targets"]]
            # Determine the observability of the targets
            keep = is_observable(constraints, night.observer, tar_coords, time_range=time_range)

            # Check if targets are due to be observed (cadence constraints)
            cat = self.programs[prog]["df"]
            # Keep the ones that are instructed to be observed
            cat = cat[self.programs[prog]["keep"]]
            if "cadence" in cat.columns:
                # Get last observation of all targets
                last_observation_dates = cat["catalog_name"].apply(
                    lambda x: self.observation_history.loc[
                        self.observation_history["target"] == x
                    ]["obs_time"].max()
                    if x in self.observation_history["target"].unique()
                    else 0
                )

                # Set default cadence to 7 days if not specified
                cat.loc[:, "cadence"] = cat["cadence"].fillna(7)

                # Calculate the percentage of time overdue
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
                top_n = filtered_sorted.head(100).sample(
                    min([self.keep_top_n, len(filtered_sorted)])
                )
                # top_n = filtered_sorted.head(self.keep_top_n)

                # Create a boolean series for marking top 50 targets
                top_n_marks = cat_mod.index.isin(top_n.index)

                # Update keep array
                keep = top_n_marks

            # Get observable targets
            obs_targets = list(np.array(self.programs[prog]["targets"])[keep])
            targets += obs_targets
        return targets

    def build_observations(self, targets: List[Target], night: Night) -> List[Observation]:
        """
        Builds a list of observations for a given list of targets, exposure time, and file name.
        """
        # print("Building observations...")
        observations = [
            Observation(target, night.obs_within_limits[0], target.exposure_time, night)
            for target in targets
        ]
        return observations

    def build_plan(self, observations: List[Observation]) -> Plan:
        """
        Builds a plan from a list of observations.
        """
        # print("Building plan...")
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
        if self.observation_history.empty:
            # After the first night, the observation history will still be empty
            self.observation_history = new_observation_history.reset_index(drop=True)
        else:
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

        # Update the time share of each program
        for i, prog in enumerate(self.programs):
            new_night_history[f"{self.prog_names[i]}_allocated"] = [prog.time_share_allocated]
            try:
                prog_time_used = relative_used_time[f"{self.prog_names[i]}"]
            except KeyError:
                # In case this program was not observed yet
                prog_time_used = 0

            new_night_history[f"{self.prog_names[i]}_used"] = [prog_time_used]
            # Update the program's time share
            prog.update_time_share(prog_time_used)

        if self.night_history.empty:
            # After the first night, the night history will still be empty
            self.night_history = new_night_history.reset_index(drop=True)
        else:
            self.night_history = pd.concat([self.night_history, new_night_history]).reset_index(
                drop=True
            )

    def plot_time_share(self, display=False):
        """
        Plots the time relative used time of each program compared to its allocated time
        """

        # Get the data for the first night
        night_data = self.night_history.iloc[-1]

        # Extract the program names, allocated time, and used time
        # prog_names = [f"{prog.progID}{prog.instrument}" for prog in self.programs]
        allocated_time = [night_data[f"{prog}_allocated"] for prog in self.prog_names]
        used_time = [night_data[f"{prog}_used"] for prog in self.prog_names]
        x = np.arange(1, len(self.prog_names) + 1)
        width = 0.35

        # Calculate the difference between used and allocated times
        time_difference = [used - allocated for used, allocated in zip(used_time, allocated_time)]

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # Adjust the figure size as needed
        # First subplot: Allocated vs Used Time
        ax1.bar(x - (width / 2), allocated_time, width, label="Allocated Time")
        ax1.bar(x + (width / 2), used_time, width, label="Used Time")
        # ax1.set_xlabel("Programs")
        ax1.set_ylabel("Time")
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.prog_names)
        ax1.set_xlim(0.5, len(self.prog_names) + 0.5)
        ax1.legend()
        ax1.set_title(f"{night_data['night']}\nAllocated vs Used Time")

        # Second subplot: Difference between Used and Allocated Time
        ax2.bar(x, time_difference, width * 0.75, color="red", label="Time Difference")
        ax2.axhline(y=0, color="grey", linestyle="--")
        ax2.set_xlabel("Programs")
        ax2.set_ylabel("Time Difference")
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.prog_names)
        ax2.set_xlim(0.5, len(self.prog_names) + 0.5)
        ax2.set_ylim(-0.5, 0.5)
        ax2.legend()
        ax2.set_title("Difference between Used and Allocated Time")

        # Save the figure
        plt.tight_layout()
        fig.savefig(f"{self.save_folder}/time_share/{night_data['night']}_time_share.png", dpi=200)
        if display:
            plt.show()
        else:
            plt.close()

    def post_simulation_diagnostic(self, display=False):
        """
        Plots the time share of each program over the course of the simulation.
        """

        # Plots the time share of each program over the course of the simulation.
        plt.figure(figsize=(8, 5))
        for i, prog in enumerate(self.programs):
            plt.plot(
                self.night_history["night"],
                self.night_history[f"{self.prog_names[i]}_used"]
                - self.night_history[f"{self.prog_names[i]}_allocated"],
                label=self.prog_names[i],
                color=prog.plot_color,
            )
        plt.xlabel("Night")
        plt.xticks(self.night_history["night"], rotation=45, ha="right")
        plt.ylabel("Difference between used and allocated time")
        plt.title("Relative Used Time of Each Program")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.save_folder}/time_share/relative_used_time.png", dpi=200)
        if display:
            plt.show()
        else:
            plt.close()

        # TODO Calculate how well it followed the cadence constraints

    def run(self):
        """
        Runs the simulation.
        """
        # Build the nights
        self.build_nights()

        if self.programs == {}:
            print("No programs have been added to the simulation.")
            return

        self.prog_names = [f"{prog.progID}{prog.instrument}" for prog in self.programs]

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
        for progname in self.prog_names:
            night_history_cols.append(f"{progname}_allocated")
            night_history_cols.append(f"{progname}_used")
        self.night_history = pd.DataFrame(columns=night_history_cols)

        self.plans = []
        # Looping over the nights
        for night in tqdm(self.nights):
            print(f"\nRunning night: {night.night_date}")
            # For each program determine the observable targets
            self.observable_targets = self.determine_observability(night)

            # Build the list of observations
            self.observations = self.build_observations(self.observable_targets, night)

            # Build the plan
            plan = self.build_plan(self.observations)

            self.plans.append(plan)

            # print("Updating tracking tables and saving plan...")
            # Update the tracking tables
            self.update_tracking_tables(plan)

            # Save the plan (plots, tables, etc.)
            plan.print_plan(
                save=True, path=f"{self.save_folder}/plans/plan_{night.night_date}.txt"
            )
            plan.plot(
                display=False,
                save=True,
                path=f"{self.save_folder}/night_plots/plot_{night.night_date}.png",
            )
            self.plot_time_share()

        # Run the post simulation diagnostic
        self.post_simulation_diagnostic()

        # Save the tracking tables to csv
        self.observation_history.to_csv(f"{self.save_folder}/observation_history.csv", index=False)
        # Create a tempirary df with the timedeltas formatted as strings in the format HH:MM:SS
        temp_nh = self.night_history.copy()
        temp_nh["observation_time"] = temp_nh["observation_time"].apply(
            lambda x: (datetime(2000, 1, 1) + x).strftime("%H:%M:%S")
        )
        temp_nh["overhead_time"] = temp_nh["overhead_time"].apply(
            lambda x: (datetime(2000, 1, 1) + x).strftime("%H:%M:%S")
        )
        temp_nh.to_csv(f"{self.save_folder}/night_history.csv", index=False)

        return self.plans, self.observation_history, self.night_history
