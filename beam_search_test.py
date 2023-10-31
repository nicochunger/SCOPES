import pickle
from datetime import timedelta

import astropy.units as u

# import pandas as pd
from astropy.time import Time, TimeDelta

import merits
import scheduler
from class_definitions import Merit, Night, Program
from helper_functions import build_observations, load_program

# Define start start and night
start_datetime = Time("2023-10-20 03:00:00")
night = Night(start_datetime.datetime.date() - timedelta(days=1), observations_within="nautical")

# Check if "obs_beamsearch.pkl" exists and load it
try:
    observations_all = pickle.load(open("obs_beamsearch.pkl", "rb"))
except FileNotFoundError:
    # If not, create it
    # Define merits
    airmass_merit = Merit("Airmass", merits.airmass, merit_type="veto", parameters={"max": 1.8})
    altitude_merit = Merit("Altitude", merits.altitude, merit_type="veto")
    at_night_merit = Merit("AtNight", merits.at_night, merit_type="veto")
    culmination_merit = Merit("Culmination", merits.culmination, merit_type="efficiency")
    egress_merit = Merit("Egress", merits.egress, merit_type="efficiency")
    merit_list = [airmass_merit, altitude_merit, at_night_merit, culmination_merit, egress_merit]

    # Create target and observation lists for each program
    progs = {600: 20, 708: 15, 714: 8}
    targets_all = []
    observations_all = []
    for prog_num, exp_time in progs.items():
        progobj = Program(prog_num, 5, instrument="CORALIE")
        targets_prog = load_program(
            f"programs/Prog{prog_num}COR_2023-10-19_obsinfo.csv",
            progobj,
            merit_list,
            pct_keep=0.75,
        )
        targets_all += targets_prog
        observations_all += build_observations(
            targets_prog, start_datetime, night, TimeDelta(exp_time * u.min)
        )

    # Save observations to file
    pickle.dump(observations_all, open("obs_beamsearch.pkl", "wb"))


# Scheduler
plan_start = Time(night.nautical_evening, format="jd")
beamsearch = scheduler.BeamSearchPlanner(plan_start.jd)

bs_plan = beamsearch.dp_beam_search(
    observations_all,
    # max_plan_length=15
)


bs_plan.plot(save=True)
