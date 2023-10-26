import pickle
from datetime import timedelta

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from tqdm import tqdm

import merits
import scheduler
from class_definitions import Merit, Night, Observation, Program, Target

# Check if "observations.pkl" exists and load it
try:
    observations = pickle.load(open("observations.pkl", "rb"))
except FileNotFoundError:
    # If not, create it
    prog708 = Program(708, 7, instrument="CORALIE")
    prog714 = Program(714, 2, instrument="CORALIE")

    start_datetime = Time("2023-10-20 01:00:00")

    # Time(datetime.combine(date(2023, 10, 20), datetime.min.time()))
    # start_datetime.datetime.date()

    night = Night(start_datetime.datetime.date() - timedelta(days=1))


    # Define merits
    cadence_merit = Merit(
        "Cadence",
        merits.cadence,
        merit_type="efficiency",
        parameters={"delay": TimeDelta(4 * u.day), "alpha": 0.05},
    )
    airmass_merit = Merit(
        "Airmass", merits.airmass, merit_type="veto", parameters={"max": 1.5}
    )
    altitude_merit = Merit("Altitude", merits.altitude, merit_type="veto")
    at_night_merit = Merit(
        "AtNight", merits.at_night, merit_type="veto", parameters={"which": "nautical"}
    )
    culmination_merit = Merit("Culmination", merits.culmination, merit_type="efficiency")
    egress_merit = Merit("Egress", merits.egress, merit_type="efficiency")

    # Test targets for the night of 2023-10-20
    test_targets = [
        "HD224953A",
        "CD-3931",
        "CD-33501",
        "BD-21397",
        "HD16157",
        "CD-231056",
        "CD-341770",
        "HD32965",
        "HD35650",
        "HD274919",
        "BD-221344",
        "CD-492340A",
        "CD-422626",
        "HD51199B",
        "BD-141750",
        "BD-191855",
        "HD58760",
        "CD-363646",
        "HD66020",
        "CD-81812",
        "HD214100",
        "HD218294",
    ]

    # Create the targets
    targets_708 = []
    for tar_name in tqdm(test_targets):
        last_obs = start_datetime - 5 * u.day
        target = Target(
            tar_name, prog708, coords=SkyCoord.from_name(tar_name), last_obs=last_obs
        )
        target.add_merit(cadence_merit)
        # target.add_merit(airmass_merit)
        target.add_merit(culmination_merit)
        # target.add_merit(egress_merit)
        target.add_merit(altitude_merit)
        target.add_merit(at_night_merit)
        targets_708.append(target)


    # Load targets from 714 vthat are observable on the 20th of October 2023
    cor714 = pd.read_csv("Prog714COR_priorP3_2023-10-20.csv", skiprows=1, sep="\t")

    targets_714 = []
    for i, tar in cor714.iterrows():
        last_obs = start_datetime - 5 * u.day
        tar_coords = tar["coordinates (DACE)"].split(" / ")
        skycoord = SkyCoord(tar_coords[0], tar_coords[1], unit=(u.hourangle, u.deg))
        target = Target(tar["catalog name"], prog714, coords=skycoord, last_obs=last_obs)
        target.add_merit(cadence_merit)
        # target.add_merit(airmass_merit)
        target.add_merit(culmination_merit)
        # target.add_merit(egress_merit)
        target.add_merit(altitude_merit)
        target.add_merit(at_night_merit)
        targets_714.append(target)

    # Merge the two programs
    targets_all = targets_708 + targets_714

    # Create the new observations with the new targets
    exp_time = TimeDelta(15 * u.min)
    observations = [
        Observation(
            target,
            start_time=Time(night.astronomical_evening, format='jd'),
            exposure_time=exp_time,
            night=night,
        )
        for target in tqdm(targets_all)
    ]

    # Save observations to file
    pickle.dump(observations, open("observations.pkl", "wb"))



# Scheduler
beamsearch = scheduler.BeamSearchPlanner()

bs_plan = beamsearch.dp_beam_search(observations, max_plan_length=15, K=4)


bs_plan.plot(save=True)
