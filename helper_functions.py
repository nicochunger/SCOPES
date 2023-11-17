import pickle
from datetime import date, datetime, timedelta

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from tqdm.auto import tqdm

import merits
import scheduler
from class_definitions import Merit, Night, Observation, Plan, Program, Target


def load_program(file, prog, merit_list, pct_keep=0.9):
    # Load targets from 714 vthat are observable on the 20th of October 2023
    cat = pd.read_csv(file, sep="\t")
    cat_mod = (
        cat.assign(priority=lambda df: df["priority"].replace({"None": np.nan}).astype(float))
        .pipe(lambda df: df[df["observable during the night"]])
        .assign(score=lambda df: df["priority"] / df["cadence percentage overdue [%]"])
        .pipe(lambda df: df[df["score"] >= 0])
        .sort_values(by="score", ascending=True)
        .pipe(lambda df: df.iloc[: int(len(df) * pct_keep)])
    )
    targets = []
    merits_copy = merit_list.copy()
    for _, tar in cat_mod.iterrows():
        tar_coords = tar["coordinates (DACE)"].split(" / ")
        skycoord = SkyCoord(tar_coords[0], tar_coords[1], unit=(u.hourangle, u.deg))
        # last_obs = Time(tar["last_obs"] + 2_400_000, format="jd").jd
        # last_obs = start_datetime - TimeDelta(5*u.day)
        target = Target(
            tar["catalog name"],
            prog,
            coords=skycoord,
            # last_obs=last_obs,
            exposure_time=0.002,
            priority=tar["priority"],
        )
        for merit in merits_copy:
            target.add_merit(merit)

        # Add target to list
        targets.append(target)
    return targets


def build_observations(targets, start_time, night, exp_time, file=None):
    """
    Builds a list of observations for a given list of targets, exposure time, and file name.

    Parameters:
    -----------
    targets : list
        A list of Target objects to observe.
    exp_time : astropy.time.TimeDelta
        The exposure time for each observation.
    file : str
        The file name to save the observations to.

    Returns:
    --------
    obs_prog : list
        A list of Observation objects for the given targets.
    """
    # Tries to load the observations from a file, if it fails it creates them.
    try:
        if file is None:
            raise FileNotFoundError
        with open(file, "rb") as f:
            obs_prog = pickle.load(f)
    except FileNotFoundError:
        obs_prog = [
            Observation(target, start_time=start_time, exposure_time=exp_time, night=night)
            for target in tqdm(targets)
        ]
        # Save the observation
        if file is not None:
            pickle.dump(obs_prog, open(file, "wb"))
    return obs_prog
