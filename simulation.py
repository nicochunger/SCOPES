from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import scheduler
from class_definitions import Night, Program, Target
from helper_functions import build_observations, load_program


class Simulation:
    def __init__(
        self,
        start_date,
        end_date,
    ):
        self.start_date = start_date
        self.end_date = end_date

        # Build the nights
        self.build_nights()

    def build_nights(self):
        """
        Builds a list of nights from the start date to the end date.
        """
        self.nights = []
        for i in range((self.end_date - self.start_date).days):
            night = Night(self.start_date + timedelta(days=i))
            self.nights.append(night)
        return self.nights

    def build_programs(self, file, merit_list, pct_keep=0.9):
        """
        Builds a list of programs from a file.
        """
        self.programs = []
        for i in range(len(self.nights)):
            prog = Program(i)
            prog.targets = load_program(file, prog, merit_list, pct_keep)
            self.programs.append(prog)
        return self.programs

    def build_observations(self, exp_time, file=None):
        """
        Builds a list of observations for a given list of targets, exposure time, and file name.
        """
        self.observations = []
        for prog in self.programs:
            for target in prog.targets:
                self.observations.append(build_observations(target, exp_time, file))
        return self.observations

    def build_plan(self, sched, observations):
        """
        Builds a plan from a list of observations.
        """
        gQ_scheduler = sched(observations[0].night.obs_within_limits[0])

        # Create the plan
        greedy_plan = gQ_scheduler.generateQ(observations, max_plan_length=None, K=1)
        return greedy_plan
