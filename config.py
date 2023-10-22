import pytz

# TIMEZONES & SCHEDULES
time_zones = {'gva' : pytz.timezone('Europe/Zurich'),
             'utc' : pytz.timezone('UTC'),
             'chile' : pytz.timezone('America/Santiago') }

#Define how many minutes earlier relative to sunset / twilight central tasks should be started
time_to_sunset = {'open dome': 35,
                  'start telescope': 20,
                  'flats': 0}
time_to_twilight = {'focus': 10,
                    'start calibrations': 180}

#time window for plotting
plot_minutes_before_telstart=20
plot_minutes_after_morning_twilight=60

# LOGS

#telescope logs info
telescope_shutters = {'folder': '/home/pedro/EulerOperations/python-scripts/euler-scripts/telescope_logs/shutters/', 'column names':
    ['tunix', 'top_open', 'bottom_open', 'flap1_open', 'flap3_open', 'flap5_open', 'flap7_open', 'flap9_open',
     'flap11_open', 'top_closed', 'bottom_closed', 'flap1_closed', 'flap3_closed', 'flap5_closed', 'flap7_closed',
     'flap9_closed', 'flap11_closed', 'date']}

telescope_m3 = {'folder': '/home/pedro/EulerOperations/python-scripts/euler-scripts/telescope_logs/m3/',
                'column names': ['tunix', 'focus', 'ilr_linActPos', 'ilr_rotActPos', 'ilr_ablActPos',
               'ib_rotLimitSwitchPlus', 'ib_rotLimitSwitchMinus', 'ib_linLimitSwitchPlus', 'ib_linLimitSwitchMinus',
               'iud_linEncodeurValue', 'iud_rotEncodeurValue', 'id_linPositionActualInternalValue',
               'id_rotPositionActualInternalValue',	'id_ablPositionActualInternalValue']}

#edp logs info
edp_exposures_list = {'folder_COR': '/home/pedro/EulerOperations/python-scripts/euler-scripts/datatwin/EDPlogs/',
                      'folder_CAM': '/home/pedro/EulerOperations/python-scripts/euler-scripts/datatwin/CAM_EDPlogs/',
                      'folder_KAL': '/home/pedro/EulerOperations/python-scripts/euler-scripts/datatwin/KAL_EDPlogs/',
                      'column names':
    ['nseq', '_fz', '_schedule', '_start_ts', 'ah', 'airmass', 'alphacat', 'ampname', 'azim', 'binx', 'biny', 'ccdgain',
     'ccdros', 'centrage', 'cmd', 'code', 'comments', 'coord', 'corbaorb', 'corbarot', 'dalpha', 'ddelta', 'defoc',
     'defoctpl', 'deltacat', 'dif', 'elev', 'equicat', 'factobth', 'filtre', 'flatlist', 'flux', 'fz', 'gtexp',
     'guidage',	'il_moon', 'kalfilter', 'lastobs', 'led', 'mualph', 'mudelt', 'mv',	'nbmes', 'nocat', 'nocoda',	'nocodb',
     'nodup', 'noprog', 'nposes', 'observation', 'phi',	'pisfiltre', 'readtype', 'refnocod', 'remarques', 'repeatOb',
     'repeatTpl', 'se_moon', 'sequence', 'sn',' st1', 'st2', 'st3',	'start', 'stopon', 'texp', 'type', 'typsp',	'ut1',
     'ut2', 'ut3', 'vitesse', 'vpicmoon', 'xdctr', 'xguid',	'ydctr', 'yguid', '_CTR_VL', '_ACQ_VL', '_ARC_VL',
     '_ALL_STATUS',	'_UNSET', '_STARTED', '_STARTED_AT_FMT']}

# DATA LOCATION & KWs
data_location = {'CORALIE raw': '/home/pedro/EulerOperations/python-scripts/euler-scripts/datatwin/CORALIE/raw/',
                 'CORALIE reduced': '/home/pedro/EulerOperations/python-scripts/euler-scripts/datatwin/CORALIE/reduced/',
                 'ECAM raw': '/home/pedro/EulerOperations/python-scripts/euler-scripts/datatwin/ECAM/raw/',
                 'KalAO raw': '/home/pedro/EulerOperations/python-scripts/euler-scripts/datatwin/KalAO/raw/'}

#KWs to read
KWs = {'CORALIE raw': ['OBSERVER', 'EMAIL', 'HIERARCH ESO OBS NOPROG', 'OBJECT', 'HIERARCH ESO TPL TYPE',
                       'HIERARCH ESO OBS TEXP', 'HIERARCH ESO OBS TARG AIRMASS', 'HIERARCH ESO OBS AMBI DIMM SEEING',
                       'HIERARCH ESO OBS TARG MV', 'HIERARCH ESO CORA PM COUNT'],
       'CORALIE guiding': ['HIERARCH GUIDING TEXP', 'HIERARCH GUIDING NEXP INT', 'HIERARCH GUIDING NEXP',
                           'HIERARCH OBJECT XCDGMOY', 'HIERARCH OBJECT YCDGMOY', 'HIERARCH FIBER XREFCUR',
                           'HIERARCH FIBER XREF', 'HIERARCH FIBER YREFCUR', 'HIERARCH FIBER YREF',
                           'HIERARCH FIBER QC', 'HIERARCH OBJECT QC'],
       'CORALIE reduced': [ 'HIERARCH ESO DRS CAL LOC FWHM203', 'HIERARCH ESO DRS CAL LOC FWHM308',
                            'HIERARCH ESO DRS CAL LOC FWHM406', 'HIERARCH ESO DRS CAL LOC FLX MAX',
                            'HIERARCH ESO DRS CAL TH ERROR', 'HIERARCH ESO DRS CAL TH FLUX RATIO',
                            'HIERARCH ESO DRS CAL QC', 'HIERARCH ESO DRS SPE EXT SN55', 'HIERARCH ESO DRS SPE EXT SN56',
                            'HIERARCH ESO DRS SPE EXT SN57', 'HIERARCH ESO DRS BJD', 'HIERARCH ESO DRS CCF RVC',
                            'HIERARCH ESO DRS CCF RV', 'HIERARCH ESO DRS CCF FWHM', 'HIERARCH ESO DRS CCF NOISE',
                            'HIERARCH ESO DRS DRIFT NOISE', 'HIERARCH ESO DRS DRIFT QC'],
       'ECAM raw': ['OBSERVER', 'EMAIL', 'HIERARCH OGE OBS NOPROG', 'OBJECT', 'EXPTIME',
                    'HIERARCH OGE TEL TARG AIRM MID', 'HIERARCH OGE AMBI DIMM SEEING', 'HIERARCH OGE OBS TARG MV',
                    'MJD-OBS', 'FILTER', 'HIERARCH OGE OBS DEFOC', 'HIERARCH OGE DET OUT RNAME'],
       'KalAO raw': ['OBSERVER', 'EMAIL', 'HIERARCH KAL PROG_ID', 'OBJECT', 'EXPTIME', 'HIERARCH ESO OBS TARG AIRMASS']}

# OVERHEADS
overheads = {'TEL': {'telescope slew': 30, 'instrument change': 170}, 'COR': {'readout': {'fast': 20, 'other': 67}},
             'CAM': {'ampname': {'ALL': 18, 'other': 30}}, 'KAL':{'all': 10}}

# default time for monitor (days)
monitor_timespan = 180

# GUIDING
#
#dates of guiding commissioning in Jan 2023
dates_commissioning = ['2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15',
                       '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19']
dates_bugs = ['2023-04-23'] # guiding not being configured for glslogin2 bug
exclude_dates_guiding = dates_commissioning + dates_bugs

# SCHEDULING

scheduling_instructions = {'folder': '/home/pedro/EulerOperations/python-scripts/euler-pi-instructions/',
                 'configfilename': 'schedulingConfig'}

scheduling_defaults = {'airmass limit': 1.5, 'telescope elevation limits': [20.0, 87.0]}

########################################################################################################################

# PERIOD & PROGRAMS INFO

# Period info for P109
#period = {'number': 109, 'start date': '2022-04-04', 'end date': '2022-10-03', 'nr_nights':160, 'night duration [h]': 10,
#          'run duration': '/home/pedro/EulerOperations/python-scripts/euler-scripts/periodinfo/P109_runduration.csv',
#          'ECAM summary Angelica': '/home/pedro/EulerOperations/python-scripts/euler-scripts/periodinfo/ECAM_time.txt'}

# coming from https://docs.google.com/spreadsheets/d/1A9YyeTegYavVQ-VTqV71RGaKzNrvz-DGwH1hkuFXTIg/edit#gid=1625842821
# for 600: time per week 2520*3.5 / 3600 = 2.45h
# per winter semester 2.45 / 10h * 26 weeks = 6.5
#programs = ['500@COR', '600@COR', '703@COR', '708@COR', '714@COR', '729@COR', '730@COR', '735@COR', '736@COR', '738@COR',
#            '756@COR', '410@CAM', '447@CAM', '448@CAM', '449@CAM', '450@CAM', '500@CAM', '000@KAL']

#programs_mangled_up = {'500@COR': {'PI':'N.Grieves', 'total time [n]': 31},
#            '600@COR': {'PI':'D.Segransan', 'total time [n]': 6.5},
#            '703@COR': {'PI':'D.Segransan', 'total time [n]': 21},
#            '708@COR': {'PI':'N.Unger', 'total time [n]': 8},
#            '714@COR': {'PI':'D.Segransan', 'total time [n]': 6},
#            '737@COR': {'PI': 'who knows?', 'total time [n]': 0},
#            '729+730+735+736+738@COR': {'PI':'who knows?', 'total time [n]':6},
#            '756@COR': {'PI':'R.Anderson', 'total time [n]': 13},
#            '410@CAM': {'PI':'COSMOGRAIL', 'total time [n]': 24},
#            '447@CAM': {'PI':'who knows?', 'total time [n]': 5},
#            '448+449@CAM': {'PI':'who knows?', 'total time [n]': 12},
#            '450@CAM': {'PI':'who knows?', 'total time [n]': 9},
#            '500@CAM': {'PI':'who knows?', 'total time [n]': 5},
#            '000@KAL': {'PI':'J. Hagelberg', 'total time [n]': 20}}

# Period info for P110
# https://docs.google.com/spreadsheets/d/1A9YyeTegYavVQ-VTqV71RGaKzNrvz-DGwH1hkuFXTIg/edit#gid=1002774307
#period = {'number': 110, 'start date': '2022-10-04', 'end date': '2023-04-03', 'nr_nights':142.5,
#          'run duration': '/home/pedro/EulerOperations/python-scripts/euler-scripts/periodinfo/P110_runduration.txt'}

# Period info for P111
# https://docs.google.com/spreadsheets/d/1A9YyeTegYavVQ-VTqV71RGaKzNrvz-DGwH1hkuFXTIg/edit#gid=1448332837
#period = {'number': 111, 'start date': '2023-04-04', 'end date': '2023-10-02', 'nr_nights':164,
#          'run duration': '/home/pedro/EulerOperations/python-scripts/euler-scripts/periodinfo/P111_runduration.txt'}

# Period info for P112
# https://docs.google.com/spreadsheets/d/1A9YyeTegYavVQ-VTqV71RGaKzNrvz-DGwH1hkuFXTIg/edit#gid=1448332837
period = {'number': 112, 'start date': '2023-10-03', 'end date': '2024-04-01', 'nr_nights':164,
          'run duration': '/home/pedro/EulerOperations/python-scripts/euler-scripts/periodinfo/P112_runduration.txt'}

programs = ['500@COR', '600@COR', '703@COR', '708@COR', '714@COR', '730@COR', '735@COR', '736@COR', '738@COR',
            '756@COR', '410@CAM', '449@CAM', '450@CAM', '500@CAM', '199@KAL']

# this represents how the programs should be considered for the plot only
programs_mangled_up = {'500@COR': {'PI':'Nolan/Angelica/Solene/Matthews ', 'total time [n]': 30},
            '600@COR': {'PI':'D.Segransan', 'total time [n]': 5},
            '703@COR': {'PI':'D.Segransan', 'total time [n]': 21},
            '708@COR': {'PI':'N.Unger', 'total time [n]': 9},
            '714@COR': {'PI':'S.Udry+P.Figueira', 'total time [n]': 6.5},
            '730+735+736+738@COR': {'PI':'L.Parc', 'total time [n]': 13},
            '756@COR': {'PI':'R.Anderson', 'total time [n]': 13},
            '410@CAM': {'PI':'COSMOGRAIL', 'total time [n]': 23.5},
            '449@CAM': {'PI':'A.Psaridi', 'total time [n]': 12},
            '450@CAM': {'PI':'A.Psaridi', 'total time [n]': 6},
            '500@CAM': {'PI':' Angelica/Solene', 'total time [n]': 7}}
            #'199@KAL': {'PI':'J. Hagelberg', 'total time [n]': 18}}

########################################################################################################################

# PROCESSED OUTPUT
reports_folder = {'edp lists': '/home/pedro/EulerOperations/python-scripts/euler-scripts/processed/data/EDP/',
                  'figures': '/home/pedro/EulerOperations/python-scripts/euler-scripts/processed/Figures/',
                  'night logs': '/home/pedro/EulerOperations/python-scripts/euler-scripts/processed/data/nightlogs/',
                  'time logs': '/home/pedro/EulerOperations/python-scripts/euler-scripts/processed/data/timelogs/',
                  'program reports': '/home/pedro/EulerOperations/python-scripts/euler-scripts/processed/data/programreports/',
                  'semester summary': '/home/pedro/EulerOperations/python-scripts/euler-scripts/processed/data/semestersummary/',
                  'programs summary': '/home/pedro/EulerOperations/python-scripts/euler-scripts/processed/data/programssummary/',
                  'scheduling': '/home/pedro/EulerOperations/python-scripts/euler-scripts/processed/data/scheduling/',
                  'CORALIE monitoring': '/home/pedro/EulerOperations/python-scripts/euler-scripts/processed/data/monitoring/'}

summary_names = {'night logs': 'NightLogsSummary_P{}'.format(period['number']),
                 'time dist': 'TimeDistSummary_P{}'.format(period['number']),
                 'input files': 'InputFilesSummary_P{}'.format(period['number']),
                 'time logs programs': 'TimeLogsProg_P{}'.format(period['number'])}
