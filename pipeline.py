import logging, beautifullogger
import sys
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
import coordinates, polyanalysis

logger = logging.getLogger(__name__)


def mk_pipeline(folder):
    import seaborn as sns
    pipeline = Database("Raphael database")
    pipeline.declare(CoordComputer(coords={"folder"}, dependencies=set(), compute=lambda db, df: df.assign(folder=str(folder))))
    pipeline = Database.join(pipeline, coordinates.pipeline)
    pipeline = Database.join(pipeline, polyanalysis.pipeline)
    return pipeline






if __name__ == "__main__":
    beautifullogger.setup(displayLevel=logging.INFO)
    folder = Path("/home/julienb/Documents/Data/Raphael/")
    p = mk_pipeline(folder).initialize()
    print(p)
    # print(p.get_coords("subject"))
    p.run_action("make_tsv", "event_data")
    p.run_action("compute", "task_graph")
    p.run_action("get_edge_dataframe", "session_graph")
    p.run_action("compute", "session_rtmt")
    p.run_action("save_dataframe", "session_rtmt")
    p.run_action("compute", "subject_rtmt")
    p.run_action("compute", "rtmt")
    