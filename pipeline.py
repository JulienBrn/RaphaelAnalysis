import logging, beautifullogger
import sys
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
import inputs, poly_preprocessing, session_computation, results
# rtmt_results, edge_count_results, trial_counts_results

logger = logging.getLogger(__name__)


def mk_pipeline(folder):
    import seaborn as sns
    pipeline = Database("Raphael database")
    pipeline.declare(CoordComputer(coords={"folder"}, dependencies=set(), compute=lambda db, df: df.assign(folder=str(folder))))
    pipeline = Database.join(pipeline, inputs.pipeline)
    pipeline = Database.join(pipeline, poly_preprocessing.pipeline)
    pipeline = Database.join(pipeline, session_computation.pipeline)
    pipeline = Database.join(pipeline, results.pipeline)
    # pipeline = Database.join(pipeline, rtmt_results.pipeline)
    # pipeline = Database.join(pipeline, edge_count_results.pipeline)
    # pipeline = Database.join(pipeline, trial_counts_results.pipeline)
    return pipeline






if __name__ == "__main__":
    beautifullogger.setup(displayLevel=logging.INFO)
    folder = Path("/home/julienb/Documents/Data/Raphael3/")
    p = mk_pipeline(folder).initialize()
    print(p)
    # p.run_action("compute", "trial_data")
    # p.run_action("compute", "all_basicmotor_fig")
    # p.run_action("compute", "all_cuemotor_fig")
    # p.run_action("compute", "all_pawmotor_fig")
    # p.run_action("compute", "all_zscored_basicmotor_fig")
    # p.run_action("compute", "all_zscored_cuemotor_fig")
    # p.run_action("compute", "all_zscored_pawmotor_fig")
    # exit()
    for d in p.db.data.keys():
        if "compute" in p.db.data[d].actions:
            if not "session" in d and not "subject" in d:
                p.run_action("compute", d)
    # p.run_action("compute", "task_graph_metadata")
    # p.run_action("compute", "task_graph")
    # p.run_action("compute", "task_pdf")
    # p.run_action("compute", "event_dataframe")
    # p.run_action("compute", "session_rtmt_data")
    # p.run_action("compute", "session_duration_stats")
    # p.run_action("compute", "session_duration")
    # p.run_action("compute", "session_rtmt_results")
    # p.run_action("compute", "session_stat_results")
    # p.run_action("compute", "session_pvalue_results")
    # p.run_action("compute", "session_rtmt_figure_results")

   
    