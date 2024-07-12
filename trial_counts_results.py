import logging, beautifullogger
from typing import List
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob, precompute
import pandas as pd, numpy as np, xarray as xr, string
from pathlib import Path
import poly_graph
import json, networkx as nx
logger = logging.getLogger(__name__)


p = Database("trialcount_results")
groups =  ["stim_type","hemi", "condition","handedness", "forced", "task"]

def compute_trial_counts(db, out_location, selection):
    dfs = db.run_action("load", "session_trial_types", selection)
    all = pd.concat([row["load"] for _, row in dfs.iterrows()], ignore_index=True)
    return all[["sucess"] + [col for col in all.columns if not "trial" in col and not "sucess" in col]].value_counts(dropna=False).to_frame("count").reset_index().sort_values("sucess")

def compute_trialpair_counts(db, selection):
    dfs = db.run_action("load", "session_trialpair_types", selection)
    all = pd.concat([row["load"].assign(session=row["session"], **{k: row[k] for k in groups}) for _, row in dfs.iterrows()], ignore_index=True)
    all[["sucess", "sucess_next"] + [col for col in all.columns if not "trial" in col and not "sucess" in col]].value_counts(dropna=False).to_frame("count").reset_index().sort_values(["sucess", "sucess_next"])
    return all



def make_trialcount_step(name, params, location: str):
    keys = [p[1] for p in string.Formatter().parse(location)]
    counts =  Data(name, dependencies=set(params), get_location=lambda d: Path(location.format_map({k:v for k,v in d.items() if k in keys})), actions=dict(
        compute=cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=True)(compute_trial_counts),
        load = precompute(name)(lambda db, out_location, selection:pd.read_csv(out_location, sep="\t", index_col=None))))
    return counts


p.declare(make_trialcount_step("session_trialtype_counts", ["folder", "session"] + groups, 
                               location="{folder}/PolyAnalysis/Sessions/{session}/Results/trial_counts.tsv"))
p.declare(make_trialcount_step("subject_trialtype_counts", ["folder", "subject"] + groups, 
                               location="{folder}/PolyAnalysis/Subjects/{subject}/Counts/trial_counts_task_{task}_hemi{hemi}.tsv"))
p.declare(make_trialcount_step("all_trialtype_counts", ["folder"] + groups, 
                               location="{folder}/PolyAnalysis/All/Counts/trial_counts_task_{task}_hemi{hemi}.tsv"))

pipeline = p