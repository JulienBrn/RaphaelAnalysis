import logging, beautifullogger
from typing import List
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob, precompute
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
import poly_graph
import json, networkx as nx
logger = logging.getLogger(__name__)


p = Database("session_computation")


def link_to_task(data: pd.DataFrame):
    data["event_priority"] = (data["family"] == 10).astype(int)
    data = data.sort_values(["t", "event_priority"])
    data["trial"] = (data["event_name"]=="line_change_to_2").cumsum()-1
    data["line_change_counter"] = (data["family"] == 10).cumsum()-1
    data["node"] = np.where(data["family"] == 10, data["_T"], np.nan)
    data["node"] = data["node"].fillna(method="ffill").astype(int)
    data["next_node"] = data["node"].where(data["family"] == 10).fillna(method="bfill").shift(-1).fillna(-1).astype(int)
    data["t_next"] = data["t"].shift(-1)
    is_fake_trial = data.groupby("trial").apply(lambda d: 
                                                len(d.loc[ (d["next_node"]==2) & ((d["node"]==3) | (d["node"]==4) | (d["node"]==32) | (d["node"]==47))].index) > 0
                                                ).to_frame("is_fake_trial").reset_index()
    is_fake_trial["fake_trial_num"] = is_fake_trial["is_fake_trial"].shift(1).cumsum()
    data = data.merge(is_fake_trial, how="left", on="trial")
    data["all_trial_num"] = data["trial"]
    data["trial"] =data["trial"] - data["fake_trial_num"]
    buggy = data.groupby("trial").apply(lambda d: d["is_fake_trial"].all())
    if buggy.any():
        print("Problem")
        print(buggy.loc[buggy])
        r = data[["trial", "all_trial_num", "is_fake_trial", "fake_trial_num", "event_name"]]
        print(r)
        r.to_csv("to_debug.tsv", sep="\t", index=False)
        exit()
    data = data.groupby("trial").apply(lambda d: d if len(d.loc[d["family"]==11].index) == 0 else pd.DataFrame()).reset_index(drop=True)
    return data.drop(columns=["fake_trial_num" , "event_priority"])


@p.register
@Data.from_class()
class EventDataframe:
    name = "session_event_task"

    @staticmethod
    def location(folder, session):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Processing"/"session_event_task.tsv"
    
    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location, selection):
        events: pd.DataFrame = db.run_action("load", "event_dataframe", selection, single=True)
        return link_to_task(events)
    
    @precompute("session_event_task")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)


@p.register
@Data.from_class()
class EventDataframe:
    name = "trial_data"

    @staticmethod
    def location(folder, session):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Processing"/"trial_data.tsv"

    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location, selection):
        graph: nx.DiGraph = db.run_action("load", "task_graph", selection, single=True)
        data=db.run_action("load", "session_event_task", selection, single=True)
        node_names = pd.DataFrame([dict(node=node, names=graph.nodes(data=True)[node]["names"]) for node in graph.nodes if "names" in graph.nodes(data=True)[node]])
        data = data.merge(node_names, how="inner", on=["node"])
        data = data[data["trial"] >=0]
        def get_trial_info(d: pd.DataFrame):
            
            import re
            all_names = set([n for _, l in d["names"].items() for n in l])
            error_types = {
                "error": r"erreur",
                "Anticip": r"Anticip.*",
                "RTCutoff": r"CutOfRT.*",
                "BadRT": r"RT[34].*",
                "BadLP": r"MT[34].*",
                "MTCutOff": r"CutOfMT.*",
            }
            trial_cat = {k: np.any([not re.fullmatch(v, n) is None for n in all_names]) for k,v in error_types.items()}
            res = {}
            if trial_cat["error"]:
                err = [k for k, v in trial_cat.items() if v and not k == "error"]
                if len(err) != 1:
                    raise Exception(f"Problem {err}\n{all_names}")
                res["trial_type"] = err[0]
            res["cue"] = "Right" if "CueRight" in all_names else "Left" if "CueLeft" in all_names else "Unknown"
            res["stim"] = "Beta" if "StimBeta" in all_names else "Conti" if "StimConti" in all_names else "NoStimYet" if "trial_type" in res and res["trial_type"] not in ["BadLP", "MTCutOff"] else "NoStim"
            if not trial_cat["error"]:
                m_start = d.loc[d["names"].apply(lambda l: np.any([n=="MT" for n in l]))]["t"].min()
                m_end = d.loc[d["names"].apply(lambda l: np.any(["MT" in n and len(n)>2 for n in l]))]["t"].min()
                if pd.isna(m_start) or pd.isna(m_end):
                    print(f"Expected start and end but not found\n{d.sort_values('t').to_string()}")
                    print(data)
                    input()
                if d.loc[(d["t"] > m_start) & (d["t"] < m_end)]["event_name"].isin(["right_pad_lift", "left_pad_lift", "right_pad_press", "left_pad_press"]).any():
                    res["trial_type"] = "BadMT"
                else:
                    
                    res["mt"] = m_end-m_start
                    res["trial_type"] = "good"
            return pd.Series(res).rename_axis("info_type")
        trials = data.groupby("trial").apply(get_trial_info).unstack("info_type").reset_index()
        trials["paw"] = trials["cue"] if selection["handedness"] == "Ambi" else selection["handedness"]
        zscore_grp = trials.loc[trials["stim"]=="NoStim"].groupby(["trial_type", "cue"])
        zscore_params = pd.DataFrame()
        zscore_params["mean"] = zscore_grp["mt"].mean()
        zscore_params["std"] = zscore_grp["mt"].std()
        zscore_params=zscore_params.reset_index()
        all = pd.merge(trials, zscore_params, how="left", on=["trial_type", "cue"])
        trials["mt_zscored"] = (trials["mt"] - all["mean"])/all["std"]
        return trials

    @precompute("trial_data")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)


pipeline=p