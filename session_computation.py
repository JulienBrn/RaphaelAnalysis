import logging, beautifullogger
from typing import List
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob, precompute
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
import poly_graph
import json, networkx as nx
logger = logging.getLogger(__name__)


p = Database("session_computation")

@p.register
@Data.from_class()
class EventDataframe:
    name = "session_rtmt_data"

    @staticmethod
    def location(folder, session):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Processing"/"rtmt_data.tsv"

    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location, selection):
        graph: nx.DiGraph = db.run_action("load", "task_graph", selection, single=True)
        edges =pd.DataFrame([dict(node=edge[0], next_node=edge[1], **graph.edges[edge]["data"]) for edge in graph.edges if "data" in graph.edges[edge]])
        events: pd.DataFrame = db.run_action("load", "event_dataframe", selection, single=True)
        data=events
        data["event_priority"] = (data["family"] == 10).astype(int)
        data = data.sort_values(["t", "event_priority"])
        data["trial"] = (data["event_name"]=="line_change_to_2").cumsum()-1
        data = data.groupby("trial").apply(lambda d: d if len(d.loc[d["family"]==11].index) == 0 else pd.DataFrame()).reset_index(drop=True)
        data["line_change_counter"] = (data["family"] == 10).cumsum()-1
        data["node"] = np.where(data["family"] == 10, data["_T"], np.nan)
        data["node"] = data["node"].fillna(method="ffill")
        data["next_node"] = data["node"].where(data["family"] == 10).fillna(method="bfill").shift(-1)
        data["t_next"] = data["t"].shift(-1)
        data = data.merge(edges, how="inner", on=["node", "next_node"])
        res= pd.DataFrame()
        grp = data.groupby(["line_change_counter","trial", "variable", "stim", "lever"])
        res["start"] = grp["t"].first()
        res["end"] = grp["t_next"].last()
        def get_first_event(d, event_name):
            filtered = d.loc[d["event_name"] == event_name]
            if len(filtered.index) > 0:
                return filtered["t"].iat[0]
            else: return np.nan
 
        res["first_right_lever_press"] = grp.apply(lambda d: get_first_event(d, "right_lever_press"))
        res["first_left_lever_press"] = grp.apply(lambda d: get_first_event(d, "left_lever_press"))
        res["first_right_pad_lift"] = grp.apply(lambda d: get_first_event(d, "right_pad_lift"))
        res["first_left_pad_lift"] = grp.apply(lambda d: get_first_event(d, "left_pad_lift"))
        res["first_right_pad_press"] = grp.apply(lambda d: get_first_event(d, "right_pad_press"))
        res["first_left_pad_press"] = grp.apply(lambda d: get_first_event(d, "left_pad_press"))
        res = res.reset_index(["trial", "variable", "stim", "lever"]).reset_index(drop=True)
        if res.duplicated(["trial", "variable"]).any():
            raise Exception("Problem")
        res["duration"] = np.where(res["variable"]=="rt", res["end"], res[["end", "first_right_lever_press", "first_left_lever_press"]].min(axis=1))  - res["start"]  
        return res
    
    @precompute("session_rtmt_data")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)

@p.register
@Data.from_class()
class EventDataframe:
    name = "session_duration_stats"

    @staticmethod
    def location(folder, session):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Processing"/"duration_stats.tsv"

    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location, selection):
        df: pd.DataFrame = db.run_action("load", "session_rtmt_data", selection, single=True)
        res = pd.DataFrame()
        grp = df.groupby(["variable", "stim", "lever"])
        res["count"] = grp["duration"].count()
        res["mean"] = grp["duration"].mean()
        res["std"] = grp["duration"].std()
        return res.reset_index()

    @precompute("session_duration_stats")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)

@p.register
@Data.from_class()
class EventDataframe:
    name = "session_duration"

    @staticmethod
    def location(folder, session):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Processing"/"duration.tsv"

    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location, selection):
        df: pd.DataFrame = db.run_action("load", "session_rtmt_data", selection, single=True)
        grp : pd.DataFrame = db.run_action("load", "session_duration_stats", selection, single=True)
        grp = grp.loc[~grp["stim"]].drop(columns="stim")
        df = df[["trial", "variable", "stim", "lever","duration"]].merge(grp, on=["variable", "lever"], how="left")
        df["session_nostim_zscore"] = (df["duration"] - df["mean"])/df["std"]
        df["nozscore"] = df.pop("duration")
        df = df.set_index(["trial", "variable", "stim", "lever"])[["nozscore", "session_nostim_zscore"]]
        df.columns.name = "zscore"
        df = df.stack().to_frame("duration").reset_index()
        return df

    @precompute("session_duration")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)

@p.register
@Data.from_class()
class EventDataframe:
    name = "session_task_counts"

    @staticmethod
    def location(folder, session):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Processing"/"session_task_counts.tsv"

    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location, selection):
        events: pd.DataFrame = db.run_action("load", "event_dataframe", selection, single=True)
        data=events
        data["event_priority"] = (data["family"] == 10).astype(int)
        data = data.sort_values(["t", "event_priority"])
        data["trial"] = (data["event_name"]=="line_change_to_2").cumsum()-1
        data = data.groupby("trial").apply(lambda d: d if len(d.loc[d["family"]==11].index) == 0 else pd.DataFrame()).reset_index(drop=True)
        data["line_change_counter"] = (data["family"] == 10).cumsum()-1
        data["node"] = np.where(data["family"] == 10, data["_T"], np.nan)
        data["node"] = data["node"].fillna(method="ffill")
        data["next_node"] = data["node"].where(data["family"] == 10).fillna(method="bfill").shift(-1)
        data["t_next"] = data["t"].shift(-1)
        data = data.loc[data["family"] == 10]
        res = data.value_counts(["node", "next_node"]).to_frame("count").reset_index()
        return res
    
    @precompute("session_task_counts")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)



pipeline=p