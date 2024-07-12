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
    data = data.groupby("trial").apply(lambda d: d if len(d.loc[d["family"]==11].index) == 0 else pd.DataFrame()).reset_index(drop=True)
    data["line_change_counter"] = (data["family"] == 10).cumsum()-1
    data["node"] = np.where(data["family"] == 10, data["_T"], np.nan)
    data["node"] = data["node"].fillna(method="ffill")
    data["next_node"] = data["node"].where(data["family"] == 10).fillna(method="bfill").shift(-1)
    data["t_next"] = data["t"].shift(-1)
    is_fake_trial = data.groupby("trial").apply(lambda d: len(d.loc[(d["node"]==3) & (d["next_node"]==2) ].index) > 0).to_frame("is_fake_trial").reset_index()
    is_fake_trial["fake_trial_num"] = is_fake_trial["is_fake_trial"].cumsum()
    data = data.merge(is_fake_trial, how="left", on="trial")
    data["all_trial_num"] = data["trial"]
    data["trial"] =data["trial"] - data["fake_trial_num"]
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
    name = "session_rtmt_data"

    @staticmethod
    def location(folder, session):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Processing"/"rtmt_data.tsv"

    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location, selection):
        graph: nx.DiGraph = db.run_action("load", "task_graph", selection, single=True)
        edges =pd.DataFrame([dict(node=edge[0], next_node=edge[1], **graph.edges[edge]["data"]) for edge in graph.edges if "data" in graph.edges[edge]])
        data=db.run_action("load", "session_event_task", selection, single=True)
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
@CoordComputer.from_function()
def zscore():
    return ["nozscore", "session_nostim_zscore"]


@p.register
@Data.from_class()
class EventDataframe:
    name = "session_rtmt_zscore_adjust"

    @staticmethod
    def location(folder, session, zscore):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Processing"/f"rtmt_{zscore}_adjust.tsv"

    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location, selection):
        if selection["zscore"] == "nozscore":
            return pd.DataFrame([[1, 0, "y=a*(x-b)"]], columns=["a", "b", "adjust_formula"])
        elif selection["zscore"] == "session_nostim_zscore":
            df: pd.DataFrame = db.run_action("load", "session_rtmt_data", selection, single=True)
            df = df.loc[~df["stim"]]
            res = pd.DataFrame()
            grp = df.groupby(["variable", "lever"])
            res["a"] = grp["duration"].std()
            res["b"] = grp["duration"].mean()
            res["adjust_formula"] =  "y=a*(x-b)"
            return res.reset_index()

    @precompute("session_rtmt_zscore_adjust")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)

@p.register
@Data.from_class()
class EventDataframe:
    name = "zscored_session_rtmt"

    @staticmethod
    def location(folder, session, zscore):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Processing"/f"rtmt_{zscore}.tsv"

    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location, selection):
        df: pd.DataFrame = db.run_action("load", "session_rtmt_data", selection, single=True)
        grp : pd.DataFrame = db.run_action("load", "session_rtmt_zscore_adjust", selection, single=True)
        if len(grp.columns) ==3:
            df = df[["trial", "variable", "stim", "lever","duration"]].merge(grp, how="cross")
        else:
            df = df[["trial", "variable", "stim", "lever","duration"]].merge(grp, on=[c for c in grp.columns if not c in ["a", "b", "adjust_formula"]], how="left")
        df["duration_nonadjusted"] = df["duration"]
        if (df["adjust_formula"] != "y=a*(x-b)").any():
            raise Exception("Problem")
        df["duration"] = df["a"] * (df["duration_nonadjusted"] - df["b"])
        return df.drop(columns=["a", "b", "adjust_formula"])

    @precompute("zscored_session_rtmt")
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
        data=db.run_action("load", "session_event_task", selection, single=True)
        data = data.loc[data["family"] == 10]
        res = data.value_counts(["node", "next_node"]).to_frame("count").reset_index()
        return res
    
    @precompute("session_task_counts")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)

@p.register
@Data.from_class()
class TrialTypes:
    name = "session_trial_types"

    @staticmethod
    def location(folder, session):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Processing"/"session_trial_types.tsv"

    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location, selection):
        graph: nx.DiGraph = db.run_action("load", "task_graph", selection, single=True)
        edges =pd.DataFrame([dict(node=edge[0], next_node=edge[1], **graph.edges[edge]["data"]) for edge in graph.edges if "data" in graph.edges[edge]])
        data=db.run_action("load", "session_event_task", selection, single=True)
        data = data.loc[data["family"] == 10]
        data = data.merge(edges, how="left", on=["node", "next_node"])
        data = data.loc[data["trial"] >=0]
        data = data.loc[~data["is_fake_trial"]]
        
    
        def trial_info(d: pd.DataFrame):
            rt = d.loc[d["variable"]=="rt"].dropna()
            if len(rt.index) == 0:
                res = dict(sucess="nort")
            elif len(rt.index) == 1:
                cue = rt["lever"].iat[0]
                stim = rt["stim"].iat[0]
                sucess = "good" if len(d.loc[d["variable"]=="mt"].dropna().index) > 0 else "nomt"
                res = dict(cue=cue, stim=stim, sucess = sucess)
            else:
                raise Exception("Problem")
            return pd.Series(res, name="trial_type").rename_axis("trial_param")
        res = data.groupby("trial").apply(trial_info).unstack("trial_param").reset_index().sort_values("trial")
        return res
    
    @precompute("session_trial_types")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)



@p.register
@Data.from_class()
class TrialTypes:
    name = "session_trialpair_types"

    @staticmethod
    def location(folder, session):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Processing"/"session_trialpair_types.tsv"

    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location, selection):
        trial_types = db.run_action("load", "session_trial_types", selection, single=True)
        trial_types["next_trial"] = trial_types["trial"]+1
        trial_pair = pd.merge(trial_types, trial_types, left_on="next_trial", right_on="trial", suffixes=("", "_next"), how="inner").drop(columns=["next_trial", "next_trial_next"])
        return trial_pair
    
    @precompute("session_trialpair_types")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)


# @p.register
# @Data.from_class()
# class TrialTypes:
#     name = "session_trial_counts"

#     @staticmethod
#     def location(folder, session):
#         return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Processing"/"session_trial_counts.tsv"

#     @staticmethod
#     @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
#     def compute(db: DatabaseInstance, out_location, selection):
#         trial_types = db.run_action("load", "session_trial_types", selection, single=True)
#         return trial_types[["sucess"] + [col for col in trial_types.columns if not "trial" in col and not "sucess" in col]].value_counts(dropna=False).to_frame("count").reset_index().sort_values(["sucess"])
    
#     @precompute("session_trial_counts")
#     def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)



# @p.register
# @Data.from_class()
# class TrialTypes:
#     name = "session_trialpair_counts"

#     @staticmethod
#     def location(folder, session):
#         return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Processing"/"session_trialpair_counts.tsv"

#     @staticmethod
#     @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
#     def compute(db: DatabaseInstance, out_location, selection):
#         trial_types = db.run_action("load", "session_trialpair_types", selection, single=True)
#         return trial_types[["sucess", "sucess_next"] + [col for col in trial_types.columns if not "trial" in col and not "sucess" in col]].value_counts(dropna=False).to_frame("count").reset_index().sort_values(["sucess", "sucess_next"])
    
#     @precompute("session_trialpair_counts")
#     def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)




pipeline=p