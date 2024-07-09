import logging, beautifullogger
from typing import List
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
import poly_graph
logger = logging.getLogger(__name__)


p = Database("poly")

@p.register
@Data.from_class()
class PolyGraph:
    name = "task_graph"

    @staticmethod
    def location(folder, stim_type, handedness, forced):
        return Path(folder)/"PolyAnalysis"/"TaskGraph"/f"{forced}_{stim_type}_{handedness}"/"task_graph.pdf"
        
    
    @staticmethod
    def compute(db: DatabaseInstance, out_location, selection):
        selection = {k:v for k,v in selection.items() if k!="condition"}
        task = db.run_action("load", "task_file", selection, single=True)
        graph = poly_graph.get_automaton(task)
        poly_graph.generate_pdf(out_location, graph=graph)
        return out_location


    @staticmethod
    def view(db, out_location, selection):
        graph_file = db.run_action("compute", "task_graph", selection, single=True)
        poly_graph.view_pdf(graph_file)

@p.register
@Data.from_class()
class SessionGraph:
    name = "session_graph"

    @staticmethod
    def location(folder, session, stim_type, handedness, forced):
        return Path(folder)/"PolyAnalysis"/"Sessions"/f"{session}_graph.pdf"
        
    @staticmethod
    def get_edge_dataframe(db: DatabaseInstance, out_location, selection):
        data = db.run_action("load", "event_data", selection, single=True)
        data = data.sort_values("t")
        data["is_line_2"] = (data["family"]==10) & (data["_T"]==2)
        data["trial"] = data["is_line_2"].cumsum() -1
        data = data.groupby("trial").apply(lambda d: d if len(d.loc[d["family"]==11].index) == 0 else pd.DataFrame()).reset_index(drop=True)
        combined = poly_graph.get_edge_dataframe(data)
        combined["duration"] = combined["t_next"] - combined["t"]
        return combined
    
    @staticmethod
    def compute(db: DatabaseInstance, out_location, selection):
        task = db.run_action("load", "task_file", selection, single=True)
        graph = poly_graph.get_automaton(task)
        edge_data = db.run_action("get_edge_dataframe", "session_graph", selection, single=True)
        annotations = edge_data.groupby("edge").apply(lambda d: pd.Series(dict(counts=len(d), avgt=(d["t_next"] - d["t"]).mean(), var=(d["t_next"] - d["t"]).var(), medt=(d["t_next"] - d["t"]).median())))
        annotations["avg(t):ms"] = np.round(annotations["avgt"]*1000)
        annotations["med(t):ms"] = np.round(annotations["medt"]*1000)
        poly_graph.attach_edge_attr(graph, annotations["counts"].to_dict(), "counts", default_value=0)
        poly_graph.attach_edge_attr(graph, annotations["avg(t):ms"].to_dict(), "avgtms")
        poly_graph.attach_edge_attr(graph, annotations["med(t):ms"].to_dict(), "medtms")
        poly_graph.attach_edge_attr(graph, annotations["var"].to_dict(), "var")
        poly_graph.generate_pdf(out_location, graph=graph)
        return out_location

@p.register
@Data.from_class()
class SessionGraph:
    name = "session_rtmt"

    @staticmethod
    def location(folder, session):
        return Path(folder)/"PolyAnalysis"/"Sessions"/f"{session}_RT_MT.pdf"
    
    @staticmethod
    def dataframe(db: DatabaseInstance, out_location, selection):
        edge_data = db.run_action("get_edge_dataframe", "session_graph", selection, single=True)
        df_dict={}
        match selection["forced"], selection["handedness"], selection["stim_type"]:
            case ("Left", "Ambi", _):
                df_dict["l1_rt_stim"] = edge_data.loc[edge_data["edge"]==(25, 26)]
                df_dict["l1_mt_stim"] = edge_data.loc[edge_data["edge"]==(26, 27)]
                df_dict["l1_rt_nostim"] = edge_data.loc[edge_data["edge"]==(5, 6)]
                df_dict["l1_mt_nostim"] = edge_data.loc[edge_data["edge"]==(6, 7)]
            case ("notforced", "Ambi", _) | ("notforced", _, "Conti"):
                df_dict["l1_rt_stim"] = edge_data.loc[edge_data["edge"]==(40, 41)]
                df_dict["l1_mt_stim"] = edge_data.loc[edge_data["edge"]==(41, 42)]
                df_dict["l1_rt_nostim"] = edge_data.loc[edge_data["edge"]==(5, 6)]
                df_dict["l1_mt_nostim"] = edge_data.loc[edge_data["edge"]==(6, 7)]
                df_dict["l2_rt_stim"] = edge_data.loc[edge_data["edge"]==(25, 26)]
                df_dict["l2_mt_stim"] = edge_data.loc[edge_data["edge"]==(26, 27)]
                df_dict["l2_rt_nostim"] = edge_data.loc[edge_data["edge"]==(8, 9)]
                df_dict["l2_mt_nostim"] = edge_data.loc[edge_data["edge"]==(9, 10)]
            case ("notforced", _, "Beta"):
                df_dict["l1_rt_stim"] = edge_data.loc[edge_data["edge"]==(40, 41)]
                df_dict["l1_mt_stim"] = edge_data.loc[edge_data["edge"]==(41, 42)]
                df_dict["l1_rt_nostim"] = edge_data.loc[edge_data["edge"]==(5, 6)]
                df_dict["l1_mt_nostim"] = edge_data.loc[edge_data["edge"]==(6, 7)]
                df_dict["l2_rt_stim"] = edge_data.loc[edge_data["edge"]==(33, 34)]
                df_dict["l2_mt_stim"] = edge_data.loc[edge_data["edge"]==(34, 35)]
                df_dict["l2_rt_nostim"] = edge_data.loc[edge_data["edge"]==(8, 9)]
                df_dict["l2_mt_nostim"] = edge_data.loc[edge_data["edge"]==(9, 10)]
            case _:
                raise Exception("Unknowm case")
        all_dfs = []
        for k, df in df_dict.items():
            df = df.copy()
            df["arrow_duration"] = df["duration"]
            if "mt" in k:
                def find_first_lever_press(d):
                    n = 1 if "l1" in k else 2
                    d = d.loc[(d["family"] == 2) & (d["nbre"] == n)]
                    d = d.sort_values("t")
                    return d["t"].iat[0]
                df["first_lever_press"] = df["inner_events"].apply(find_first_lever_press)
                df["duration"] = df["first_lever_press"] - df["t"]

            df=df[["duration", "arrow_duration"]]
            df["lever"] = "Left" if "l1" in k else "Right"
            df["value"] = "rt" if "rt" in k else "mt"
            df["stim"] = False if "nostim" in k else True
            all_dfs.append(df)

        all = pd.concat(all_dfs, ignore_index=True)
        return all
    
    @staticmethod
    def compute(db: DatabaseInstance, out_location, selection):
        import matplotlib.pyplot as plt
        import seaborn as sns
        all = db.run_action("dataframe", "session_rtmt", selection, single=True)
        sns.displot(data= all, common_norm=False, stat= "density", 
                    row="value", col="lever", hue="stim", x="duration", bins=100, kde=True, 
                    aspect=16/9, facet_kws=dict(margin_titles=True, gridspec_kws=dict(top=0.9)))
        plt.suptitle(",".join([f'{k}={v}' for k,v in selection.items() if not k in ["session", "folder"]]))
        out_location.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_location)
        return out_location
    
    @staticmethod
    def save_dataframe(db: DatabaseInstance, out_location: Path, selection):
        all: pd.DataFrame = db.run_action("dataframe", "session_rtmt", selection, single=True)
        out = out_location.with_suffix(".tsv")
        out.parent.mkdir(exist_ok=True, parents=True)
        all.to_csv(out, sep="\t", index=False)
        return out

@p.register
@Data.from_class()
class SessionGraph:
    name = "subject_rtmt"

    @staticmethod
    def location(folder, subject, hemi):
        return Path(folder)/"PolyAnalysis"/"Subjects"/f"{subject}_hemi{hemi}_RT_MT.pdf"
    
    @staticmethod
    def dataframe(db: DatabaseInstance, out_location, selection):
        dfs = db.run_action("dataframe", "session_rtmt", selection)
        all_dfs = []
        for df in dfs["dataframe"].to_list():
            df: pd.DataFrame
            nostim: pd.DataFrame = df.loc[~df["stim"]]
            mean = nostim.groupby(["lever", "value"])["duration"].mean().to_frame("mean")
            std = nostim.groupby(["lever", "value"])["duration"].std(ddof=0).to_frame("std")
            joined = df.join(mean, on=["lever", "value"]).join(std, on=["lever", "value"])
            joined["duration_zscored"] =( joined["duration"] - joined["mean"] )/joined["std"]
            all_dfs.append(joined[["lever", "value",  "stim", "duration_zscored"]])
        all = pd.concat(all_dfs, ignore_index=True)
        return all
    
    @staticmethod
    def compute(db: DatabaseInstance, out_location: Path, selection):
        import matplotlib.pyplot as plt
        import seaborn as sns
        all = db.run_action("dataframe", "subject_rtmt", selection, single=True)
        sns.displot(data= all, common_norm=False, stat= "density", 
                    row="value", col="lever", hue="stim", x="duration_zscored", bins=100, kde=True, 
                    aspect=16/9, facet_kws=dict(margin_titles=True, gridspec_kws=dict(top=0.9)))
        plt.suptitle(",".join([f'{k}={v}' for k,v in selection.items() if not k in ["session", "folder"]]))
        out_location.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_location)
        return out_location

@p.register
@Data.from_class()
class SessionGraph:
    name = "rtmt"

    @staticmethod
    def location(folder, hemi, handedness, forced):
        return Path(folder)/"PolyAnalysis"/"RTMT"/f"{forced}_hemi{hemi}_hand{handedness}RT_MT.pdf"

    @staticmethod
    def dataframe(db: DatabaseInstance, out_location, selection):
        dfs = db.run_action("dataframe", "session_rtmt", selection)
        all_dfs = []
        for df in dfs["dataframe"].to_list():
            df: pd.DataFrame
            nostim: pd.DataFrame = df.loc[~df["stim"]]
            mean = nostim.groupby(["lever", "value"])["duration"].mean().to_frame("mean")
            std = nostim.groupby(["lever", "value"])["duration"].std(ddof=0).to_frame("std")
            joined = df.join(mean, on=["lever", "value"]).join(std, on=["lever", "value"])
            joined["duration_zscored"] =( joined["duration"] - joined["mean"] )/joined["std"]
            all_dfs.append(joined[["lever", "value",  "stim", "duration_zscored"]])
        all = pd.concat(all_dfs, ignore_index=True)
        return all
    
    @staticmethod
    def compute(db: DatabaseInstance, out_location: Path, selection):
        import matplotlib.pyplot as plt
        import seaborn as sns
        all = db.run_action("dataframe", "rtmt", selection, single=True)
        f = sns.displot(data= all, common_norm=False, stat= "density", 
                    row="value", col="lever", hue="stim", x="duration_zscored", bins=100, kde=True, 
                    aspect=16/9, facet_kws=dict(margin_titles=True, gridspec_kws=dict(top=0.9)))
        f.figure.set_layout_engine("constrained")
        plt.suptitle(",".join([f'{k}={v}' for k,v in selection.items() if not k in ["session", "folder"]]))
        out_location.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_location)
        return out_location





pipeline = p