

import logging, beautifullogger
from typing import List
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob, precompute
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
import poly_graph
import json, networkx as nx
logger = logging.getLogger(__name__)


p = Database("poly_preprocessing")

@p.register
@Data.from_class()
class EventDataframe:
    name = "event_dataframe"

    @staticmethod
    def location(folder, session):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Processing"/"event_dataframe.tsv"

    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False))
    def compute(db: DatabaseInstance, out_location, selection):
        import poly_graph
        data_location = db.run_action("location", "data_file", selection, single=True)
        df =  poly_graph.load_poly_data_file(data_location)
        df["event_name"] = np.where(df["family"] == 10, "line_change_to_" + df["_T"].astype(str),
                           np.where((df["family"] == 2) & (df["nbre"] == 1) & (df["_V"] ==0), "left_lever_lift",
                           np.where((df["family"] == 2) & (df["nbre"] == 1) & (df["_V"] ==1), "left_lever_press",
                           np.where((df["family"] == 2) & (df["nbre"] == 2) & (df["_V"] ==0), "right_lever_lift",
                           np.where((df["family"] == 2) & (df["nbre"] == 2) & (df["_V"] ==1), "right_lever_press",         
                           np.where((df["family"] == 6) & (df["nbre"] == 22) & (df["_P"] ==0) & (df["_V"] ==0), "right_pad_lift",
                           np.where((df["family"] == 6) & (df["nbre"] == 22) & (df["_P"] ==1) & (df["_V"] ==0), "right_pad_press",
                           np.where((df["family"] == 6) & (df["nbre"] == 22) & (df["_P"] ==0) & (df["_V"] ==1), "left_pad_lift",
                           np.where((df["family"] == 6) & (df["nbre"] == 22) & (df["_P"] ==1) & (df["_V"] ==1), "left_pad_press",
                           np.where(df["family"] == 11, "pause_pressed",
                           "unprocessed"))))))))))
        return df
    
    @precompute("event_dataframe")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)



@p.register
@Data.from_class()
class TaskGraph:
    name = "task_graph"

    @staticmethod
    def location(folder, task):
        return Path(folder)/ "PolyAnalysis" / "Tasks" / f"{task}"/ f"graph.json"

    @staticmethod
    @cache(lambda out, graph: json.dump(nx.cytoscape_data(graph), out.open("w"), indent=4), force_recompute=False)
    def compute(db: DatabaseInstance, out_location: Path, selection):
        import networkx as nx
        task_file = db.run_action("location", "task_file", selection, single=True)
        task_df = poly_graph.load_task_file(task_file)
        graph: nx.DiGraph = poly_graph.get_automaton(task_df)
        for node in graph.nodes:
            names=[]
            for i in range(3):
                if not f"T{i}" in graph.nodes(data=True)[node]:
                    continue
                desc = graph.nodes(data=True)[node][f"T{i}"]
                import re
                m = re.match(r'\d*_(\w+)$', desc)
                if not m is None:
                     names.append(m.group(1))
            if "ASND(6,20)" in graph.nodes(data=True)[node]:
                if graph.nodes(data=True)[node][f"ASND(6,20)"] == "on(50,1,1,40,5000)":
                    names.append("CueRight")
                if graph.nodes(data=True)[node][f"ASND(6,20)"] == "on(50,1,1,40,1000)":
                    names.append("CueLeft")
            if "TTLP6(15,6)" in graph.nodes(data=True)[node]:
                if str(graph.nodes(data=True)[node][f"TTLP6(15,6)"]).startswith("ON"):
                    names.append("StimBeta")
            if "TTLP2(15,2)" in graph.nodes(data=True)[node]:
                if str(graph.nodes(data=True)[node][f"TTLP2(15,2)"]).startswith("ON"):
                    names.append("StimConti")

                   
            graph.nodes(data=True)[node]["names"] = names
        return graph
    
    @precompute("task_graph")
    def load(db: DatabaseInstance, out_location, selection): return nx.cytoscape_graph(json.load(out_location.open("r")))
        

@p.register
@Data.from_class()
class TaskGraph:
    name = "task_pdf"

    @staticmethod
    def location(folder, task):
        return Path(folder)/ "PolyAnalysis" / "Tasks" / f"{task}"/ f"graph.pdf"

    @staticmethod
    @cache(lambda out, graph: poly_graph.generate_output(pdf_location=out, graph=graph), force_recompute=False)
    def compute(db: DatabaseInstance, out_location: Path, selection):
        return db.run_action("load", "task_graph", selection, single=True)


pipeline=p