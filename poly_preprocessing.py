

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
class TaskEdgeMapping:
    name = "task_graph_metadata"

    @staticmethod
    def location(folder, task):
        return Path(folder)/ "PolyAnalysis" / "Tasks" /f"{task}"/ f"graph_metadata.json"

    @staticmethod
    @cache(lambda out, d: json.dump(d, out, indent=4), open="w",  force_recompute=False)
    def compute(db: DatabaseInstance, out_location, selection):
        match selection["forced"], selection["handedness"], selection["stim_type"]:
            case ("Left", "Ambi", _):
                edges = { (25, 26): "rt_stim_leftlever", (26, 27): "mt_stim_leftlever", (5, 6): "rt_nostim_leftlever",  (6, 7): "mt_nostim_leftlever"}
            case ("notforced", "Ambi", _) | ("notforced", _, "Conti"):
                edges = {(25, 26): "rt_stim_rightlever", (26, 27): "mt_stim_rightlever", (5, 6): "rt_nostim_leftlever",  (6, 7): "mt_nostim_leftlever",
                        (40, 41): "rt_stim_leftlever", (41, 42): "mt_stim_leftlever", (8, 9): "rt_nostim_rightlever",  (9, 10): "mt_nostim_rightlever"
                        }
            case ("notforced", _, "Beta"):
                edges = {(33, 34): "rt_stim_rightlever", (34, 35): "mt_stim_rightlever", (5, 6): "rt_nostim_leftlever",  (6, 7): "mt_nostim_leftlever",
                        (40, 41): "rt_stim_leftlever", (41, 42): "mt_stim_leftlever", (8, 9): "rt_nostim_rightlever",  (9, 10): "mt_nostim_rightlever"
                        }
        def compute_edge_metadata(edge, edge_name):
            return dict(
                alias=edge_name, 
                start=edge[0],
                end=edge[1],
                data=dict(lever = "Left" if "left" in edge_name else "Right",
                stim = False if "nostim" in edge_name else True,
                variable = "rt" if "rt" in edge_name else "mt"),
                )
        
        edges = [compute_edge_metadata(k, v) for k, v in edges.items()]
        nodes = [dict(node=2, alias="Trial_start", loop_start=True)]

        return dict(edges=edges, nodes=nodes)
        
    
    @precompute("task_graph_metadata")
    def load(db: DatabaseInstance, out_location, selection): 
        with out_location.open("r") as f:
            return json.load(f)

@p.register
@Data.from_class()
class TaskGraph:
    name = "task_graph"

    @staticmethod
    def location(folder, task):
        return Path(folder)/ "PolyAnalysis" / "Tasks" / f"{task}"/ f"graph.json"

    @staticmethod
    @cache(lambda out, graph: json.dump(nx.cytoscape_data(graph), out.open("w"), indent=4))
    def compute(db: DatabaseInstance, out_location: Path, selection):
        import networkx as nx
        task_file = db.run_action("location", "task_file", selection, single=True)
        task_df = poly_graph.load_task_file(task_file)
        graph: nx.DiGraph = poly_graph.get_automaton(task_df)
        metadata = db.run_action("load", "task_graph_metadata", selection, single=True)
        for node in metadata["nodes"]:
            n = node["node"]
            for k, v in node.items():
                if k in ["node"]:
                    continue
                graph.nodes[n][k] =v
        for edge in metadata["edges"]:
            s = edge["start"]
            e = edge["end"]
            for k, v in edge.items():
                if k in ["start", "end"]:
                    continue
                graph.edges[(s,e)][k] =v
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
    @cache(lambda out, graph: poly_graph.generate_output(pdf_location=out, graph=graph))
    def compute(db: DatabaseInstance, out_location: Path, selection):
        return db.run_action("load", "task_graph", selection, single=True)


pipeline=p