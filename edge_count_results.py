import logging, beautifullogger
from typing import List
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob, precompute
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
import poly_graph
import json, networkx as nx
logger = logging.getLogger(__name__)


p = Database("edge_count_results")


def compute_edgecounts(db, selection):
    graph: nx.DiGraph = db.run_action("load", "task_graph", selection, single=True)
    counts =  db.run_action("load", "session_task_counts", selection)["load"].to_list()
    res = {e: 0 for e in graph.edges}
    for df in counts:
        for _, row in df.iterrows():
            res[(row["node"], row["next_node"])] += row["count"]
    nx.set_edge_attributes(graph, res, "count")
    return graph




@p.register
@Data.from_class()
class SessionEdgecounts:
    name = "session_edgecounts_results"

    @staticmethod
    def location(folder, session, task):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Results"/"graph_counts.pdf"
    
    @staticmethod
    @cache(lambda out, graph: poly_graph.generate_output(pdf_location=out, graph=graph))
    def compute(db: DatabaseInstance, out_location: Path, selection): return compute_edgecounts(db, selection)
        
@p.register
@Data.from_class()
class SubjectEdgecounts:
    name = "subject_edgecounts_results"

    @staticmethod
    def location(folder, subject, task, hemi):
        return Path(folder)/ "PolyAnalysis"/"Subjects"/f"{subject}"/"Counts"/f"counts_task_{task}_hemi{hemi}.pdf"
    
    @staticmethod
    @cache(lambda out, graph: poly_graph.generate_output(pdf_location=out, graph=graph))
    def compute(db: DatabaseInstance, out_location: Path, selection): return compute_edgecounts(db, selection)
        
@p.register
@Data.from_class()
class AllEdgecounts:
    name = "all_edgecounts_results"

    @staticmethod
    def location(folder, task, hemi):
        return Path(folder)/ "PolyAnalysis"/"All"/"Counts"/f"counts_task_{task}_hemi{hemi}.pdf"
    
    @staticmethod
    @cache(lambda out, graph: poly_graph.generate_output(pdf_location=out, graph=graph))
    def compute(db: DatabaseInstance, out_location: Path, selection): return compute_edgecounts(db, selection)
        

    
pipeline = p