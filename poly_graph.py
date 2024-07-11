import logging, beautifullogger
import sys
from typing import Dict, Any, Tuple
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
import networkx as nx, graphviz

logger = logging.getLogger(__name__)

def load_task_file(file: Path) -> pd.DataFrame:
    df = pd.read_csv(file, sep="\t", header=11)
    df = df.rename(columns={"Unnamed: 0":"line_num"})
    df = df.loc[~pd.isna(df["line_num"])]
    df["line_num"] = df["line_num"].astype(int)
    return df

def get_automaton(task_df: pd.DataFrame):
    df: pd.DataFrame = task_df
    graph = nx.DiGraph(size = "9, 16" )
    for _, row in df.iterrows():
        graph.add_node(row["line_num"], **row.dropna().to_dict())
        for col in [col for col in df.columns if "NEXT" in col]:
            if pd.isna(row[col]):
                continue
            import re
            pattern = r'\(.+\)$'
            ns = re.findall(pattern, row[col])
            if len(ns) == 0:
                next_line = row["line_num"]+1
                cond = row[col]
            elif len(ns) ==1:
                cond = row[col][:-len(ns[0])]
                nlname = ns[0][1: -1]
                if re.match(r'\d+', nlname):
                    next_line = int(nlname)
                else:
                    next_line = df.loc[(df[["T1", "T2", "T3"]].apply(lambda s: s.str.lstrip("_")) == nlname).any(axis=1)]["line_num"]
                    if len(next_line) != 1:
                        raise Exception(f"problem {len(next_line)} {nlname}")
                    next_line = next_line.iat[0]
            else:
                raise Exception("Problem")
            graph.add_edge(row["line_num"], next_line, cond=cond)
    return graph

def generate_dotfile(out_location: Path, graph: nx.DiGraph):
    import networkx as nx
    graph = graph.edge_subgraph(graph.edges)
    node_labels = {node: "\n".join([f'{k}={v}' for k,v in graph.nodes[node].items() if not "NEXT" in k]) for node in graph.nodes}
    edge_labels = {edge: "\n".join([f'{k}={v}' for k,v in graph.edges[edge].items()]) for edge in graph.edges}
    nx.set_node_attributes(graph, node_labels, "label")
    nx.set_edge_attributes(graph, edge_labels, "label")
    for node in graph.nodes:
        attrs = []
        for attr in graph.nodes(data=True)[node]:
                if attr != "label":
                    attrs.append(attr)
        for attr in attrs:
            graph.nodes[node].pop(attr, None)
    nx.nx_pydot.write_dot(graph, out_location)

def view_pdf(pdf_location: Path):
    graphviz.view(pdf_location)


def default_node_formatter(attrs):
    labels = []
    other_attr = {}
    if "alias" in attrs:
        labels.append(f'{attrs["alias"]}')
        other_attr["color"]="green"
    if "line_num" in attrs:
        labels.append(f'line={attrs["line_num"]}')
    labels+=[f'{k}={v}' for k,v in attrs.items() if not k in ["alias", "line_num", "loop_start", "id", "value", "name", "data"]]
    return dict(label="\n".join(labels), **other_attr)

def default_edge_formatter(attrs):
    labels = []
    other_attr = {}
    if "alias" in attrs:
        labels.append(f'{attrs["alias"]}')
        other_attr["arrowsize"]=2
        other_attr["color"]="red"
    labels+=[f'{k}={v}' for k,v in attrs.items() if not k in ["alias", "source", "target", "data"]]
    return dict(label="\n".join(labels), **other_attr)


def change_graph_attributes(graph, node_attr = lambda attr: attr, edge_attr=lambda attr:attr):
    from copy import deepcopy
    graph = deepcopy(graph)
    for node in graph.nodes:
        attrs = {}
        for attr in graph.nodes(data=True)[node]:
            attrs[attr] = graph.nodes(data=True)[node][attr]
        for attr in attrs:
            graph.nodes[node].pop(attr, None)
        for k, v in node_attr(attrs).items():
            graph.nodes[node][k] = v

    for edge in graph.edges:
        attrs = {}
        for attr in graph.edges[edge]:
            attrs[attr] = graph.edges[edge][attr]
        for attr in attrs:
            graph.edges[edge].pop(attr, None)
        for k, v in edge_attr(attrs).items():
            graph.edges[edge][k] = v
    return graph

def generate_output(pdf_location=None, dot_location=None, graph=None, use_dot=False,view=False, format_graph=(default_node_formatter, default_edge_formatter)):
    rm_dotfile = False
    if not use_dot:
        if graph is None:
            raise Exception("Graph or dot file needs to be provided")
        if dot_location is None:
            dot_location = pdf_location.with_suffix(".tmp.dot")
            rm_dotfile=True
        pdf_location.parent.mkdir(exist_ok=True, parents=True)
        nx.nx_pydot.write_dot(change_graph_attributes(graph, *format_graph), dot_location)
    if not pdf_location is None:
        graphviz.render("dot", filepath=dot_location, outfile=pdf_location, format="pdf")
        if rm_dotfile:
            dot_location.unlink()
        if view:
            view_pdf(pdf_location)


def load_poly_data_file(file: Path) -> pd.DataFrame:
    df =  pd.read_csv(file, sep="\t", names=['time (ms)', 'family', 'nbre', '_P', '_V', '_L', '_R', '_T', '_W', '_X', '_Y', '_Z'], skiprows=13)
    df.insert(0, "t", df["time (ms)"]/1000)
    return df

def get_line_counters(data: Path | str | pd.DataFrame) -> Dict[int, int]:
    if not hasattr(data, "groupby"):
        data = load_poly_data_file(data)

    lines: pd.Series = data.loc[data["family"]==10]
    return lines["_T"].value_counts().to_dict()


def get_edge_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values("t")
    data["line_change_counter"] = (data["family"] == 10).cumsum()-1
    data["node"] = np.where(data["family"] == 10, data["_T"], np.nan)
    data["node"] = data["node"].fillna(method="ffill")
    data["edge"] = np.where(data["family"] == 10, (data["_T"], data["node"].shift(1)), np.nan)
    data["edge"] = data["edge"].fillna(method="ffill")
    data["next_t"] = data["t"].shift(1)
    return data

def get_edge_info(data: Path | str | pd.DataFrame) -> pd.DataFrame:
    if not hasattr(data, "groupby"):
        data = load_poly_data_file(data)

    lines: pd.DataFrame = data.loc[data["family"]==10].sort_values("t")
    combined = pd.concat([lines.iloc[:-1, :].reset_index(drop=True), lines.iloc[1:, :].rename(columns={k:k+"_next" for k in lines.columns}).reset_index(drop=True)], axis=1)
    combined["edge"] = combined.apply(lambda row: (row["_T"], row["_T_next"]), axis=1)
    result = combined.groupby("edge").apply(lambda d: pd.Series(dict(counts=len(d), avgt=(d["t_next"] - d["t"]).mean(), var=(d["t_next"] - d["t"]).var(), medt=(d["t_next"] - d["t"]).median())))
    return result

def attach_node_attr(graph: nx.DiGraph, prop: Dict[int, Any], prop_name: str, default_value=None):
    if not default_value is None:
        prop = {k: prop[k] if k in prop else default_value for k in graph.nodes}
    nx.set_node_attributes(graph, prop, prop_name)

def attach_edge_attr(graph: nx.DiGraph, prop: Dict[Tuple[int, int], Any], prop_name: str, default_value=None):
    if not default_value is None:
        prop = {(k1, k2): prop[k1, k2] if (k1,k2) in prop else default_value for k1,k2 in graph.edges}
    nx.set_edge_attributes(graph, prop, prop_name)
