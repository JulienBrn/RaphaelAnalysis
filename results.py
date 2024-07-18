import logging, beautifullogger
from typing import List
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob, precompute
import pandas as pd, numpy as np, xarray as xr, string
from pathlib import Path
import poly_graph
import json, networkx as nx, functools
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import seaborn as sns
import warnings, datetime
from babel.dates import format_date

p = Database("results")
groups =  ["stim_type","hemi", "condition","handedness", "forced", "task"]
def tsvload(db, out_location, selection):
    return pd.read_csv(out_location, sep="\t", index_col=None)



@cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
def compute_trial_counts(db, out_location, selection):
    dfs = db.run_action("load", "session_trial_types", selection)
    all = pd.concat([row["load"] for _, row in dfs.iterrows()], ignore_index=True)
    return all[["sucess"] + [col for col in all.columns if not "trial" in col and not "sucess" in col]].value_counts(dropna=False).to_frame("count").reset_index().sort_values("sucess")

@cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
def compute_trialpair_counts(db, out_location, selection):
    dfs = db.run_action("load", "session_trialpair_types", selection)
    all = pd.concat([row["load"] for _, row in dfs.iterrows()], ignore_index=True)
    return all[["sucess", "sucess_next"] + [col for col in all.columns if not "trial" in col and not "sucess" in col]].value_counts(dropna=False).to_frame("count").reset_index().sort_values(["sucess", "sucess_next"])

@cache(lambda out, graph: poly_graph.generate_output(pdf_location=out, graph=graph))
def compute_edgecounts(db, out_location, selection):
    graph: nx.DiGraph = db.run_action("load", "task_graph", selection, single=True)
    counts =  db.run_action("load", "session_task_counts", selection)["load"].to_list()
    res = {e: 0 for e in graph.edges}
    for df in counts:
        for _, row in df.iterrows():
            res[(row["node"], row["next_node"])] += row["count"]
    nx.set_edge_attributes(graph, res, "count")
    graph = graph.edge_subgraph([e for e in graph.edges if graph.edges[e]["count"] > 0])
    return graph

@cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
def compute_rtmt_data(db: DatabaseInstance, out_location, selection):
    dfs = db.run_action("load", "zscored_session_rtmt", selection)
    all = pd.concat([row["load"].assign(session=row["session"]) for _, row in dfs.iterrows()], ignore_index=True)
    return all

def compute_rtmt_stats(data_name):
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def inner(db: DatabaseInstance, out_location, selection):
        df = db.run_action("load", data_name, selection, single=True)
        res = pd.DataFrame()
        grp = df.groupby(["variable", "stim", "lever"])
        res["count"] = grp["duration"].count()
        res["mean"] = grp["duration"].mean()
        res["std"] = grp["duration"].std()
        return res.reset_index()
    return inner

def compute_rtmt_pvalue(data_name):
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def inner(db: DatabaseInstance, out_location, selection):
        import scipy.stats
        df: pd.DataFrame = db.run_action("load", data_name, selection, single=True)
        res = pd.DataFrame()
        grp = df.groupby(["variable", "lever"])
        res["ttest_pvalue"] = grp.apply(lambda d: scipy.stats.ttest_ind(d.loc[d["stim"]]["duration"], d.loc[~d["stim"]]["duration"]).pvalue)
        return res.reset_index()
    return inner

def compute_rtmt_figure(data_name):
    @cache(lambda out, f: f.savefig(out), force_recompute=False)
    def inner(db: DatabaseInstance, out_location, selection):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all = db.run_action("load", data_name, selection, single=True)
        
            dur_name = "duration" if selection["zscore"] == "nozscore" else f'duration(zscored)'
            all[dur_name] = all.pop("duration")
            f = sns.displot(data= all, common_norm=False, stat= "density", 
                        row="variable", col="lever", hue="stim", x=dur_name, bins=100, kde=True, 
                        aspect=(16/9)*all["variable"].nunique()/all["lever"].nunique(), facet_kws=dict(margin_titles=True, gridspec_kws=dict(top=0.9)))
            plt.suptitle(",".join([f'{k}={v if not isinstance(v, datetime.datetime) else format_date(v, "dd MMM yyyy", locale="en")}' for k,v in selection.items() if not k in ["session", "folder", "task"]]), fontsize="small")
            return plt.gcf()
    return inner


def compute_trial_counts_fig(data_name):
    @cache(lambda out, f: f.savefig(out), force_recompute=False)
    def inner(db, out_location, selection):
        df: pd.DataFrame = db.run_action("load", data_name, selection, single=True)
        df = df.loc[df["sucess"]!="nort"]
        res = df.set_index(["cue","stim", "sucess"])["count"].unstack("sucess")
        res["sum"] = res.sum(axis=1).astype(int)
        res["ratio"] = res["good"]/res["sum"]
        res = res.reset_index()
        res["cue_index"] = np.where(res["cue"] == "Left", 0, 1)
        res["stim_index"] = np.where(res["stim"], 0, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.figure(figsize=(16, 9))
            for s, grp in res.groupby("stim"):
                r = plt.bar(grp["cue_index"] + grp["stim_index"]*0.3, 100*grp["ratio"], 0.3, label=f"stim={s}")
                for _, row in grp.iterrows():
                    plt.text(row["cue_index"] + row["stim_index"]*0.3, 100*row["ratio"]+1, f'n_trials={row["sum"]}', horizontalalignment="center")
            
            c= res.drop_duplicates(["cue"])
            plt.xticks(c["cue_index"]+0.15, c["cue"])
            plt.xlabel("Cue")
            plt.ylabel("% correct trials among trials where mouvement was initiated")
            plt.suptitle(",".join([f'{k}={v if not isinstance(v, datetime.datetime) else format_date(v, "dd MMM yyyy", locale="en")}' for k,v in selection.items() if not k in ["session", "folder", "task"]]), fontsize="small")
            plt.legend()
        return plt.gcf()
    return inner


def compute_trialpair_counts_fig(data_name):
    @cache(lambda out, f: f.savefig(out), force_recompute=True)
    def inner(db, out_location, selection):
        df: pd.DataFrame = db.run_action("load", data_name, selection, single=True)
        # print(df)
        df = df.loc[(df["sucess"]!="nort") &  (df["sucess_next"]!="nort")]
        res = df.set_index(["cue","stim", "sucess", "cue_next", "stim_next", "sucess_next"])["count"].unstack("sucess_next")
        res["sum"] = res.sum(axis=1).astype(int)
        res["ratio"] = res["good"]/res["sum"]
        res = res.reset_index()
        res["cue_index"] = np.where(res["cue"] == "Left", 0, 1)
        res["stim_index"] = np.where(res["stim"], 0, 1)
        res["cue_index_next"] = np.where(res["cue_next"] == "Left", 0, 1)
        res["stim_index_next"] = np.where(res["stim_next"], 0, 1)
        res["sucess_index"] = np.where(res["sucess"]=="good", 0, 1)
        res["sucess"] = np.where(res["sucess"]=="good", "good", "fail")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = plt.figure(figsize=(16, 9))
            axs = f.subplots(2, 2)
            for stim in range(2):
                for cue in range(2):
                    ax: plt.Axes = axs[stim , cue]
                    d = res.loc[(res["stim_index"]==stim) & (res["cue_index"]==cue)]
                    if len(d.index) > 0:
                        grps = d.groupby(["stim_next", "sucess"])
                        for s, grp in grps:
                            ax.bar(grp["cue_index_next"] + grp["stim_index_next"]*0.1 + 2*grp["sucess_index"]*0.1, 100*grp["ratio"], 0.1,
                                    color="green" if grp["sucess_index"].iat[0]==0 else "red", edgecolor="black", hatch = "*" if grp["stim_index_next"].iat[0]==0 else "", label=f"(stim, prev)={s}")
                            for _, row in grp.iterrows():
                                ax.text(row["cue_index_next"] + row["stim_index_next"]*0.1 + 2*row["sucess_index"]*0.1, 100*row["ratio"]+1, f'n={row["sum"]}', horizontalalignment="center", fontsize="small")
                
                        c= d.drop_duplicates(["cue_next"])
                        ax.set_xticks(c["cue_index_next"]+0.2, c["cue_next"])
                        ax.set_xlabel("Cue")
                        ax.set_title(f'prev_stim={d["stim"].iat[0]}, prev_cue={d["cue"].iat[0]}', fontsize="small")
                        ax.set_ylabel("% correct trials among trials where mouvement was initiated", fontsize="x-small")
            plt.suptitle(",".join([f'{k}={v if not isinstance(v, datetime.datetime) else format_date(v, "dd MMM yyyy", locale="en")}' for k,v in selection.items() if not k in ["session", "folder", "task"]]), fontsize="small")
            # plt.legend()
            plt.tight_layout()
        return f
    return inner





functions = pd.DataFrame([
    ["trial_counts", "tsv", dict(compute=lambda n: compute_trial_counts, load=lambda n: precompute(n)(tsvload)), []],
    ["trial_counts_fig", "pdf", dict(compute=lambda n: compute_trial_counts_fig(n.replace("_fig", ""))), []],
    ["trialpair_counts", "tsv",  dict(compute=lambda n: compute_trialpair_counts, load=lambda n: precompute(n)(tsvload)), []],
    ["trialpair_counts_fig", "pdf", dict(compute=lambda n: compute_trialpair_counts_fig(n.replace("_fig", ""))), []],
    ["graphedge_counts", "pdf",  dict(compute=lambda n: compute_edgecounts), []],
    ["rtmt_resdata", "tsv",  dict(compute=lambda n: compute_rtmt_data, load=lambda n: precompute(n)(tsvload)), ["zscore"]],
    ["rtmt_stats", "tsv",  dict(compute=lambda n: compute_rtmt_stats(n.replace("stats", "resdata")), load=lambda n: precompute(n)(tsvload)), ["zscore"]],
    ["rtmt_pvalue", "tsv",  dict(compute=lambda n: compute_rtmt_pvalue(n.replace("pvalue", "resdata")), load=lambda n: precompute(n)(tsvload)), ["zscore"]],
    ["rtmt_figure", "pdf",  dict(compute=lambda n: compute_rtmt_figure(n.replace("figure", "resdata"))), ["zscore"]],
    ], columns=["name", "ext", "actions", "additional_deps"]
)

groupings = pd.DataFrame([
    ["session", ["session"], "{{folder}}/PolyAnalysis/Sessions/{{session}}/Results/{name}{fdeps}.{ext}"],
    ["subject", ["subject"], "{{folder}}/PolyAnalysis/Subjects/{{subject}}/{{task}}_hemi{{hemi}}_cond{{condition}}/{name}{fdeps}.{ext}"],
    ["all", [], "{{folder}}/PolyAnalysis/All/{{task}}_hemi{{hemi}}_cond{{condition}}/{name}{fdeps}.{ext}"],
    ], columns=["name", "grouping", "loc"]
)


def make_computer(name, params, location, actions):
    keys = [p[1] for p in string.Formatter().parse(location)]
    res =  Data(name, dependencies=set(params), get_location=lambda d: Path(location.format_map({k:v for k,v in d.items() if k in keys})), actions=actions)
    return res

for _, f in functions.iterrows():
    for _, grp in groupings.iterrows():
        name = grp["name"]+"_"+f["name"]
        params = ["folder"] + grp["grouping"] + groups + f["additional_deps"]
        deps_str = "" if not f["additional_deps"] else "_" + "_".join(["{"+d+"}" for d in f["additional_deps"]])
        loc = grp["loc"].format(name=f["name"], ext=f["ext"], fdeps = deps_str)
        actions = {k: v(name) for k, v in f["actions"].items()}
        p.declare(make_computer(name, params, loc, actions))



    



pipeline = p