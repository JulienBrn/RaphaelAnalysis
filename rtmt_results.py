import logging, beautifullogger
from typing import List
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob, precompute
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
import poly_graph
import json, networkx as nx
logger = logging.getLogger(__name__)


p = Database("rtmt_results")
groups =  ["stim_type","hemi", "condition","handedness", "forced"]

def compute_rtmt_results(db: DatabaseInstance, selection):
    dfs = db.run_action("load", "session_duration", selection)
    all = pd.concat([row["load"].assign(session=row["session"], **{k: row[k] for k in groups}) for _, row in dfs.iterrows()], ignore_index=True)
    return all

def compute_rtmt_stats(db: DatabaseInstance, result_name, selection):
    df = db.run_action("load", result_name, selection, single=True)
    res = pd.DataFrame()
    grp = df.groupby(["zscore", "variable", "stim", "lever"] + groups)
    res["count"] = grp["duration"].count()
    res["mean"] = grp["duration"].mean()
    res["std"] = grp["duration"].std()
    return res.reset_index()

def compute_rtmt_pvalue(db: DatabaseInstance, result_name, selection):
    import scipy.stats
    df: pd.DataFrame = db.run_action("load", result_name, selection, single=True)
    res = pd.DataFrame()
    grp = df.groupby(["zscore", "variable", "lever"] + groups)
    res["ttest_pvalue"] = grp.apply(lambda d: scipy.stats.ttest_ind(d.loc[d["stim"]]["duration"], d.loc[~d["stim"]]["duration"]).pvalue)
    return res.reset_index()

def compute_rtmt_figure(db: DatabaseInstance, result_name, selection):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings, datetime
    from babel.dates import format_date
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all = db.run_action("load", result_name, selection, single=True)
        for k in ["zscore"] + groups:
            if k in selection:
                all = all.loc[all[k]==selection[k]]

        dur_name = "duration" if selection["zscore"] == "nozscore" else f'duration ({selection["zscore"]})'
        all[dur_name] = all.pop("duration")
        f = sns.displot(data= all, common_norm=False, stat= "density", 
                    row="variable", col="lever", hue="stim", x=dur_name, bins=100, kde=True, 
                    aspect=(16/9)*all["variable"].nunique()/all["lever"].nunique(), facet_kws=dict(margin_titles=True, gridspec_kws=dict(top=0.9)))
        plt.suptitle(",".join([f'{k}={v if not isinstance(v, datetime.datetime) else format_date(v, "dd MMM yyyy", locale="en")}' for k,v in selection.items() if not k in ["session", "folder"]]), fontsize="small")
        return plt.gcf()


@p.register
@CoordComputer.from_function()
def zscore():
    return ["nozscore", "session_nostim_zscore"]



@p.register
@Data.from_class()
class SessionRTMT:
    name = "session_rtmt_results"

    @staticmethod
    def location(folder, session):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Results"/"rtmt.tsv"
    
    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location: Path, selection): return compute_rtmt_results(db, selection)
    
    @precompute("session_rtmt_results")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)
    
@p.register
@Data.from_class()
class SessionStat:
    name = "session_stat_results"

    @staticmethod
    def location(folder, session):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Results"/"stats.tsv"
    
    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location: Path, selection): return compute_rtmt_stats(db, "session_rtmt_results", selection)
    
    @precompute("session_stat_results")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)
    
@p.register
@Data.from_class()
class SessionPValue:
    name = "session_pvalue_results"

    @staticmethod
    def location(folder, session):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Results"/"significance.tsv"
    
    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location: Path, selection): return compute_rtmt_pvalue(db, "session_rtmt_results", selection)
    
    @precompute("session_pvalue_results")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)

@p.register
@Data.from_class()
class SessionFigure:
    name = "session_rtmt_figure_results"

    @staticmethod
    def location(folder, session, zscore):
        return Path(folder)/ "PolyAnalysis"/"Sessions"/f"{session}"/"Results"/f"rtmt_fig_{zscore}.pdf"
    
    @staticmethod
    @cache(lambda out, f: f.savefig(out), force_recompute=False)
    def compute(db: DatabaseInstance, out_location: Path, selection): return compute_rtmt_figure(db, "session_rtmt_results", selection)
        









@p.register
@Data.from_class()
class SubjectRTMT:
    name = "subject_rtmt_results"

    @staticmethod
    def location(folder, subject):
        return Path(folder)/ "PolyAnalysis"/"Subjects"/f"{subject}"/"rtmt.tsv"
    
    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location: Path, selection): return compute_rtmt_results(db, selection)
    
    @precompute("subject_rtmt_results")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)
    
@p.register
@Data.from_class()
class SubjectStat:
    name = "subject_stat_results"

    @staticmethod
    def location(folder, subject):
        return Path(folder)/ "PolyAnalysis"/"Subjects"/f"{subject}"/"stats.tsv"
    
    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location: Path, selection): return compute_rtmt_stats(db, "subject_rtmt_results", selection)
    
    @precompute("subject_stat_results")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)
    
@p.register
@Data.from_class()
class SubjectPValue:
    name = "subject_pvalue_results"

    @staticmethod
    def location(folder, subject):
        return Path(folder)/ "PolyAnalysis"/"Subjects"/f"{subject}"/"significance.tsv"
    
    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location: Path, selection): return compute_rtmt_pvalue(db, "subject_rtmt_results", selection)
    
    @precompute("subject_pvalue_results")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)
    

@p.register
@Data.from_class()
class SubjectFigure:
    name = "subject_rtmt_figure_results"

    @staticmethod
    def location(folder, subject, task, hemi, condition, zscore):
        return Path(folder)/ "PolyAnalysis"/"Subjects"/f"{subject}"/f"rtmt_fig_cond{condition}_hemi{hemi}_{task}_{zscore}.pdf"
    
    @staticmethod
    @cache(lambda out, f: f.savefig(out), force_recompute=False)
    def compute(db: DatabaseInstance, out_location: Path, selection): return compute_rtmt_figure(db, "subject_rtmt_results", selection)
        







@p.register
@Data.from_class()
class AllRTMT:
    name = "all_rtmt_results"

    @staticmethod
    def location(folder):
        return Path(folder)/ "PolyAnalysis"/"All"/"rtmt.tsv"
    
    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location: Path, selection): return compute_rtmt_results(db, selection)
    
    @precompute("all_rtmt_results")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)
    
@p.register
@Data.from_class()
class AllStat:
    name = "all_stat_results"

    @staticmethod
    def location(folder):
        return Path(folder)/ "PolyAnalysis"/"All"/"stats.tsv"
    
    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location: Path, selection): return compute_rtmt_stats(db, "all_rtmt_results", selection)
    
    @precompute("all_stat_results")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)
    
@p.register
@Data.from_class()
class AllPValue:
    name = "all_pvalue_results"

    @staticmethod
    def location(folder):
        return Path(folder)/ "PolyAnalysis"/"All"/"significance.tsv"
    
    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location: Path, selection): return compute_rtmt_pvalue(db, "all_rtmt_results", selection)
    
    @precompute("all_pvalue_results")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)
    

@p.register
@Data.from_class()
class AllFigure:
    name = "all_rtmt_figure_results"

    @staticmethod
    def location(folder, task, hemi, condition, zscore):
        return Path(folder)/ "PolyAnalysis"/"All"/f"rtmt_fig_cond{condition}_hemi{hemi}_{task}_{zscore}.pdf"
    
    @staticmethod
    @cache(lambda out, f: f.savefig(out), force_recompute=False)
    def compute(db: DatabaseInstance, out_location: Path, selection): return compute_rtmt_figure(db, "all_rtmt_results", selection)
        

    
pipeline = p