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

def tsvload(db, out_location, selection):
   return pd.read_csv(out_location, sep="\t", index_col=None)

def mplpdfsave(out, fig):
    fig.savefig(out)
    plt.close(fig)

@p.register
@Data.from_class()
class AggTrialData:
    name = "agg_trial_data"

    @staticmethod
    def location(folder):
        return Path(folder)/ "PolyAnalysis"/"All"/"agg_trial_data.tsv"
    
    @staticmethod
    @cache(lambda out, df: df.to_csv(out, sep="\t", index=False), force_recompute=False)
    def compute(db: DatabaseInstance, out_location, selection):
        dfs = db.run_action("load", "trial_data", selection)
        all = pd.concat([row["load"].assign(**{k:v for k,v in row.items() if not k=="load"}) for _, row in dfs.iterrows()], ignore_index=True)
        all["rel_cue"] = np.where(~all["cue"].isin(["Right", "Left"]), "Unknown",
                         np.where(all["cue"] == all["hemi"], "ipsi", "contra"))
        all["rel_paw"] = np.where(~all["paw"].isin(["Right", "Left"]), "Unknown",
                         np.where(all["paw"] == all["hemi"], "ipsi", "contra"))
        all["mv_type"] = np.where(~all["cue"].isin(["Right", "Left"]), "Unknown",
                    np.where(all["cue"] == all["paw"], "straight", "crossed"))
        return all

    @precompute("agg_trial_data")
    def load(db: DatabaseInstance, out_location, selection): return pd.read_csv(out_location, sep="\t", index_col=None)

@cache(force_recompute=True)
def compute_motor_data(db, out_location, selection):
    all: pd.DataFrame = db.run_action("load", "agg_trial_data", selection, single=True)
    for k, v in selection.items():
        if k in all.columns:
            all = all.loc[all[k]==v]
    all = all.reset_index(drop=True)
    return all

subject_palette = {
    # "#516": "green",
    "#517": "red",
    "#525": "purple",
    "#527": "blue",
    "#531": "orange",
}

markers = dict(ipsi="+", contra="X", Left="o", Right="o")
linestyles = dict(ipsi=(0, (1,1)), contra="-", Left="-", Right="-")
markers = dict(mv_type=dict(straight="+", crossed="X"), hemi=dict(Left="o", Right="o"))
linestyles = dict(mv_type=dict(straight="-", crossed=(0, (1,1))), hemi=dict(Left="-", Right="-"))

def violin_with_pvalues(data: pd.DataFrame, x, y, ax: plt.Axes, order=None, sub_points=None, **kwargs): 
    grps = data.groupby(x)
    dfs = {k:v for k,v in grps}
    if order is None:
        order=[k for k in dfs.keys()]
    sns.violinplot(data, x=x, y=y, ax=ax, order=order, color = "lightblue", linewidth=1, **kwargs)
    c = 0

    scatter_df: pd.DataFrame = data.groupby([x]+sub_points)[y].mean().reset_index()
    scatter_df[x] = scatter_df[x].map({k: order.index(k) for k in order}) + (scatter_df.groupby(sub_points).ngroup() - scatter_df.groupby(sub_points).ngroups/2)*0.05
    if not sub_points is None:
        if len(sub_points)==1:
            sns.scatterplot(data=scatter_df, x=x, y=y, hue=sub_points[0], ax=ax, zorder=2, palette=subject_palette)
        if len(sub_points)==2:
            for v, sdf in scatter_df.groupby(sub_points[1]):
                sns.scatterplot(data=sdf, x=x, y=y, hue=sub_points[0], ax=ax, zorder=2, palette=subject_palette, legend=False, marker=markers[sub_points[1]][v])

    for i in range(len(dfs)):
        x1 = i
        df1 = dfs[order[i]]
        dfmean = df1[y].mean()
        ax.scatter([x1], [dfmean], color="black", s=22, marker="_", zorder=2)

        for j in range(i+1, len(dfs)):
            x2 = j
            df2 = dfs[order[j]]
            import scipy.stats
            p = scipy.stats.kstest(df1[y].to_numpy(), df2[y].to_numpy()).pvalue
            # p = scipy.stats.ttest_ind(df1[y].to_numpy(), df2[y].to_numpy(), nan_policy="omit").pvalue
            rel_y_pos = 0.4*(j-i)/(len(dfs)-1) +0.5
            ax.text((x1+x2)/2, rel_y_pos*ax.get_ylim()[1] + (1-rel_y_pos)*ax.get_ylim()[0], f"ks_pvalue<{np.ceil(p*10**3)/10**3}", ha="center", va="bottom")
            ax.hlines([rel_y_pos*ax.get_ylim()[1] + (1-rel_y_pos)*ax.get_ylim()[0]], [x1+0.05], [x2-0.05])
            c+=1

def plot_motor_fig(data1, data2, mt_col, sub, full_legend):
    f, axs = plt.subplots(2, 2, figsize=(16, 9))
    data1 = data1.copy()
    data2=data2.copy()
    axs[0,0].sharey(axs[0, 1])
    for ax  in (axs[0, 0], axs[0, 1]):
        if "zscore" in mt_col:
            ax.set_ylim([-2, 7])
        else:
            ax.set_ylim([0, 3])
    axs[1,0].sharey(axs[1, 1])
    for ax  in (axs[1, 0], axs[1, 1]):
        ax.set_ylim([0, 1])
    
    for i, d in enumerate([data1, data2]):
        d =d.loc[(d["stim"]!="NoStimYet")]
        if len(d.index) == 0:
            continue
        stim_order= [c for c in ["NoStim" , "Beta", "Conti"] if c in d["stim"].values]
        violin_with_pvalues(d.loc[(d["stim"]!="NoStimYet") & (d["trial_type"]=="good")], x="stim", y=mt_col, sub_points=["subject", sub],
                        ax=axs[0, i], order=stim_order, common_norm=False, cut=0)
        trial_type_order_map = {k: i for i, k in enumerate(t for t in ["Anticip", "RTCutoff", "BadRT", "MTCutOff", "BadMT", "BadLP", "good"] if t in d["trial_type"].values)}
        h_data = d.assign(x_order=d["trial_type"].map(trial_type_order_map)).sort_values("x_order")
        sizes = h_data.groupby("stim").size()
        stim_rename_dict = {k: f"{k}, n={v}" for k,v in sizes.items()}
        h_data["stim, count"] = h_data["stim"].map(stim_rename_dict)

        sns.histplot(data = h_data, x="trial_type", hue="stim, count", ax=axs[1, i], stat="probability",
                      multiple="dodge", common_norm=False, hue_order=[stim_rename_dict[v] for v in stim_order]
                      , shrink=.8, palette="pastel")
        
        sub_size = h_data.groupby(["stim", "subject", sub]).size()
        # sub_size = sub_size.to_frame("sub_size").join(sizes.rename("all_size"))["sub_size"].fillna(0)
        s_data = (h_data.groupby(["stim", "trial_type", "subject", sub]).size()/sub_size).rename("sub_prop").reset_index()
        s_data["x"] = s_data["trial_type"].map(trial_type_order_map) + (s_data["stim"].map({k: i for i, k in enumerate(stim_order)}) -(len(stim_order) -1)/2)/(len(stim_order)/0.8)
        for v, sdf in s_data.groupby(sub):
            if v in linestyles[sub]:
                sns.scatterplot(sdf, x="x", y="sub_prop", hue="subject", ax=axs[1, i], marker="_", linewidths=2, legend=False, palette=subject_palette, zorder=2, linestyle=linestyles[sub][v]
                                )
        import xarray as xr, scipy.stats
        mp_stats: xr.DataArray = d.groupby(["trial_type", "stim"]).size().to_xarray()
        p_values = {}
        for k in range(mp_stats["stim"].size):
            for j in range(k+1, mp_stats["stim"].size):
                table = mp_stats.isel(stim=[k, j])
                p_values[mp_stats["stim"].isel(stim=k).item(), mp_stats["stim"].isel(stim=j).item()] = scipy.stats.contingency.chi2_contingency(table.to_numpy()).pvalue
        # print(p_values)
        axs[1, i].text(0.1, 0.5, "\n".join([f"{k}={np.ceil(v*10**3)/10**3}" for k,v in p_values.items()]))
        # exit()
        # for container in axs[1, i].containers:
        #     axs[1, i].bar_label(container)
        # print([c for cont in axs[1, i].containers for c in cont])
        # exit()
        # tmp = d.groupby(["stim", "trial_type"]).apply(lambda d: len(d.index)).to_frame("count")
        # tmp = tmp["count"].unstack("stim")
        # tmp = tmp[[c for c in ["trial_type", "NoStimYet", "NoStim" , "Beta", "Conti"] if c in tmp.columns]]
        # tmp=tmp.rename(columns={col: f'{col}, n={int(tmp[col].sum())}' for col in tmp.columns})
        # tmp = tmp/tmp.sum(axis=0)
        # tmp: pd.DataFrame = tmp.reset_index()
        # tmp["order"]=tmp["trial_type"].map({"Anticip": 0, "RTCutoff":1, "BadRT":2, "MTCutOff":3, "BadMT": 4, "BadLP": 5, "good": 6})
        # tmp=tmp.sort_values("order")
        # tmp=tmp.drop(columns=["order"])
        # tmp.plot.bar(x="trial_type", ax=axs[1, i])
    axs[0, 0].set_ylabel(f'{mt_col} for good trials')
    axs[0, 1].set_ylabel(f'{mt_col} for good trials')
    axs[1, 0].set_ylabel(f'percentage of trials (sum for a given stimulation=1)')
    axs[1, 1].set_ylabel(f'percentage of trials (sum for a given stimulation=1)')

    fig_handles=[]
    for subject, color in subject_palette.items():
        import matplotlib.lines as mlines
        fig_handles.append(mlines.Line2D([], [], color=color, label=subject))
    if full_legend:
        for dir, type in markers[sub].items():
            import matplotlib.lines as mlines
            fig_handles.append(mlines.Line2D([], [], color="black", label=f"{sub}={dir}", linestyle='None', marker=type))
        for dir, type in linestyles[sub].items():
            if dir in["Left", "Right"]: continue
            import matplotlib.lines as mlines
            fig_handles.append(mlines.Line2D([], [], color="black", label=f"{sub}={dir}", linestyle=type))
    f.legend(handles=fig_handles)
    return f,axs

def plot_basicmotor_fig(data_name, mt_col, sub):
    @cache(mplpdfsave, force_recompute=True)
    def inner(db, out_location, selection):
        data: pd.DataFrame = db.run_action("load", data_name, selection, single=True)
        data1 = data.loc[data["condition"] == "Control"]
        data2 = data.loc[data["condition"] == "Chrimson"]
        f ,axs= plot_motor_fig(data1, data2, mt_col, sub, full_legend=False)
        axs[0, 0].set_title(f'Control')
        axs[0, 1].set_title(f'Chrimson')
        title = "Motor performance with different stimulations. Control vs Chrimson "
        plt.suptitle(title + "\n" + ",".join([f'{k}={v if not isinstance(v, datetime.datetime) else format_date(v, "dd MMM yyyy", locale="en")}' for k,v in selection.items() if not k in ["session", "folder", "task"]]), fontsize="small")
        # plt.show()
        return f
    return inner

def plot_cuemotor_fig(data_name, mt_col, sub):
    @cache(mplpdfsave, force_recompute=True)
    def inner(db, out_location, selection):
        data: pd.DataFrame = db.run_action("load", data_name, selection, single=True)
        data = data.loc[(data["condition"] == "Chrimson") & (data["stim"] != "Beta")]
        data1 = data.loc[data[f"rel_cue"] == "ipsi"]
        data2 = data.loc[data[f"rel_cue"] == "contra"]
        f,axs = plot_motor_fig(data1, data2, mt_col, sub, full_legend=True)
        axs[0, 0].set_title(f'Cue ipsi')
        axs[0, 1].set_title(f'Cue contra')
        title = "Motor performance with different stimulations in Chrimson hemispheres, when cue is ipsi or contra to the stimulated hemisphere."
        plt.suptitle(title + "\n" + ",".join([f'{k}={v if not isinstance(v, datetime.datetime) else format_date(v, "dd MMM yyyy", locale="en")}' for k,v in selection.items() if not k in ["session", "folder", "task"]]), fontsize="small")
        return f
    return inner

def plot_pawmotor_fig(data_name, mt_col, sub):
    @cache(mplpdfsave, force_recompute=True)
    def inner(db, out_location, selection):
        data: pd.DataFrame = db.run_action("load", data_name, selection, single=True)
        data = data.loc[(data["condition"] == "Chrimson") & (data["stim"] != "Beta")]
        data1 = data.loc[data[f"rel_paw"] == "ipsi"]
        data2 = data.loc[data[f"rel_paw"] == "contra"] 
        f,axs = plot_motor_fig(data1, data2, mt_col, sub, full_legend=False)
        axs[0, 0].set_title(f'Paw ipsi')
        axs[0, 1].set_title(f'Paw contra')
        title = "Motor performance with different stimulations in Chrimson hemispheres, when demanded paw to press lever is ipsi or contra to the stimulated hemisphere."
        plt.suptitle(title + "\n" + ",".join([f'{k}={v if not isinstance(v, datetime.datetime) else format_date(v, "dd MMM yyyy", locale="en")}' for k,v in selection.items() if not k in ["session", "folder", "task"]]), fontsize="small")
        return f
    return inner


functions = pd.DataFrame([
    ["motor_data", "tsv", dict(compute=lambda n: compute_motor_data, load=lambda n: precompute(n)(tsvload)), []],
    ["basicmotor_fig", "pdf", dict(compute=lambda n: plot_basicmotor_fig(n.replace("_fig", "_data").replace("basicmotor", "motor"), "mt", "hemi")), []],
    ["cuemotor_fig", "pdf", dict(compute=lambda n: plot_cuemotor_fig(n.replace("_fig", "_data").replace("cuemotor", "motor"), "mt", "mv_type")), []],
    ["pawmotor_fig", "pdf", dict(compute=lambda n: plot_pawmotor_fig(n.replace("_fig", "_data").replace("pawmotor", "motor"), "mt", "hemi")), []],
    ["zscored_basicmotor_fig", "pdf", dict(compute=lambda n: plot_basicmotor_fig(n.replace("_fig", "_data").replace("zscored_basicmotor", "motor"), "mt_zscored", "hemi")), []],
    ["zscored_cuemotor_fig", "pdf", dict(compute=lambda n: plot_cuemotor_fig(n.replace("_fig", "_data").replace("zscored_cuemotor", "motor"), "mt_zscored", "mv_type")), []],
    ["zscored_pawmotor_fig", "pdf", dict(compute=lambda n: plot_pawmotor_fig(n.replace("_fig", "_data").replace("zscored_pawmotor", "motor"), "mt_zscored", "hemi")), []],
    ], columns=["name", "ext", "actions", "additional_deps"]
)

groupings = pd.DataFrame([
    ["session", ["session"], "{{folder}}/PolyAnalysis/Sessions/{{session}}/Results/{name}{fdeps}.{ext}"],
    ["subject", ["subject"], "{{folder}}/PolyAnalysis/Subjects/{{subject}}/{name}{fdeps}.{ext}"],
    ["all", [], "{{folder}}/PolyAnalysis/All/{name}{fdeps}.{ext}"],
    ], columns=["name", "grouping", "loc"]
)


def make_computer(name, params, location, actions):
    keys = [p[1] for p in string.Formatter().parse(location)]
    res =  Data(name, dependencies=set(params), get_location=lambda d: Path(location.format_map({k:v for k,v in d.items() if k in keys})), actions=actions)
    return res

for _, f in functions.iterrows():
    for _, grp in groupings.iterrows():
        name = grp["name"]+"_"+f["name"]
        params = ["folder"] + grp["grouping"] + f["additional_deps"]
        deps_str = "" if not f["additional_deps"] else "_" + "_".join(["{"+d+"}" for d in f["additional_deps"]])
        loc = grp["loc"].format(name=f["name"], ext=f["ext"], fdeps = deps_str)
        actions = {k: v(name) for k, v in f["actions"].items()}
        p.declare(make_computer(name, params, loc, actions))



    



pipeline = p