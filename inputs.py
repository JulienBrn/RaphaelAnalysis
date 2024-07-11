import logging, beautifullogger
import sys
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob, precompute
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
logger = logging.getLogger(__name__)


p = Database("coordinates")

@p.register
@Data.from_class()
class AllDatFiles:
    name = "all_dat_files"

    @staticmethod
    def location(folder):
        return  Path(folder)/"PolyAnalysis"/"all_dat_files.txt"
    
    @staticmethod
    @cache(lambda *args: np.savetxt(*args, fmt="%s"), force_recompute=True)
    def compute(db: DatabaseInstance, out_location: Path, selection):
        folder=Path(selection["folder"]) / "Poly_Data"
        return np.array([str(f.relative_to(folder)) for f in folder.glob("**/*.dat")])

    @precompute("all_dat_files")
    def load(db: DatabaseInstance, out_location, selection): return np.loadtxt(out_location, dtype=str, comments=None)

@p.register
@CoordComputer.from_function(vectorized=False, adapt_return=False, coords=["session","subject", "date", "stim_type", "hemi", "condition", "handedness", "forced"], database_arg="db")
def session(db: DatabaseInstance, folder):
    l = db.run_action("load", "all_dat_files", folder=folder, single=True)
    df = pd.DataFrame()
    df["file_stem"] = [Path(f).stem for f in l]
    df["session"] = [str(Path(f).with_suffix("")) for f in l]
    df["subject"] = [str(Path(f).parent.parent.stem) for f in l]
    df["date"] = pd.to_datetime([str(Path(f).parent.stem) for f in l], format="%d%m%Y")

    df = df.loc[df["date"] >= pd.to_datetime("21062024", format="%d%m%Y")]
    df = df.loc[df["date"] != pd.to_datetime("05072024", format="%d%m%Y")]



    df["stim_type"] = np.where(df["file_stem"].str.contains("Conti"), "Conti", 
                               np.where(df["file_stem"].str.contains("Beta"), "Beta", 
                    "Unknown"))
    df["hemi"] = np.where(df["file_stem"].str.contains("LeftHemi"), "Left", 
                               np.where(df["file_stem"].str.contains("RightHemi"), "Right", 
                    "Unknown"))
    df["condition"] = np.where(df["file_stem"].str.contains("CHR"), "Chrimson", 
                               np.where(df["file_stem"].str.contains("CTRL"), "Control", 
                    "Unknown"))
    df["handedness"] = np.where(df["file_stem"].str.contains("Ambidexter"), "Ambi", 
                               np.where(df["file_stem"].str.contains("LeftHanded"), "Left", 
                                        np.where(df["file_stem"].str.contains("RightHanded"), "Right", 
                    "Unknown")))
    df["forced"] = np.where(df["subject"].isin(["#516"]), "Left", "notforced")

    return df.drop(columns=["file_stem"])

@p.register
@CoordComputer.from_function()
def task(stim_type, handedness, forced):
    if forced == "notforced":
        return [f"stim={stim_type}-hand={handedness}-{forced}"]
    else:
        return [f"stim{stim_type}-hand{handedness}-forced{forced}"]

@p.register
@Data.from_class()
class TaskFile:
    name = "task_file"

    @staticmethod
    def location(folder, stim_type, handedness, forced, task):
        ex_folder = Path(folder)/"Poly_Exercices"
        if forced=="notforced":
            file = singleglob(ex_folder, f"*{handedness}*{stim_type}*L2_Nico.xls")
        else:
            file = singleglob(ex_folder, f"*100{forced}*{stim_type}*.xls")
        return file
    

@p.register
@Data.from_class()
class DataFile:
    name = "data_file"

    @staticmethod
    def location(folder, session):
        return Path(folder)/"Poly_Data"/f"{session}.dat"

pipeline = p