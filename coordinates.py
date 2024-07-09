import logging, beautifullogger
import sys
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob
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
        return  Path(folder)/"all_dat_files.txt"
    
    @staticmethod
    @cache(lambda *args: np.savetxt(*args, fmt="%s"), force_recompute=True)
    def compute(db: DatabaseInstance, out_location: Path, selection):
        folder=Path(selection["folder"]) / "Poly_Data"
        return np.array([str(f.relative_to(folder)) for f in folder.glob("**/*.dat")])

    @staticmethod
    def load(db: DatabaseInstance, out_location, selection):
        loc = db.compute_single("all_dat_files", selection)
        return np.loadtxt(loc, dtype=str, comments=None)
    
    @staticmethod
    def show(db, out_location, selection):
        v = db.run_action("load", "all_dat_files", selection, single=True)
        print(v)

@p.register
@CoordComputer.from_function(vectorized=False, adapt_return=False, coords=["session","subject", "date", "stim_type", "hemi", "condition", "handedness", "forced"], database_arg="db")
def session(db, folder):
    l = db.run_action("load", "all_dat_files", folder=folder, single=True)
    # l = [f for f in l if "BAGOSMOV" in f]
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
@Data.from_class()
class Task:
    name = "task_file"

    @staticmethod
    def location(folder, stim_type, handedness, forced):
        if pd.isna(folder):
            raise Exception("folder is nan")
        if forced=="notforced":
            file = singleglob(Path(folder)/"Poly_Exercices", f"*{handedness}*{stim_type}*L2_Nico.xls")
        else:
            file = singleglob(Path(folder)/"Poly_Exercices", f"*100{forced}*{stim_type}*.xls")
        return file
    
    @staticmethod
    def load(db: DatabaseInstance, out_location, selection):
        import poly_graph
        return poly_graph.load_task_file(out_location)

@p.register
@Data.from_class()
class EventData:
    name = "event_data"

    @staticmethod
    def location(folder, session):
        return Path(folder)/"Poly_Data"/f"{session}.dat"
        
    
    @staticmethod
    def load(db: DatabaseInstance, out_location, selection):
        import poly_graph
        return poly_graph.load_poly_data_file(out_location)

    @staticmethod
    def make_tsv(db: DatabaseInstance, out_location, selection):
        df = db.run_action("load", "event_data", selection, single=True)
        out = Path(selection["folder"])/ "PolyAnalysis"/"Sessions"/f"{selection['session']}_graph.tsv"
        out.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(out, sep="\t", index=False)
        return out
pipeline = p