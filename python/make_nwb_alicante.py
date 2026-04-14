"""
make_nwb_alicante.py
=====================
Builds a complete NWB file from a MATLAB .mat file produced by the
treadmill dot-tracking task pipeline.  The .mat file must contain the
following top-level variables:

    evts_npx   -- behavioral signals aligned to the Neuropixels clock
                  (continuous boolean arrays + sync edges)
    kilo       -- Kilosort output: good clusters, spike times, peak channels
    histo      -- Probe anatomy: clust2chan_map with region assignment
    sync       -- Synchronisation square-wave info (sampling rates, edges)

LFP data is not yet present in this format and is treated as a future
addition guarded by an ``if exists`` check (Stage 3.5).

Usage
-----
    uv run python make_nwb_alicante.py

Edit SESSION_CONFIG at the top before running.
"""

# %%============================================================
# IMPORTS
# ============================================================
import numpy as np
import scipy.io as sio
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from pynwb import NWBFile, NWBHDF5IO
from pynwb.base import TimeSeries
from pynwb.ecephys import ElectricalSeries, LFP
from pynwb.file import Subject
from pynwb.behavior import SpatialSeries, BehavioralTimeSeries
from hdmf.backends.hdf5.h5_utils import H5DataIO


# %%============================================================
# SESSION CONFIG  --  edit this block for every session
# ============================================================
SESSION_CONFIG = {
    # --- Paths ---
    "mat_path":   Path("/Volumes/T7/NWB_Alicante/20250529_SC19.mat"),
    "output_dir": Path("/Volumes/T7/NWB_Alicante/NWB"),  # filename derived from session_id

    # --- Session metadata ---
    "session_description":    "Treadmill dot-tracking task with Neuropixels recording",
    "session_start_time":     datetime(2025, 5, 29, 0, 0, 0,
                                       tzinfo=ZoneInfo("Europe/Paris")),
    "experimenter":           ["Firstname Lastname"],
    "lab":                    "Your Lab",
    "institution":            "Your Institution",
    "experiment_description": "Treadmill-based visuo-motor task with optogenetic distractor",
    "session_id":             "SC19_20250529",

    # --- Subject ---
    "subject_id":      "SC19",
    "subject_species": "Mus musculus",
    "subject_sex":     "U",    # M / F / U
    "subject_age":     "P0D",  # e.g. P90D

    # --- Probe / recording ---
    "probe_name": "Neuropixels 1.0",

    # --- Histology sidecar CSVs ----------------------------------------
    # Export from MATLAB:
    #   writetable(histo.borders_table_channels, 'borders_table_channels.csv')
    #   writetable(histo.clust2chan_map,          'clust2chan_map.csv')
    "borders_table_channels_csv": Path("/Volumes/T7/NWB_Alicante/borders_table_channels.csv"),
    "clust2chan_csv":              Path("/Volumes/T7/NWB_Alicante/clust2chan_map.csv"),
    # Column name overrides (only needed if auto-detection fails):
    # Run in MATLAB: disp(histo.borders_table_channels.Properties.VariableNames)
    "btc_col_name":  None,  # e.g. "acronym"
    "btc_col_start": None,  # e.g. "lowerBorder"
    "btc_col_end":   None,  # e.g. "upperBorder"

    # --- LFP (future addition -- leave as None if not yet available) ---
    "lfp_bin_path":         None,   # Path to SpikeGLX .lf.bin file
    "lfp_source_rate":      2500.0, # Hz
    "lfp_target_rate":      1000.0, # Hz
    "lfp_compression":      "gzip",
    "lfp_compression_opts": 4,

    # --- Behavioral event CSVs -----------------------------------------
    # Export from MATLAB:
    #   writetable(evts_npx.trial,      'trial.csv')
    #   writetable(evts_npx.area,       'area.csv')
    #   writetable(evts_npx.reward,     'reward.csv')
    #   writetable(evts_npx.distr,      'distr.csv')
    #   writetable(evts_npx.first_move, 'first_move.csv')
    # All CSVs have columns: start_idx, end_idx, start_ts, end_ts
    "trial_csv":      Path("/Volumes/T7/NWB_Alicante/trial.csv"),       # required
    "area_csv":       Path("/Volumes/T7/NWB_Alicante/area.csv"),        # optional, NaN if absent
    "reward_csv":     Path("/Volumes/T7/NWB_Alicante/reward.csv"),      # optional, NaN if absent
    "distr_csv":      Path("/Volumes/T7/NWB_Alicante/distr.csv"),       # optional
    "first_move_csv": Path("/Volumes/T7/NWB_Alicante/first_move.csv"),  # optional, NaN if absent
}


# ============================================================
# HELPERS
# ============================================================
def _mat_load(mat_path: Path) -> dict:
    """Load a MATLAB v5 .mat file with simplify_cells=True."""
    return sio.loadmat(str(mat_path), simplify_cells=True)


def _safe_float(val, default: float = np.nan) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ============================================================
# STAGE 1 -- Load MATLAB file
# ============================================================
print("\n-- Stage 1 : Load MATLAB file")

mat       = _mat_load(SESSION_CONFIG["mat_path"])
evts_npx  = mat["evts_npx"]
kilo      = mat["kilo"]
histo     = mat["histo"]
sync_data = mat["sync"]

spikes_SR = float(sync_data["spikes_SR"])  # AP sampling rate (~30 000 Hz)
lfp_SR    = float(sync_data["SR"])         # LFP / sync line rate (~2 500 Hz)

print(f"  AP sampling rate  : {spikes_SR:.2f} Hz")
print(f"  LFP/sync rate     : {lfp_SR:.2f} Hz")
print(f"  Good clusters     : {len(kilo['good_clusters'])}")
print("Stage 1 done")


# ============================================================
# STAGE 2 -- NWB File Initialisation
# ============================================================
print("\n-- Stage 2 : NWB file initialisation")

nwbfile = NWBFile(
    session_description=SESSION_CONFIG["session_description"],
    identifier=SESSION_CONFIG["session_id"],
    session_start_time=SESSION_CONFIG["session_start_time"],
    experimenter=SESSION_CONFIG["experimenter"],
    lab=SESSION_CONFIG["lab"],
    institution=SESSION_CONFIG["institution"],
    experiment_description=SESSION_CONFIG["experiment_description"],
    session_id=SESSION_CONFIG["session_id"],
    subject=Subject(
        subject_id=SESSION_CONFIG["subject_id"],
        species=SESSION_CONFIG["subject_species"],
        sex=SESSION_CONFIG["subject_sex"],
        age=SESSION_CONFIG["subject_age"],
    ),
)

print(f"  Session : {SESSION_CONFIG['session_id']}")
print(f"  Subject : {SESSION_CONFIG['subject_id']}")
print("Stage 2 done")


# ============================================================
# STAGE 3 -- Probe Anatomy -> ElectrodeTable (all 384 channels)
# ============================================================
print("\n-- Stage 3 : Probe anatomy -> ElectrodeTable")

import pandas as pd

# ----------------------------------------------------------------
# INDEX CONVENTION
# kilo.peak_chan     : MATLAB 1-based -> subtract 1 for 0-based
# kilo.good_clusters : Phy 0-based cluster IDs, no conversion needed
# kilo.channel_map   : 0-based, 383 entries (channel 191 = reference)
# ----------------------------------------------------------------
good_clusters  = kilo["good_clusters"].astype(int)  # (n_units,) 0-based
peak_chan_1idx = kilo["peak_chan"].astype(int)       # 1-based as in MATLAB
peak_chan_0idx = peak_chan_1idx - 1                  # 0-based

n_units = len(good_clusters)
print(f"  {n_units} good clusters")
print(f"  peak_chan (MATLAB 1-based): {peak_chan_1idx.min()} - {peak_chan_1idx.max()}")
print(f"  peak_chan (0-based):        {peak_chan_0idx.min()} - {peak_chan_0idx.max()}")

# ----------------------------------------------------------------
# PROBE GEOMETRY  --  Neuropixels 1.0
# 384 hardware channels, staggered dual-column layout:
#   channel N -> row = N // 2  (row 0 = tip)
#                col = N %  2  (col 0 = right +11um, col 1 = left -11um)
#   depth_from_tip (um) = row * 20
#
# probe_length_histo = number of electrode SITES inside brain (484).
# NP1.0 has 2 sites per row:
#   insertion_depth_histo = (n_sites / 2) * 20 um = 4840 um
# shrinkage_factor (>1) corrects for histological tissue shrinkage:
#   insertion_depth_true = insertion_depth_histo / shrinkage_factor
# ----------------------------------------------------------------
N_CHANNELS        = 384
NP1_ROW_PITCH_UM  = 20.0   # um between rows (tip-to-base)
NP1_COL_OFFSET_UM = 11.0   # um from shank centreline to electrode pad
REF_CHANNEL       = 191    # hardware reference channel (not a recording site)

n_sites_histo         = int(histo["probe_length_histo"])
shrinkage_factor      = float(histo["shrinkage_factor"])
insertion_depth_histo = (n_sites_histo / 2.0) * NP1_ROW_PITCH_UM
insertion_depth_true  = insertion_depth_histo / shrinkage_factor

print(f"  Probe sites inside brain  : {n_sites_histo}")
print(f"  Insertion depth (histo)   : {insertion_depth_histo:.0f} um")
print(f"  Insertion depth (true)    : {insertion_depth_true:.0f} um  "
      f"(shrinkage factor {shrinkage_factor:.4f})")

all_channels          = np.arange(N_CHANNELS)
row_idx               = all_channels // 2
col_idx               = all_channels %  2
x_um                  = np.where(col_idx == 0, NP1_COL_OFFSET_UM, -NP1_COL_OFFSET_UM)
depth_from_tip_um     = row_idx * NP1_ROW_PITCH_UM
depth_from_surface_um = insertion_depth_true - depth_from_tip_um

# ----------------------------------------------------------------
# REGION NAMES
# histo.borders_table_channels and histo.clust2chan_map are MCOS
# tables opaque to scipy -- export from MATLAB first (see SESSION_CONFIG).
#
# borders_table_channels.csv (preferred -- covers every channel):
#   lowerBorder : deeper boundary, 0-based fractional channel number
#   upperBorder : shallower boundary, 0-based fractional channel number
#   acronym     : brain region string
#
# clust2chan_map.csv (fallback -- good clusters only):
#   Cluster (0-based), Channel (1-based), Name, Color
# ----------------------------------------------------------------
channel_region = np.array(["root"] * N_CHANNELS, dtype=object)
borders_loaded = False
_borders_csv   = SESSION_CONFIG.get("borders_table_channels_csv")

if _borders_csv is not None and Path(_borders_csv).exists():
    btc = pd.read_csv(_borders_csv)
    btc.columns = [c.strip() for c in btc.columns]
    print(f"  borders_table_channels columns: {list(btc.columns)}")

    def _find_col(df, candidates):
        cl = [c.lower() for c in df.columns]
        for cand in candidates:
            if cand.lower() in cl:
                return df.columns[cl.index(cand.lower())]
        return None

    col_acronym = _find_col(btc, SESSION_CONFIG.get("btc_col_name",  None) or
                            ["acronym", "name", "region", "brain_region", "area"])
    col_lower   = _find_col(btc, SESSION_CONFIG.get("btc_col_start", None) or
                            ["lowerBorder", "lower_border", "lowerborder",
                             "startchannel", "start_channel", "start", "ch_start"])
    col_upper   = _find_col(btc, SESSION_CONFIG.get("btc_col_end",   None) or
                            ["upperBorder", "upper_border", "upperborder",
                             "endchannel", "end_channel", "end", "ch_end"])

    if None in (col_acronym, col_lower, col_upper):
        print(f"  WARNING: could not identify required columns.")
        print(f"  Found: {list(btc.columns)}")
        print(f"  Set SESSION_CONFIG['btc_col_name'/'btc_col_start'/'btc_col_end'] and re-run.")
    else:
        btc_sorted = btc.sort_values(col_lower).reset_index(drop=True)
        for _, row in btc_sorted.iterrows():
            ch_s = max(0, min(int(np.floor(float(row[col_lower]))), N_CHANNELS - 1))
            ch_e = max(0, min(int(np.floor(float(row[col_upper]))), N_CHANNELS - 1))
            if ch_s <= ch_e:
                channel_region[ch_s : ch_e + 1] = str(row[col_acronym]).strip()
        borders_loaded = True
        seen, unique_r = set(), []
        for r in channel_region:
            if r not in seen:
                seen.add(r); unique_r.append(r)
        n_ann = int((channel_region != "root").sum())
        print(f"  Loaded {len(btc_sorted)} border rows, {n_ann}/{N_CHANNELS} channels annotated")
        print(f"  Region order tip->base: {unique_r}")
elif _borders_csv is not None:
    print(f"  Warning: borders_table_channels_csv not found at {_borders_csv}")

clust_region = {}
_clust_csv   = SESSION_CONFIG.get("clust2chan_csv")

if _clust_csv is not None and Path(_clust_csv).exists():
    c2c = pd.read_csv(_clust_csv)
    c2c.columns = [c.strip().lower() for c in c2c.columns]
    if {"cluster", "name"}.issubset(c2c.columns):
        clust_region = dict(zip(c2c["cluster"].astype(int),
                                c2c["name"].astype(str).str.strip()))
        if not borders_loaded:
            for cl, ch0 in zip(good_clusters, peak_chan_0idx):
                r = clust_region.get(int(cl))
                if r:
                    channel_region[int(ch0)] = r
        print(f"  Loaded clust2chan_map: {len(clust_region)} clusters, "
              f"regions: {sorted(set(clust_region.values()))}")
    else:
        print(f"  Warning: clust2chan_csv missing 'Cluster'/'Name' columns. "
              f"Found: {list(c2c.columns)}")
elif _clust_csv is not None:
    print(f"  Warning: clust2chan_csv not found at {_clust_csv}")

if not borders_loaded and not clust_region:
    print("  No region CSV found. All channels will be labelled 'root'.")
    print("  Run in MATLAB:")
    print("    writetable(histo.borders_table_channels, 'borders_table_channels.csv')")
    print("    writetable(histo.clust2chan_map,          'clust2chan_map.csv')")

# Per-unit region: prefer explicit clust_region, fall back to channel_region
region_names = [clust_region.get(int(cl), channel_region[int(ch0)])
                for cl, ch0 in zip(good_clusters, peak_chan_0idx)]

# ----------------------------------------------------------------
# NWB DEVICE + ELECTRODE GROUP
# ----------------------------------------------------------------
device = nwbfile.create_device(
    name=SESSION_CONFIG["probe_name"],
    description=(
        "Neuropixels 1.0 silicon probe (IMEC). 384 hardware channels, "
        "staggered dual-column layout, 20 um row pitch, +/-11 um column offset. "
        "Channel 191 is the hardware reference (not a recording site). "
        f"Insertion depth (histology): {insertion_depth_histo:.0f} um; "
        f"shrinkage-corrected: {insertion_depth_true:.0f} um."
    ),
    manufacturer="IMEC",
)

unique_regions = sorted(set(r for r in channel_region if r != "root"))
electrode_group = nwbfile.create_electrode_group(
    name="shank0",
    description="Single shank Neuropixels 1.0 probe",
    location=", ".join(unique_regions) if unique_regions else "root",
    device=device,
)

# ----------------------------------------------------------------
# ELECTRODE TABLE  --  one row per hardware channel (0-383)
# ----------------------------------------------------------------
nwbfile.add_electrode_column("channel_id",
    "Hardware channel index (0-based, 0 = probe tip, 383 = probe base)")
nwbfile.add_electrode_column("is_reference",
    "True if this channel is the hardware reference (channel 191, not recorded)")
nwbfile.add_electrode_column("probe_row",
    "Probe row index (0 = tip). Two channels share each row.")
nwbfile.add_electrode_column("probe_col",
    "Probe column index: 0 = right (+11 um from shank), 1 = left (-11 um)")
nwbfile.add_electrode_column("depth_from_tip_um",
    "Distance from probe tip (um). depth_from_tip = probe_row * 20 um")
nwbfile.add_electrode_column("depth_from_surface_um",
    "Estimated depth from brain surface (um), shrinkage-corrected.")
nwbfile.add_electrode_column("brain_region",
    "Brain region from probe tracking (borders_table_channels CSV preferred, "
    "clust2chan_map CSV fallback).")

# channel_id -> electrode row index  (row == channel since we insert sequentially)
channel_to_electrode_row = {ch: ch for ch in range(N_CHANNELS)}

for ch in all_channels:
    nwbfile.add_electrode(
        x=float(x_um[ch]) * 1e-6,
        y=float(depth_from_surface_um[ch]) * 1e-6,
        z=0.0,
        imp=0.0 if ch == REF_CHANNEL else -1.0,
        location=channel_region[ch],
        filtering="Unknown",
        group=electrode_group,
        channel_id=int(ch),
        is_reference=bool(ch == REF_CHANNEL),
        probe_row=int(row_idx[ch]),
        probe_col=int(col_idx[ch]),
        depth_from_tip_um=float(depth_from_tip_um[ch]),
        depth_from_surface_um=float(depth_from_surface_um[ch]),
        brain_region=channel_region[ch],
    )

n_annotated = int((channel_region != "root").sum())
print(f"  384 electrode rows added (channels 0-383)")
print(f"  Reference channel: {REF_CHANNEL}")
print(f"  Channels with region annotation: {n_annotated} / {N_CHANNELS}")
print(f"  Depth range (from surface): "
      f"{depth_from_surface_um.min():.0f} - {depth_from_surface_um.max():.0f} um")
if unique_regions:
    print(f"  Regions: {unique_regions}")
print("Stage 3 done")


# ============================================================
# STAGE 3.5 -- LFP (future addition)
# ============================================================
print("\n-- Stage 3.5 : LFP (future addition)")

_lfp_bin = SESSION_CONFIG.get("lfp_bin_path")

if _lfp_bin is not None and Path(_lfp_bin).exists():
    import spikeinterface.extractors as se
    from scipy.signal import decimate as _decimate

    fs_raw        = SESSION_CONFIG["lfp_source_rate"]
    fs_lfp        = SESSION_CONFIG["lfp_target_rate"]
    compression   = SESSION_CONFIG["lfp_compression"]
    compress_opts = SESSION_CONFIG["lfp_compression_opts"]
    lfp_bin_path  = Path(_lfp_bin)

    decimate_factor = int(round(fs_raw / fs_lfp))
    actual_fs_lfp   = fs_raw / decimate_factor

    _, available_streams = se.get_neo_streams("spikeglx", lfp_bin_path.parent)
    lf_streams  = [s for s in available_streams if s.endswith(".lf")]
    matched     = [s for s in lf_streams if s.split(".lf")[0] in lfp_bin_path.name]
    stream_name = matched[0] if matched else lf_streams[0]

    recording = se.read_spikeglx(
        folder_path=lfp_bin_path.parent,
        stream_name=stream_name,
        load_sync_channel=False,
    )
    n_ch  = recording.get_num_channels()
    n_smp = recording.get_num_samples(segment_index=0)

    CHUNK_SEC   = 60
    chunk_size  = int(CHUNK_SEC * fs_raw)
    n_out       = int(np.ceil(n_smp / decimate_factor))
    lfp_traces  = np.empty((n_out, n_ch), dtype=np.float32)
    lfp_written = 0

    for ci in range(int(np.ceil(n_smp / chunk_size))):
        start = ci * chunk_size
        end   = min(start + chunk_size, n_smp)
        raw   = recording.get_traces(start_frame=start, end_frame=end, segment_index=0)
        dec   = _decimate(raw, q=decimate_factor, ftype="iir",
                          axis=0, zero_phase=True).astype(np.float32)
        lfp_traces[lfp_written : lfp_written + dec.shape[0]] = dec
        lfp_written += dec.shape[0]

    lfp_traces = lfp_traces[:lfp_written]
    lfp_electrode_region = nwbfile.create_electrode_table_region(
        region=list(range(N_CHANNELS)),
        description="All 384 Neuropixels 1.0 hardware channels",
    )
    compressed_lfp = H5DataIO(
        data=lfp_traces,
        compression=compression,
        compression_opts=compress_opts,
        chunks=(int(actual_fs_lfp), n_ch),
        shuffle=True,
    )
    lfp_es = ElectricalSeries(
        name="LFP",
        description=(
            f"LFP decimated from {fs_raw:.0f} Hz to {actual_fs_lfp:.0f} Hz "
            f"(scipy.signal.decimate zero-phase IIR). "
            f"Compression: {compression} level {compress_opts} + byte shuffle. "
            f"Shape: (time, channel). Units: uV."
        ),
        data=compressed_lfp,
        electrodes=lfp_electrode_region,
        starting_time=0.0,
        rate=float(actual_fs_lfp),
        conversion=1e-6,
    )
    ecephys_mod = nwbfile.create_processing_module(
        name="ecephys",
        description="Extracellular electrophysiology processed data",
    )
    ecephys_mod.add(LFP(electrical_series=lfp_es))
    print(f"  LFP stored: {lfp_traces.shape} at {actual_fs_lfp:.0f} Hz")
else:
    print("  LFP bin file not found / not configured -- skipped")
    print("  To add LFP: set 'lfp_bin_path' in SESSION_CONFIG to a SpikeGLX .lf.bin file")

print("Stage 3.5 done")


# ============================================================
# STAGE 4 -- Kilosort -> Units Table
# ============================================================
print("\n-- Stage 4 : Kilosort -> Units table")

# spikes_per_cluster: dict {"cluster_N": np.ndarray of spike sample indices}
# Sample indices are in AP clock samples (spikes_SR ~ 30 000 Hz).
# Cluster IDs in the dict keys are 0-based Phy IDs, matching good_clusters.
spikes_per_cluster = kilo["spikes_per_cluster"]

nwbfile.add_unit_column("cluster_id",
    "Phy / Kilosort cluster ID (0-based)")
nwbfile.add_unit_column("peak_channel_id_matlab",
    "Peak channel index (1-based, as stored in MATLAB kilo.peak_chan)")
nwbfile.add_unit_column("depth_from_tip_um",
    "Depth of peak channel from probe tip (um)")
nwbfile.add_unit_column("depth_from_surface_um",
    "Estimated depth of peak channel from brain surface (um), shrinkage-corrected")
nwbfile.add_unit_column("brain_region",
    "Brain region of peak channel from probe tracking")

total_spikes = 0
skipped      = 0

unit_depth_from_tip_um     = depth_from_tip_um[peak_chan_0idx]
unit_depth_from_surface_um = depth_from_surface_um[peak_chan_0idx]

for i, (cl, ch0, ch1, dft, dfs, region) in enumerate(zip(
        good_clusters, peak_chan_0idx, peak_chan_1idx,
        unit_depth_from_tip_um, unit_depth_from_surface_um, region_names)):

    cluster_key = f"cluster_{cl}"
    if cluster_key not in spikes_per_cluster:
        print(f"  Warning: {cluster_key} not in spikes_per_cluster -- skipped")
        skipped += 1
        continue

    spike_samples = spikes_per_cluster[cluster_key].astype(np.int64)
    spike_times_s = spike_samples.astype(np.float64) / spikes_SR
    total_spikes += len(spike_samples)

    nwbfile.add_unit(
        spike_times            = spike_times_s,
        electrodes             = [channel_to_electrode_row[int(ch0)]],
        cluster_id             = int(cl),
        peak_channel_id_matlab = int(ch1),
        depth_from_tip_um      = float(dft),
        depth_from_surface_um  = float(dfs),
        brain_region           = region,
    )

n_added = n_units - skipped
print(f"  {n_added} units added, {total_spikes:,} total spikes")
if skipped:
    print(f"  {skipped} clusters skipped (not found in spikes_per_cluster)")
print(f"  Peak channel range (0-based): {peak_chan_0idx.min()} - {peak_chan_0idx.max()}")
print(f"  Depth range: "
      f"{unit_depth_from_tip_um.min():.0f} - {unit_depth_from_tip_um.max():.0f} um from tip")
if unique_regions:
    region_counts = {}
    for r in region_names:
        region_counts[r] = region_counts.get(r, 0) + 1
    print(f"  Units per region: { {k: v for k, v in sorted(region_counts.items())} }")
print("Stage 4 done")


# ============================================================
# STAGE 5 -- Behavioral signals -> Trials + TimeSeries
# ============================================================
print("\n-- Stage 5 : Behavioral signals -> Trials table + TimeSeries")

# ----------------------------------------------------------------
# EVENT TABLES
# evts_npx.trial, .area, .reward, .distr, .first_move are MATLAB
# MCOS timetables that scipy cannot decode. Export once from MATLAB:
#
#   writetable(evts_npx.trial,      'trial.csv')
#   writetable(evts_npx.area,       'area.csv')
#   writetable(evts_npx.reward,     'reward.csv')
#   writetable(evts_npx.distr,      'distr.csv')
#   writetable(evts_npx.first_move, 'first_move.csv')
#
# All CSVs have columns: start_idx, end_idx, start_ts, end_ts
# We use:
#   col 3 (start_ts) for all event onset times
#   col 4 (end_ts)   for trial stop time only (from trial.csv)
# ang_speed, dir are read directly from the mat file.
# ----------------------------------------------------------------

import pandas as pd

def _load_event_csv(path, label):
    """Load an event CSV. Returns DataFrame with lowercase columns, or None."""
    if path is None:
        print(f"  {label}: not configured (set SESSION_CONFIG['{label}_csv'])")
        return None
    p = Path(path)
    if not p.exists():
        print(f"  {label}: file not found at {p}")
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"  {label}: {len(df)} rows")
    return df

df_trial      = _load_event_csv(SESSION_CONFIG.get("trial_csv"),      "trial")
df_area       = _load_event_csv(SESSION_CONFIG.get("area_csv"),       "area")
df_reward     = _load_event_csv(SESSION_CONFIG.get("reward_csv"),     "reward")
df_distr      = _load_event_csv(SESSION_CONFIG.get("distr_csv"),      "distr")
df_first_move = _load_event_csv(SESSION_CONFIG.get("first_move_csv"), "first_move")

if df_trial is None:
    raise RuntimeError(
        "trial_csv is required. Export from MATLAB:\n"
        "  writetable(evts_npx.trial, 'trial.csv')\n"
        "then set SESSION_CONFIG['trial_csv']."
    )

# col 3 = start_ts, col 4 = end_ts (seconds, NPX clock)
trial_start_ts = df_trial["start_ts"].values.astype(np.float64)
trial_end_ts   = df_trial["end_ts"].values.astype(np.float64)
n_trials       = len(trial_start_ts)
print(f"  {n_trials} trials  ({trial_start_ts[0]:.2f} s - {trial_end_ts[-1]:.2f} s)")

# col 3 = start_ts only for all other events
area_ts  = df_area["start_ts"].values.astype(np.float64)       if df_area       is not None else None
rew_ts   = df_reward["start_ts"].values.astype(np.float64)     if df_reward     is not None else None
fm_ts    = df_first_move["start_ts"].values.astype(np.float64) if df_first_move is not None else None
distr_ts = df_distr["start_ts"].values.astype(np.float64)      if df_distr      is not None else None

# Treadmill signals (read directly from mat file)
ts_a_npx       = evts_npx["ts_a_as_npx"].astype(np.float64)  # (N_a, 2)
ts_a_seconds   = ts_a_npx[:, 1]                               # NPX clock (s)
ang_speed      = evts_npx["ang_speed"].astype(np.float32)
direction      = evts_npx["dir"].astype(np.int8)

def _first_onset_in_trial(event_ts, t_start, t_stop):
    """Return start_ts of first event inside [t_start, t_stop], or np.nan."""
    if event_ts is None:
        return np.nan
    hits = event_ts[(event_ts >= t_start) & (event_ts <= t_stop)]
    return float(hits[0]) if len(hits) > 0 else np.nan

nwbfile.add_trial_column("trial_number",
    "Trial number (1-indexed)")
nwbfile.add_trial_column("rewarded",
    "Animal reached the goal area and received a reward (bool)")
nwbfile.add_trial_column("distractor",
    "A distractor dot was shown during this trial (bool)")
nwbfile.add_trial_column("area_entry_time",
    "Time of first goal-area entry (s, NPX clock); NaN if animal never entered")
nwbfile.add_trial_column("reward_time",
    "Time of reward onset (s, NPX clock); NaN if trial not rewarded")
nwbfile.add_trial_column("first_move_time",
    "Time of first movement onset (s, NPX clock); NaN if no movement detected")
nwbfile.add_trial_column("distractor_time",
    "Time of distractor dot appearance (s, NPX clock); NaN if no distractor this trial")

n_rewarded = 0
for i in range(n_trials):
    t_start = float(trial_start_ts[i])
    t_stop  = float(trial_end_ts[i])

    area_on  = _first_onset_in_trial(area_ts,  t_start, t_stop)
    rew_on   = _first_onset_in_trial(rew_ts,   t_start, t_stop)
    fm_on    = _first_onset_in_trial(fm_ts,    t_start, t_stop)
    distr_on = _first_onset_in_trial(distr_ts, t_start, t_stop)

    rewarded   = not np.isnan(rew_on)
    has_distr  = not np.isnan(distr_on)
    if rewarded:
        n_rewarded += 1

    nwbfile.add_trial(
        start_time      = t_start,
        stop_time       = t_stop,
        trial_number    = i + 1,
        rewarded        = rewarded,
        distractor      = has_distr,
        area_entry_time = area_on,
        reward_time     = rew_on,
        first_move_time = fm_on,
        distractor_time = distr_on,
    )

print(f"  {n_trials} trials, {n_rewarded} rewarded "
      f"({100 * n_rewarded / max(n_trials, 1):.1f}%)")
if area_ts  is None: print("  Warning: area_csv not set -- area_entry_time will be NaN")
if rew_ts   is None: print("  Warning: reward_csv not set -- reward_time will be NaN")
if fm_ts    is None: print("  Warning: first_move_csv not set -- first_move_time will be NaN")
if distr_ts is None: print("  Warning: distr_csv not set -- distractor_time will be NaN")

# Treadmill TimeSeries -- all in one BehavioralTimeSeries to avoid duplicate-name error
behavior_module = nwbfile.create_processing_module(
    name="behavior",
    description="Treadmill behavioral data aligned to Neuropixels clock",
)

ts_p      = evts_npx["ts_p"].astype(np.float64)
bool_data = np.stack(
    [evts_npx["in_trial"].astype(np.uint8),
     evts_npx["in_area"].astype(np.uint8),
     evts_npx["reward_given"].astype(np.uint8)],
    axis=1,
)  # (N_p, 3)

behavior_module.add(
    BehavioralTimeSeries(
        name="treadmill",
        time_series=[
            SpatialSeries(
                name="angular_speed",
                description=(
                    "Treadmill angular speed (arbitrary units, 100 Hz Arduino). "
                    "Timestamps re-expressed on the Neuropixels clock via linear "
                    "interpolation of the synchronisation square wave."
                ),
                data=ang_speed,
                timestamps=ts_a_seconds,
                reference_frame="treadmill_ball",
                unit="au",
            ),
            SpatialSeries(
                name="heading_direction",
                description=(
                    "Treadmill heading direction: -1=left, 0=still, +1=right. "
                    "Sampled at 100 Hz (Arduino), timestamps on Neuropixels clock."
                ),
                data=direction,
                timestamps=ts_a_seconds,
                reference_frame="treadmill_ball",
                unit="category",
            ),
            TimeSeries(
                name="state_signals",
                description=(
                    "Three boolean state signals sampled at ~134 Hz (Python clock), "
                    "re-clocked to Neuropixels timeline. "
                    "Columns: [in_trial, in_area, reward_given]."
                ),
                data=bool_data,
                timestamps=ts_p,
                unit="binary",
            ),
        ]
    )
)

print("  Treadmill TimeSeries stored (BehavioralTimeSeries 'treadmill'):")
print("    angular_speed     (100 Hz, NPX clock)")
print("    heading_direction (100 Hz, NPX clock)")
print("    state_signals     (~134 Hz, NPX clock) [in_trial, in_area, reward_given]")
print("Stage 5 done")


# ============================================================
# STAGE 6 -- Synchronisation edges
# ============================================================
print("\n-- Stage 6 : Synchronisation edges")

edges_npx = sync_data["edges"].astype(np.float64)
a_edges   = evts_npx["a_edges"].astype(np.float64)
p_edges   = evts_npx["p_edges"].astype(np.float64)

sync_module = nwbfile.create_processing_module(
    name="sync",
    description="Synchronisation square-wave edge times used to align recording systems",
)

sync_module.add(TimeSeries(
    name="npx_edges",
    description=(
        "Times (s) of synchronisation square-wave transitions recorded by the "
        f"Neuropixels system (LFP file, {lfp_SR:.4f} Hz)."
    ),
    data=np.ones(len(edges_npx), dtype=np.uint8),
    timestamps=edges_npx,
    unit="V",
))
sync_module.add(TimeSeries(
    name="arduino_edges",
    description="Times (s) of synchronisation transitions recorded by the Arduino (100 Hz).",
    data=np.ones(len(a_edges), dtype=np.uint8),
    timestamps=a_edges,
    unit="V",
))
sync_module.add(TimeSeries(
    name="python_edges",
    description="Times (s) of synchronisation transitions recorded by the Python script.",
    data=np.ones(len(p_edges), dtype=np.uint8),
    timestamps=p_edges,
    unit="V",
))

print(f"  {len(edges_npx)} edge pairs stored (npx / arduino / python)")
print("Stage 6 done")


# ============================================================
# STAGE 7 -- Write & Validate
# ============================================================
print("\n-- Stage 7 : Write and validate")

output_dir  = SESSION_CONFIG["output_dir"]
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{SESSION_CONFIG['session_id']}.nwb"

with NWBHDF5IO(str(output_path), mode="w") as io:
    io.write(nwbfile)

file_size_mb = output_path.stat().st_size / (1024 ** 2)
print(f"  Written to : {output_path}  ({file_size_mb:.1f} MB)")

with NWBHDF5IO(str(output_path), mode="r") as io:
    nwb_check = io.read()
    n_el    = len(nwb_check.electrodes)
    n_u     = len(nwb_check.units)
    n_s     = sum(len(nwb_check.units["spike_times"][i]) for i in range(n_u))
    n_t     = len(nwb_check.trials)
    n_rew   = int(np.sum(nwb_check.trials["rewarded"].data[:]))
    has_lfp = (
        "ecephys" in nwb_check.processing
        and "LFP" in nwb_check.processing["ecephys"]
    )

print(f"\n  Validation report")
print(f"    Electrodes  : {n_el}")
print(f"    Units       : {n_u} units, {n_s:,} spikes")
print(f"    Trials      : {n_t} ({n_rew} rewarded)")
print(f"    LFP         : {'yes' if has_lfp else 'not yet included (future addition)'}")
print(f"\nAll done -- NWB file ready at {output_path}")
# %%