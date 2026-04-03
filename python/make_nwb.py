"""
create_nwb.py
=============
Builds a complete NWB file from:
  - A SpikeInterface SortingAnalyzer folder  (units, waveforms, quality metrics,
                                              curation labels)
  - A behavioral CSV                         (trials table with IMEC sync events)
  - A probe anatomy CSV                      (Allen CCF electrode coordinates)
  - The raw recording via the analyzer       (LFP: low-pass, downsampled to 1 kHz,
                                              gzip-compressed)

Usage
-----
    uv run python create_nwb.py

Edit SESSION_CONFIG at the top before running.
"""

# %%============================================================
# IMPORTS
# ============================================================
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from scipy.signal import decimate

import spikeinterface.core as si
import spikeinterface.extractors as se

from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries, LFP
from pynwb.file import Subject


# ============================================================
# SESSION CONFIG  --  edit this block for every session
# ============================================================
SESSION_CONFIG = {
    # --- Paths ---
    "analyzer_path":     Path("/Volumes/T7/NWB_Joana/sorted/analyzer_output"),
    "behavior_csv":      Path("/Volumes/T7/NWB_Joana/behavior/behavior_paired.csv"),
    "probe_anatomy_csv": Path("/Volumes/T7/NWB_Joana/histology/999770_neuropixels_probe1.csv"),
    "output_nwb":        Path("/Volumes/T7/NWB_Joana/NWB/999770_20251111_probe01.nwb"),

    # --- Session metadata ---
    "session_description":    "Adaptive sensorimotor task with PFC-STR recordings - Beginner session",
    "session_start_time":     datetime(2025, 11, 11, 14, 0, 0,
                                       tzinfo=ZoneInfo("Europe/Paris")),
    "experimenter":           ["Joana Catarino"],
    "lab":                    "Carlen Lab",
    "institution":            "Karolinska Institutet",
    "experiment_description": "Neuropixels recording during an adaptive sensorimotor task",
    "session_id":             "999770_20251111_probe01",

    # --- Subject ---
    "subject_id":      "999770",
    "subject_species": "Mus musculus",
    "subject_sex":     "M",
    "subject_age":     "P60D",

    # --- Probe / recording ---
    "probe_name":    "Neuropixels 1.0",
    "sampling_rate": 30000.0,

    # --- LFP ---
    "lfp_bin_path":        Path("/Volumes/T7/NWB_Joana/lfp/999770_day1_1_g0_t0.imec1.lf.bin"),
    "lfp_source_rate":     2500.0,  # Hz -- actual sampling rate of the LFP recording
    "lfp_target_rate":     1000.0,  # Hz -- target rate after downsampling
    "lfp_lowpass_hz":       400.0,  # Hz -- low-pass cutoff (decimate handles this)
    "lfp_compression":     "gzip",  # HDF5 compression algorithm
    "lfp_compression_opts":     4,  # gzip level (1=fast / 9=smallest)

    # --- Curation JSON files ---
    # None = look in analyzer_path automatically. Set a Path to override.
    "curation_json":   None,
    "passing_qc_json": None,
    "bombcell_json":   None,
    "unitrefine_json": None,
}


# ============================================================
# STAGE 1 -- NWB File Initialisation
# ============================================================
print("\n-- Stage 1 : NWB file initialisation")

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
print("Stage 1 done")


# ============================================================
# STAGE 2 -- Probe Anatomy -> ElectrodeTable
# ============================================================
print("\n-- Stage 2 : Probe anatomy -> ElectrodeTable")

# Each row in the CSV = one recording site at a given depth.
# Neuropixels 1.0 has two electrode pads per site (left column / right column),
# so each site contributes two physical channels.
# NWB requires one row per physical channel -> we expand 192 sites to 384 channels.
probe_df = pd.read_csv(SESSION_CONFIG["probe_anatomy_csv"], index_col=0)
probe_df = probe_df[probe_df["inside_brain"] == True].reset_index(drop=True)
n_sites    = len(probe_df)
n_channels = n_sites * 2
print(f"  {n_sites} sites inside brain -> {n_channels} channels, "
      f"{probe_df['acronym'].nunique()} brain regions")

device = nwbfile.create_device(
    name=SESSION_CONFIG["probe_name"],
    description="Neuropixels 1.0 silicon probe (IMEC)",
    manufacturer="IMEC",
)

electrode_group = nwbfile.create_electrode_group(
    name="shank0",
    description="Single shank Neuropixels probe",
    location=", ".join(probe_df["acronym"].unique().tolist()),
    device=device,
)

# Custom columns -- one value per physical channel
nwbfile.add_electrode_column("channel_id",        "Physical channel index on probe (1-384)")
nwbfile.add_electrode_column("pad_side",          "Electrode pad side at this site: left or right")
nwbfile.add_electrode_column("site_index",        "Recording site index (0-based, one site = two channels)")
nwbfile.add_electrode_column("distance_to_tip",   "Distance from probe tip (um)")
nwbfile.add_electrode_column("depth_um",          "Depth from brain surface (um)")
nwbfile.add_electrode_column("ap_coords_vox",     "Allen CCF AP coordinate (voxels)")
nwbfile.add_electrode_column("dv_coords_vox",     "Allen CCF DV coordinate (voxels)")
nwbfile.add_electrode_column("ml_coords_vox",     "Allen CCF ML coordinate (voxels)")
nwbfile.add_electrode_column("structure_id",      "Allen Brain Atlas structure ID")
nwbfile.add_electrode_column("acronym",           "Allen Brain Atlas region acronym")
nwbfile.add_electrode_column("brain_region",      "Allen Brain Atlas full region name")
nwbfile.add_electrode_column("dist_to_structure", "Distance to nearest structure boundary (um)")

for site_idx, row in probe_df.iterrows():
    try:
        struct_id = int(row["structure_id"])
    except (ValueError, TypeError):
        struct_id = -1

    shared = dict(
        y=float(row["dv_mm"]),
        z=float(row["ap_mm"]),
        imp=-1.0,
        location=str(row["acronym"]),
        filtering="Unknown",
        group=electrode_group,
        site_index=int(site_idx),
        distance_to_tip=float(row["distance_to_tip(um)"]),
        depth_um=float(row["depth(um)"]),
        ap_coords_vox=int(row["ap_coords"]),
        dv_coords_vox=int(row["dv_coords"]),
        ml_coords_vox=int(row["ml_coords"]),
        structure_id=struct_id,
        acronym=str(row["acronym"]),
        brain_region=str(row["name"]),
        dist_to_structure=float(row["distance_to_nearest_structure(um)"]),
    )

    # Left pad -- NWB x = ML coordinate (left pad is slightly more lateral)
    nwbfile.add_electrode(
        x=float(row["ml_mm"]) - 0.016,  # Neuropixels 1.0: pads offset ~16 um in ML
        channel_id=int(row["channel_l"]),
        pad_side="left",
        **shared,
    )

    # Right pad
    nwbfile.add_electrode(
        x=float(row["ml_mm"]) + 0.016,
        channel_id=int(row["channel_r"]),
        pad_side="right",
        **shared,
    )

print(f"  {n_channels} electrodes added ({n_sites} sites x 2 pads)")
print(f"  Regions traversed: {', '.join(probe_df['acronym'].unique())}")
print("Stage 2 done")


# ============================================================
# STAGE 3 -- SpikeInterface Analyzer -> Units Table
# ============================================================
print("\n-- Stage 3 : SpikeInterface -> Units table")


def _load_json_if_exists(path):
    """Return parsed JSON or None if file does not exist."""
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    print(f"  Note: {p.name} not found -- column will be skipped")
    return None


def _resolve_json(key, default_name):
    """Return path from SESSION_CONFIG override or fall back to analyzer folder."""
    override = SESSION_CONFIG.get(key)
    if override is not None:
        return Path(override)
    return SESSION_CONFIG["analyzer_path"] / default_name


analyzer     = si.load_sorting_analyzer(SESSION_CONFIG["analyzer_path"])
sorting      = analyzer.sorting
fs           = sorting.get_sampling_frequency()
all_unit_ids = sorting.unit_ids

# --- Load curation files ---
curation_raw   = _load_json_if_exists(_resolve_json("curation_json",   "curation.json"))
passing_qc_raw = _load_json_if_exists(_resolve_json("passing_qc_json", "passing_qc.json"))
bombcell_raw   = _load_json_if_exists(_resolve_json("bombcell_json",   "bombcell_labels.json"))
unitrefine_raw = _load_json_if_exists(_resolve_json("unitrefine_json", "unitrefine_labels.json"))

# Parse curation.json -> {unit_id: quality_label}
curation_quality = {}
if curation_raw is not None:
    for entry in curation_raw.get("manual_labels", []):
        uid   = int(entry["unit_id"])
        label = entry.get("quality",
                entry.get("labels", {}).get("quality", [None]))[0]
        if label is not None:
            curation_quality[uid] = str(label)

# Parse passing_qc.json -> {unit_id: bool}
passing_qc = {}
if passing_qc_raw is not None:
    passing_qc = {int(k): bool(v) for k, v in passing_qc_raw.items()}

# Parse bombcell_labels.json -> {unit_id: str}
bombcell = {}
if bombcell_raw is not None:
    for k, v in bombcell_raw.get("bombcell_label", {}).items():
        bombcell[int(k)] = str(v).strip("[]' \"")

# Parse unitrefine_labels.json -> prediction + probability
unitrefine_pred = {}
unitrefine_prob = {}
if unitrefine_raw is not None:
    for k, v in unitrefine_raw.get("unitrefine_prediction", {}).items():
        unitrefine_pred[int(k)] = str(v)
    for k, v in unitrefine_raw.get("unitrefine_probability", {}).items():
        unitrefine_prob[int(k)] = float(v)

# --- Filter to good units ---
if curation_quality:
    unit_ids = np.array([
        uid for uid in all_unit_ids
        if curation_quality.get(int(uid), "").lower() == "good"
    ])
    label_counts = {}
    for uid in all_unit_ids:
        lbl = curation_quality.get(int(uid), "unlabelled")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print(f"  curation.json labels: {label_counts}")
    print(f"  Keeping {len(unit_ids)} / {len(all_unit_ids)} units labelled 'good'")
    if len(unit_ids) == 0:
        raise RuntimeError(
            "No units labelled 'good' in curation.json. "
            "Check manual_labels entries."
        )
else:
    print("  Warning: curation.json has no manual_labels -- using all units")
    unit_ids = all_unit_ids

n_units = len(unit_ids)
print(f"  Loaded analyzer: {n_units} good units at {fs} Hz")

# --- Quality metrics ---
metrics_ext = analyzer.get_extension("quality_metrics")
if metrics_ext is None:
    raise RuntimeError("quality_metrics extension not found. Run compute_quality_metrics() first.")
metrics_df = metrics_ext.get_data()

# --- Templates ---
templates_ext = analyzer.get_extension("templates")
if templates_ext is None:
    raise RuntimeError("templates extension not found. Run compute_waveforms() first.")

# get_templates() without operator returns the default (usually "average").
# Try "average" explicitly first, fall back to default if not available.
try:
    templates = templates_ext.get_templates(operator="average")
    print("  Template operator: average")
except Exception:
    templates = templates_ext.get_templates()
    print("  Template operator: default (average not explicitly available)")

if templates is None or np.all(templates == 0):
    raise RuntimeError(
        "Templates array is None or all zeros. "
        "Re-run: analyzer.compute('templates', operators=['average'])"
    )

_, n_template_samples, n_template_channels = templates.shape
print(f"  Templates: {templates.shape} "
      f"({n_template_samples} samples x {n_template_channels} channels)")
print(f"  Value range: [{templates.min():.2f}, {templates.max():.2f}] uV")

# --- Declare unit columns ---
if nwbfile.units is not None:
    raise RuntimeError("Units table already exists. Restart the kernel and run all cells.")

METRIC_COL_MAP = {
    "firing_rate":          ("firing_rate",          "Mean firing rate (Hz)"),
    "snr":                  ("snr",                  "Signal-to-noise ratio of the unit"),
    "isi_violations_ratio": ("isi_violations_ratio", "ISI violation ratio (fraction of spikes < 1.5 ms refractory)"),
    "presence_ratio":       ("presence_ratio",       "Fraction of recording time with detectable spikes"),
    "amplitude_cutoff":     ("amplitude_cutoff",     "Estimated fraction of spikes below detection threshold"),
    "nn_hit_rate":          ("nn_hit_rate",           "Nearest-neighbour hit rate (isolation quality, 0-1)"),
}
for si_col, (nwb_col, desc) in METRIC_COL_MAP.items():
    if si_col in metrics_df.columns:
        nwbfile.add_unit_column(name=nwb_col, description=desc)
    else:
        print(f"  Warning: metric '{si_col}' not found in analyzer -- skipped")

nwbfile.add_unit_column(
    name="waveform_mean",
    description=(
        f"Mean waveform at the peak channel ({n_template_samples} samples), "
        f"uV, sampled at {fs} Hz"
    ),
    index=False,
)
nwbfile.add_unit_column(
    name="peak_channel_id",
    description="Channel index with the largest peak-to-peak template amplitude",
)
if curation_quality:
    nwbfile.add_unit_column("quality",      "Manual quality label from curation.json (good / MUA / noise)")
if passing_qc:
    nwbfile.add_unit_column("passing_qc",  "Passes automated QC thresholds (bool) from passing_qc.json")
if bombcell:
    nwbfile.add_unit_column("bombcell_label", "Bombcell automated label (good / mua / noise) from bombcell_labels.json")
if unitrefine_pred:
    nwbfile.add_unit_column("unitrefine_prediction", "UnitRefine ML prediction (sua / mua / noise) from unitrefine_labels.json")
if unitrefine_prob:
    nwbfile.add_unit_column("unitrefine_probability", "UnitRefine ML confidence score [0-1] from unitrefine_labels.json")

# --- Populate rows ---
def samples_to_seconds(spike_samples):
    return spike_samples.astype(np.float64) / fs

all_unit_id_list = list(all_unit_ids)
total_spikes = 0

for uid in unit_ids:
    uid_int = int(uid)
    spike_samples = sorting.get_unit_spike_train(uid, segment_index=0)
    spike_times_s = samples_to_seconds(spike_samples)
    total_spikes += len(spike_samples)

    # Index into the full template array using the unit's position
    # in the original (unfiltered) unit list -- not the good-only list
    tmpl_idx  = all_unit_id_list.index(uid)
    template  = templates[tmpl_idx]              # (n_samples, n_channels)
    peak_ch   = int(np.argmax(np.ptp(template, axis=0)))
    waveform  = template[:, peak_ch]             # (n_samples,) at peak channel

    # Sanity check: warn if this unit's waveform looks flat
    if np.max(np.abs(waveform)) < 1e-6:
        print(f"  Warning: unit {uid_int} waveform appears to be all zeros")

    row_kwargs = dict(
        spike_times     = spike_times_s,
        waveform_mean   = waveform,
        peak_channel_id = peak_ch,
    )
    for si_col, (nwb_col, _) in METRIC_COL_MAP.items():
        if si_col in metrics_df.columns:
            val = metrics_df.loc[uid, si_col]
            row_kwargs[nwb_col] = float(val) if not pd.isna(val) else -1.0
    if curation_quality:
        row_kwargs["quality"] = curation_quality.get(uid_int, "unlabelled")
    if passing_qc:
        row_kwargs["passing_qc"] = passing_qc.get(uid_int, False)
    if bombcell:
        row_kwargs["bombcell_label"] = bombcell.get(uid_int, "unknown")
    if unitrefine_pred:
        row_kwargs["unitrefine_prediction"] = unitrefine_pred.get(uid_int, "unknown")
    if unitrefine_prob:
        row_kwargs["unitrefine_probability"] = unitrefine_prob.get(uid_int, -1.0)

    nwbfile.add_unit(**row_kwargs)

print(f"  {n_units} units, {total_spikes:,} total spikes")
print(f"  Metrics stored  : {[v[0] for k,v in METRIC_COL_MAP.items() if k in metrics_df.columns]}")
print(f"  Label columns   : quality={bool(curation_quality)}, passing_qc={bool(passing_qc)}, "
      f"bombcell={bool(bombcell)}, unitrefine={bool(unitrefine_pred)}")
print("Stage 3 done")


# ============================================================
# STAGE 3.5 -- LFP: low-pass, downsample, compress
# ============================================================
print("\n-- Stage 3.5 : LFP extraction and compression")

fs_raw        = SESSION_CONFIG["lfp_source_rate"]  # 2500 Hz LFP band
fs_lfp        = SESSION_CONFIG["lfp_target_rate"]
compression   = SESSION_CONFIG["lfp_compression"]
compress_opts = SESSION_CONFIG["lfp_compression_opts"]
lowpass_hz    = SESSION_CONFIG["lfp_lowpass_hz"]

# Decimation factor must be integer
decimate_factor = int(round(fs_raw / fs_lfp))
actual_fs_lfp   = fs_raw / decimate_factor
if actual_fs_lfp != fs_lfp:
    print(f"  Note: target {fs_lfp} Hz not exactly achievable with integer factor "
          f"{decimate_factor}. Actual LFP rate: {actual_fs_lfp} Hz")

# Load LFP band directly from the SpikeGLX .lf.bin file.
# SpikeInterface reads the paired .lf.meta automatically from the same folder.
lfp_bin_path = Path(SESSION_CONFIG["lfp_bin_path"])
if not lfp_bin_path.exists():
    raise FileNotFoundError(
        f"LFP bin file not found: {lfp_bin_path}\n"
        "Check lfp_bin_path in SESSION_CONFIG."
    )

# Auto-detect the correct LFP stream name from available streams in the folder.
# SpikeGLX names streams like "imec0.lf", "imec1.lf", etc.
# We match on the probe index embedded in the bin filename (e.g. imec1 -> imec1.lf).
_, available_streams = se.get_neo_streams("spikeglx", lfp_bin_path.parent)
lf_streams = [s for s in available_streams if s.endswith(".lf")]

if len(lf_streams) == 0:
    raise RuntimeError(
        f"No .lf stream found in {lfp_bin_path.parent}.\n"
        f"Available streams: {available_streams}"
    )

# Pick the stream whose probe tag (e.g. "imec1") appears in the bin filename
matched = [s for s in lf_streams if s.split(".lf")[0] in lfp_bin_path.name]
if len(matched) == 1:
    stream_name = matched[0]
elif len(matched) == 0:
    # Fallback: use the only lf stream available, warn if ambiguous
    stream_name = lf_streams[0]
    print(f"  Warning: could not match stream to filename, using {stream_name}")
else:
    raise RuntimeError(
        f"Multiple .lf streams match the filename: {matched}. "
        "Set stream_name explicitly in SESSION_CONFIG."
    )

print(f"  Available streams : {available_streams}")
print(f"  Selected stream   : {stream_name}")

recording = se.read_spikeglx(
    folder_path=lfp_bin_path.parent,
    stream_name=stream_name,
    load_sync_channel=False,
)

n_channels = recording.get_num_channels()
n_samples  = recording.get_num_samples(segment_index=0)
print(f"  LFP recording: {n_channels} channels, {n_samples:,} samples at {fs_raw} Hz")
print(f"  Source file  : {lfp_bin_path.name}")

# Process in chunks to avoid loading the full recording into RAM.
# Each chunk is CHUNK_SEC seconds of raw LFP, decimated and accumulated.
CHUNK_SEC   = 60                          # seconds per chunk
chunk_size  = int(CHUNK_SEC * fs_raw)     # samples per chunk (raw rate)
n_lfp_samples = int(np.ceil(n_samples / decimate_factor))

print(f"  Downsampling {fs_raw:.0f} Hz -> {actual_fs_lfp:.0f} Hz "
      f"(factor {decimate_factor}, chunk size {CHUNK_SEC} s)...")
print(f"  Expected output: ({n_lfp_samples}, {n_channels}) float32 -- "
      f"{n_lfp_samples * n_channels * 4 / 1e6:.1f} MB")

lfp_traces = np.empty((n_lfp_samples, n_channels), dtype=np.float32)
lfp_written = 0

n_chunks = int(np.ceil(n_samples / chunk_size))
for chunk_idx in range(n_chunks):
    start = chunk_idx * chunk_size
    end   = min(start + chunk_size, n_samples)

    raw_chunk = recording.get_traces(
        start_frame=start,
        end_frame=end,
        segment_index=0,
    )  # (chunk_samples, n_channels)

    dec_chunk = decimate(
        raw_chunk,
        q=decimate_factor,
        ftype="iir",
        axis=0,
        zero_phase=True,
    ).astype(np.float32)

    out_end = lfp_written + dec_chunk.shape[0]
    lfp_traces[lfp_written:out_end] = dec_chunk
    lfp_written = out_end

    if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
        print(f"  Chunk {chunk_idx + 1}/{n_chunks} -- "
              f"{100 * (chunk_idx + 1) / n_chunks:.0f}% done")

# Trim to actual written samples (rounding may leave one extra row)
lfp_traces = lfp_traces[:lfp_written]
n_lfp_samples = lfp_written
print(f"  LFP array: {lfp_traces.shape} -- "
      f"{n_lfp_samples / actual_fs_lfp:.1f} s, "
      f"{lfp_traces.nbytes / 1e6:.1f} MB")

# Reference all electrodes (384 = 192 sites x 2 pads)
all_electrode_indices  = list(range(len(probe_df) * 2))
electrode_table_region = nwbfile.create_electrode_table_region(
    region=all_electrode_indices,
    description="All Neuropixels physical channels inside brain (384 total)",
)

# Wrap with H5DataIO: gzip compression + byte shuffle + chunked by 1 s
chunk_samples = int(actual_fs_lfp)  # 1 second of LFP samples
compressed_lfp = H5DataIO(
    data=lfp_traces,
    compression=compression,
    compression_opts=compress_opts,
    chunks=(chunk_samples, n_channels),
    shuffle=True,
)

lfp_electrical_series = ElectricalSeries(
    name="LFP",
    description=(
        f"Local field potential: decimated from {fs_raw:.0f} Hz to "
        f"{actual_fs_lfp:.0f} Hz using scipy.signal.decimate (zero-phase IIR, "
        f"internal anti-aliasing). Compression: {compression} level "
        f"{compress_opts} with byte shuffle. Shape: (time, channel). Units: uV."
    ),
    data=compressed_lfp,
    electrodes=electrode_table_region,
    starting_time=0.0,
    rate=float(actual_fs_lfp),
    filtering=(
        f"Anti-aliasing IIR Chebyshev type I applied internally by "
        f"scipy.signal.decimate, zero-phase. Effective low-pass at "
        f"{lowpass_hz} Hz."
    ),
    conversion=1e-6,  # stored in uV; NWB standard unit is V
)

# LFP is processed data -> goes in a processing module, not acquisition
ecephys_module = nwbfile.create_processing_module(
    name="ecephys",
    description="Extracellular electrophysiology processed data",
)
ecephys_module.add(LFP(electrical_series=lfp_electrical_series))

print(f"  LFP stored at {actual_fs_lfp:.0f} Hz, "
      f"{compression} level {compress_opts}, shuffle=True, "
      f"chunks=({chunk_samples}, {n_channels})")
print("Stage 3.5 done")


# ============================================================
# STAGE 4 -- Behavioral CSV -> Trials Table
# ============================================================
print("\n-- Stage 4 : Behaviour -> Trials table")

beh_df = pd.read_csv(SESSION_CONFIG["behavior_csv"])
t0     = beh_df["session_start"].iloc[0]
fs_beh = SESSION_CONFIG["sampling_rate"]


def imec_to_seconds(val):
    if pd.notna(val):
        return float(val) / fs_beh
    return -1.0


def safe_float(val, ref=0.0):
    if pd.notna(val):
        return float(val) - ref
    return -1.0


def safe_bool(val):
    return bool(int(val)) if pd.notna(val) else False


nwbfile.add_trial_column("rw_start",            "Reward window start time (s from session start)")
nwbfile.add_trial_column("trial_duration",       "Total trial duration (s)")
nwbfile.add_trial_column("iti",                  "Inter-trial interval (s)")
nwbfile.add_trial_column("lick_time",            "First lick time (s from session start); -1 if no lick")
nwbfile.add_trial_column("trial_number",         "Trial number (1-indexed)")
nwbfile.add_trial_column("block",                "Task block type (e.g. sound, action-left)")
nwbfile.add_trial_column("stim",                 "Stimulus presented (bool)")
nwbfile.add_trial_column("freq_8khz",            "8 kHz tone presented (bool)")
nwbfile.add_trial_column("freq_16khz",           "16 kHz tone presented (bool)")
nwbfile.add_trial_column("early_lick",           "Early lick before response window (bool)")
nwbfile.add_trial_column("lick",                 "Lick in response window (bool)")
nwbfile.add_trial_column("left_spout",           "Lick to left spout (bool)")
nwbfile.add_trial_column("right_spout",          "Lick to right spout (bool)")
nwbfile.add_trial_column("reward",               "Reward delivered (bool)")
nwbfile.add_trial_column("punishment",           "Punishment delivered (bool)")
nwbfile.add_trial_column("omission",             "Omission trial outcome (bool)")
nwbfile.add_trial_column("catch_trial",          "Catch trial (bool)")
nwbfile.add_trial_column("distractor_trial",     "Distractor trial (bool)")
nwbfile.add_trial_column("distractor_left",      "Distractor on left side (bool)")
nwbfile.add_trial_column("distractor_right",     "Distractor on right side (bool)")
nwbfile.add_trial_column("imec_blue_led_on",     "Blue LED onset, IMEC clock (s); -1 if absent")
nwbfile.add_trial_column("imec_blue_led_off",    "Blue LED offset, IMEC clock (s); -1 if absent")
nwbfile.add_trial_column("imec_stim_on",         "Stimulus onset, IMEC clock (s); -1 if absent")
nwbfile.add_trial_column("imec_stim_off",        "Stimulus offset, IMEC clock (s); -1 if absent")
nwbfile.add_trial_column("imec_punishment_on",   "Punishment onset, IMEC clock (s); -1 if absent")
nwbfile.add_trial_column("imec_punishment_off",  "Punishment offset, IMEC clock (s); -1 if absent")
nwbfile.add_trial_column("imec_reward_on",       "Reward onset, IMEC clock (s); -1 if absent")
nwbfile.add_trial_column("imec_reward_off",      "Reward offset, IMEC clock (s); -1 if absent")
nwbfile.add_trial_column("imec_lick",            "Lick event, IMEC clock (s); -1 if absent")

for _, row in beh_df.iterrows():
    nwbfile.add_trial(
        start_time           = float(row["trial_start"] - t0),
        stop_time            = float(row["trial_end"]   - t0),
        rw_start             = float(row["RW_start"] - t0),
        trial_duration       = float(row["trial_duration"]),
        iti                  = float(row["ITI"]),
        lick_time            = safe_float(row["lick_time"], ref=t0),
        trial_number         = int(row["trial_number"]),
        block                = str(row["block"]),
        stim                 = safe_bool(row["stim"]),
        freq_8khz            = safe_bool(row["8KHz"]),
        freq_16khz           = safe_bool(row["16KHz"]),
        early_lick           = safe_bool(row["early_lick"]),
        lick                 = safe_bool(row["lick"]),
        left_spout           = safe_bool(row["left_spout"]),
        right_spout          = safe_bool(row["right_spout"]),
        reward               = safe_bool(row["reward"]),
        punishment           = safe_bool(row["punishment"]),
        omission             = safe_bool(row["omission"]),
        catch_trial          = safe_bool(row["catch_trial"]),
        distractor_trial     = safe_bool(row["distractor_trial"]),
        distractor_left      = safe_bool(row["distractor_left"]),
        distractor_right     = safe_bool(row["distractor_right"]),
        imec_blue_led_on     = imec_to_seconds(row["blue_led_imec_up"]),
        imec_blue_led_off    = imec_to_seconds(row["blue_led_imec_down"]),
        imec_stim_on         = imec_to_seconds(row["stim_imec_up"]),
        imec_stim_off        = imec_to_seconds(row["stim_imec_down"]),
        imec_punishment_on   = imec_to_seconds(row["punishment_imec_up"]),
        imec_punishment_off  = imec_to_seconds(row["punishment_imec_down"]),
        imec_reward_on       = imec_to_seconds(row["reward_imec_up"]),
        imec_reward_off      = imec_to_seconds(row["reward_imec_down"]),
        imec_lick            = imec_to_seconds(row["lick_imec"]),
    )

n_trials   = len(beh_df)
n_rewarded = int(beh_df["reward"].sum())
print(f"  {n_trials} trials, {n_rewarded} rewarded ({100 * n_rewarded / n_trials:.1f}%)")
print(f"  Blocks: {beh_df['block'].unique().tolist()}")
print(f"  t0 (session start) = {t0:.3f} s (Unix epoch)")
print("Stage 4 done")


# ============================================================
# STAGE 5 -- Write & Validate
# ============================================================
print("\n-- Stage 5 : Write and validate")

output_path = SESSION_CONFIG["output_nwb"]
output_path.parent.mkdir(parents=True, exist_ok=True)

with NWBHDF5IO(output_path, mode="w") as io:
    io.write(nwbfile)

file_size_mb = output_path.stat().st_size / (1024 ** 2)
print(f"  Written to : {output_path}  ({file_size_mb:.1f} MB)")

with NWBHDF5IO(output_path, mode="r") as io:
    nwb_check = io.read()

    n_el    = len(nwb_check.electrodes)
    regions = set(nwb_check.electrodes["acronym"].data[:])
    n_u     = len(nwb_check.units)
    n_s     = sum(len(nwb_check.units["spike_times"][i]) for i in range(n_u))
    n_t     = len(nwb_check.trials)
    blocks  = set(nwb_check.trials["block"].data[:])
    lfp_shape = nwb_check.processing["ecephys"]["LFP"]["LFP"].data.shape

print(f"\n  Validation report")
print(f"    Electrodes  : {n_el} channels, {len(regions)} regions")
print(f"    Units       : {n_u} units, {n_s:,} spikes")
print(f"    Trials      : {n_t} trials, blocks={blocks}")
print(f"    LFP         : shape={lfp_shape}, "
      f"rate={fs_raw / decimate_factor:.0f} Hz, "
      f"compression={compression} level {compress_opts}")
print(f"\nAll done -- NWB file ready at {output_path}")