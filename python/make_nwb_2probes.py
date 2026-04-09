"""
create_nwb_2probes.py
=====================
Builds a single NWB file from two Neuropixels probes recorded simultaneously,
including:
  - Two SpikeInterface SortingAnalyzers  (units, waveforms, quality metrics,
                                          curation labels)
  - Two probe anatomy CSVs               (Allen CCF electrode coordinates)
  - Two SpikeGLX .lf.bin files           (LFP per probe, downsampled, compressed)
  - One shared behavioral CSV            (trials table with IMEC sync events)

Both probes share a single NWBFile. Each probe gets its own Device,
ElectrodeGroup, electrode rows, Units table extension, and LFP ElectricalSeries.
The trials table is session-level and appears only once.

Usage
-----
    uv run python create_nwb_2probes.py

Edit SESSION_CONFIG and PROBE_CONFIGS at the top before running.
"""

# ============================================================
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
# SESSION CONFIG  --  shared across both probes
# ============================================================
SESSION_CONFIG = {
    # --- Shared paths ---
    "behavior_csv": Path("/Volumes/T7/NWB_Joana/behavior/behavior_paired.csv"),
    "output_nwb":   Path("/Volumes/T7/NWB_Joana/NWB/999770_20251111_2probes.nwb"),

    # --- Session metadata ---
    "session_description":    "Adaptive sensorimotor task with PFC-STR recordings - Beginner session",
    "session_start_time":     datetime(2025, 11, 11, 14, 0, 0,
                                       tzinfo=ZoneInfo("Europe/Paris")),
    "experimenter":           ["Joana Catarino"],
    "lab":                    "Carlen Lab",
    "institution":            "Karolinska Institutet",
    "experiment_description": "Dual Neuropixels recording during an adaptive sensorimotor task",
    "session_id":             "999770_20251111_2probes",

    # --- Subject ---
    "subject_id":      "999770",
    "subject_species": "Mus musculus",
    "subject_sex":     "M",
    "subject_age":     "P60D",

    # --- AP sampling rate (spike sorting + IMEC sync conversion) ---
    "sampling_rate": 30000.0,

    # --- LFP compression (shared settings for both probes) ---
    "lfp_source_rate":     2500.0,  # Hz -- SpikeGLX LFP band sample rate
    "lfp_target_rate":     1000.0,  # Hz -- target rate after downsampling
    "lfp_lowpass_hz":       400.0,  # Hz -- anti-aliasing cutoff
    "lfp_compression":     "gzip",
    "lfp_compression_opts":     4,
}


# ============================================================
# PROBE CONFIGS  --  one dict per probe
# ============================================================
# Each probe entry is independent. Add or remove probes by editing this list.
# probe_label   : short name used in NWB (Device name, ElectrodeGroup name,
#                 LFP series name, Units table name)
# probe_name    : human-readable hardware name stored in Device
# shank_id      : electrode group name (shank0, shank1, ...)
# analyzer_path : SpikeInterface SortingAnalyzer folder
# anatomy_csv   : Allen CCF probe anatomy CSV
# lfp_bin_path  : SpikeGLX .lf.bin file
# curation_json etc. : None = auto-detect in analyzer_path folder

PROBE_CONFIGS = [
    {
        "probe_label":    "probe0",
        "probe_name":     "Neuropixels 1.0 -- probe 0",
        "shank_id":       "shank0",
        "analyzer_path":  Path("/Volumes/T7/NWB_Joana/sorted/analyzer_probe0"),
        "anatomy_csv":    Path("/Volumes/T7/NWB_Joana/histology/999770_neuropixels_probe0.csv"),
        "lfp_bin_path":   Path("/Volumes/T7/NWB_Joana/lfp/999770_day1_1_g0_t0.imec0.lf.bin"),
        "curation_json":  None,
        "passing_qc_json": None,
        "bombcell_json":  None,
        "unitrefine_json": None,
    },
    {
        "probe_label":    "probe1",
        "probe_name":     "Neuropixels 1.0 -- probe 1",
        "shank_id":       "shank0",
        "analyzer_path":  Path("/Volumes/T7/NWB_Joana/sorted/analyzer_probe1"),
        "anatomy_csv":    Path("/Volumes/T7/NWB_Joana/histology/999770_neuropixels_probe1.csv"),
        "lfp_bin_path":   Path("/Volumes/T7/NWB_Joana/lfp/999770_day1_1_g0_t0.imec1.lf.bin"),
        "curation_json":  None,
        "passing_qc_json": None,
        "bombcell_json":  None,
        "unitrefine_json": None,
    },
]


# ============================================================
# HELPERS
# ============================================================
def load_json_if_exists(path):
    """Return parsed JSON or None if file does not exist."""
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    print(f"    Note: {p.name} not found -- column will be skipped")
    return None


def resolve_json(probe_cfg, key, default_name):
    """Return path from probe config override or fall back to analyzer folder."""
    override = probe_cfg.get(key)
    if override is not None:
        return Path(override)
    return probe_cfg["analyzer_path"] / default_name


def parse_curation(curation_raw):
    """Parse curation.json -> {unit_id (int): quality_label (str)}."""
    result = {}
    if curation_raw is None:
        return result
    for entry in curation_raw.get("manual_labels", []):
        uid   = int(entry["unit_id"])
        label = entry.get("quality",
                entry.get("labels", {}).get("quality", [None]))[0]
        if label is not None:
            result[uid] = str(label)
    return result


def parse_passing_qc(raw):
    if raw is None:
        return {}
    return {int(k): bool(v) for k, v in raw.items()}


def parse_bombcell(raw):
    if raw is None:
        return {}
    return {int(k): str(v).strip("[]' \"")
            for k, v in raw.get("bombcell_label", {}).items()}


def parse_unitrefine(raw):
    pred, prob = {}, {}
    if raw is None:
        return pred, prob
    for k, v in raw.get("unitrefine_prediction", {}).items():
        pred[int(k)] = str(v)
    for k, v in raw.get("unitrefine_probability", {}).items():
        prob[int(k)] = float(v)
    return pred, prob


def samples_to_seconds(spike_samples, fs):
    return spike_samples.astype(np.float64) / fs


METRIC_COL_MAP = {
    "firing_rate":          ("firing_rate",          "Mean firing rate (Hz)"),
    "snr":                  ("snr",                  "Signal-to-noise ratio of the unit"),
    "isi_violations_ratio": ("isi_violations_ratio", "ISI violation ratio (fraction of spikes < 1.5 ms refractory)"),
    "presence_ratio":       ("presence_ratio",       "Fraction of recording time with detectable spikes"),
    "amplitude_cutoff":     ("amplitude_cutoff",     "Estimated fraction of spikes below detection threshold"),
    "nn_hit_rate":          ("nn_hit_rate",           "Nearest-neighbour hit rate (isolation quality, 0-1)"),
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

# One shared ecephys processing module for all LFP data
ecephys_module = nwbfile.create_processing_module(
    name="ecephys",
    description="Extracellular electrophysiology processed data",
)

print(f"  Session : {SESSION_CONFIG['session_id']}")
print(f"  Subject : {SESSION_CONFIG['subject_id']}")
print(f"  Probes  : {[p['probe_label'] for p in PROBE_CONFIGS]}")
print("Stage 1 done")


# ============================================================
# STAGES 2 + 3 + 3.5 -- Per-probe loop
# ============================================================

# electrode_offsets tracks how many electrode rows have been added so far,
# so each probe's LFP ElectricalSeries references the correct rows.
electrode_offset = 0
probe_electrode_counts = {}   # probe_label -> n_channels added

for probe_cfg in PROBE_CONFIGS:
    label    = probe_cfg["probe_label"]
    print(f"\n{'='*60}")
    print(f"  Processing {label}")
    print(f"{'='*60}")

    # ----------------------------------------------------------
    # STAGE 2 -- Probe Anatomy -> ElectrodeTable
    # ----------------------------------------------------------
    print(f"\n-- Stage 2 [{label}] : Probe anatomy -> ElectrodeTable")

    probe_df = pd.read_csv(probe_cfg["anatomy_csv"], index_col=0)
    probe_df = probe_df[probe_df["inside_brain"] == True].reset_index(drop=True)
    n_sites    = len(probe_df)
    n_channels = n_sites * 2
    print(f"  {n_sites} sites inside brain -> {n_channels} channels, "
          f"{probe_df['acronym'].nunique()} brain regions")

    # One Device per probe
    device = nwbfile.create_device(
        name=probe_cfg["probe_name"],
        description="Neuropixels 1.0 silicon probe (IMEC)",
        manufacturer="IMEC",
    )

    # One ElectrodeGroup per probe
    electrode_group = nwbfile.create_electrode_group(
        name=f"{label}_{probe_cfg['shank_id']}",
        description=f"Single shank Neuropixels probe ({label})",
        location=", ".join(probe_df["acronym"].unique().tolist()),
        device=device,
    )

    # Declare custom columns only on the first probe -- they apply to all rows
    if electrode_offset == 0:
        nwbfile.add_electrode_column("probe_label",       "Probe identifier (probe0, probe1, ...)")
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
            probe_label=label,
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

        # Left pad
        nwbfile.add_electrode(
            x=float(row["ml_mm"]) - 0.016,
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

    probe_electrode_counts[label] = n_channels
    print(f"  {n_channels} electrodes added")
    print(f"  Regions: {', '.join(probe_df['acronym'].unique())}")
    print(f"Stage 2 [{label}] done")

    # ----------------------------------------------------------
    # STAGE 3 -- SpikeInterface Analyzer -> Units
    # ----------------------------------------------------------
    print(f"\n-- Stage 3 [{label}] : SpikeInterface -> Units table")

    # Load curation files
    curation_raw   = load_json_if_exists(resolve_json(probe_cfg, "curation_json",   "curation.json"))
    passing_qc_raw = load_json_if_exists(resolve_json(probe_cfg, "passing_qc_json", "passing_qc.json"))
    bombcell_raw   = load_json_if_exists(resolve_json(probe_cfg, "bombcell_json",   "bombcell_labels.json"))
    unitrefine_raw = load_json_if_exists(resolve_json(probe_cfg, "unitrefine_json", "unitrefine_labels.json"))

    curation_quality              = parse_curation(curation_raw)
    passing_qc                    = parse_passing_qc(passing_qc_raw)
    bombcell                      = parse_bombcell(bombcell_raw)
    unitrefine_pred, unitrefine_prob = parse_unitrefine(unitrefine_raw)

    analyzer     = si.load_sorting_analyzer(probe_cfg["analyzer_path"])
    sorting      = analyzer.sorting
    fs           = sorting.get_sampling_frequency()
    all_unit_ids = sorting.unit_ids

    # Filter to good units
    if curation_quality:
        unit_ids = np.array([
            uid for uid in all_unit_ids
            if curation_quality.get(int(uid), "").lower() == "good"
        ])
        label_counts = {}
        for uid in all_unit_ids:
            lbl = curation_quality.get(int(uid), "unlabelled")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        print(f"  Labels: {label_counts}")
        print(f"  Keeping {len(unit_ids)} / {len(all_unit_ids)} good units")
        if len(unit_ids) == 0:
            raise RuntimeError(f"No good units found for {label}.")
    else:
        print(f"  Warning: no curation.json -- using all units")
        unit_ids = all_unit_ids

    # Quality metrics
    metrics_ext = analyzer.get_extension("quality_metrics")
    if metrics_ext is None:
        raise RuntimeError(f"quality_metrics not found for {label}.")
    metrics_df = metrics_ext.get_data()

    # Templates
    templates_ext = analyzer.get_extension("templates")
    if templates_ext is None:
        raise RuntimeError(f"templates not found for {label}.")
    try:
        templates = templates_ext.get_templates(operator="average")
        print("  Template operator: average")
    except Exception:
        templates = templates_ext.get_templates()
        print("  Template operator: default")

    if templates is None or np.all(templates == 0):
        raise RuntimeError(
            f"Templates for {label} are all zeros. "
            "Re-run: analyzer.compute('templates', operators=['average'])"
        )

    _, n_template_samples, n_template_channels = templates.shape
    print(f"  Templates: {templates.shape}")

    # NWB stores multiple units tables via a processing module per probe.
    # We use a ProcessingModule named after the probe label to hold each
    # probe's Units object separately. This avoids NWB's single global
    # units table limitation when probes have different sets of good units.
    from pynwb.misc import Units as NWBUnits

    probe_units = NWBUnits(
        name=f"units_{label}",
        description=f"Spike sorted units from {label} (good units only)",
    )

    # Add columns to this probe's Units object
    for si_col, (nwb_col, desc) in METRIC_COL_MAP.items():
        if si_col in metrics_df.columns:
            probe_units.add_column(name=nwb_col, description=desc)
        else:
            print(f"  Warning: metric '{si_col}' not found -- skipped")

    probe_units.add_column(
        name="waveform_mean",
        description=(
            f"Mean waveform at peak channel ({n_template_samples} samples), "
            f"uV, sampled at {fs} Hz"
        ),
    )
    probe_units.add_column(
        name="peak_channel_id",
        description="Channel index with the largest peak-to-peak amplitude",
    )
    if curation_quality:
        probe_units.add_column("quality",      "Manual quality label (good / MUA / noise)")
    if passing_qc:
        probe_units.add_column("passing_qc",  "Passes automated QC thresholds (bool)")
    if bombcell:
        probe_units.add_column("bombcell_label", "Bombcell automated label")
    if unitrefine_pred:
        probe_units.add_column("unitrefine_prediction", "UnitRefine ML prediction")
    if unitrefine_prob:
        probe_units.add_column("unitrefine_probability", "UnitRefine ML confidence score [0-1]")

    all_unit_id_list = list(all_unit_ids)
    total_spikes = 0

    for uid in unit_ids:
        uid_int   = int(uid)
        spikes    = sorting.get_unit_spike_train(uid, segment_index=0)
        times_s   = samples_to_seconds(spikes, fs)
        total_spikes += len(spikes)

        tmpl_idx = all_unit_id_list.index(uid)
        template = templates[tmpl_idx]
        peak_ch  = int(np.argmax(np.ptp(template, axis=0)))
        waveform = template[:, peak_ch]

        if np.max(np.abs(waveform)) < 1e-6:
            print(f"  Warning: unit {uid_int} waveform is all zeros")

        row_kwargs = dict(
            spike_times     = times_s,
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

        probe_units.add_unit(**row_kwargs)

    # Store units in a per-probe processing module
    probe_module = nwbfile.create_processing_module(
        name=f"units_{label}",
        description=f"Spike sorted units from {label}",
    )
    probe_module.add(probe_units)

    print(f"  {len(unit_ids)} units, {total_spikes:,} spikes")
    print(f"Stage 3 [{label}] done")

    # ----------------------------------------------------------
    # STAGE 3.5 -- LFP
    # ----------------------------------------------------------
    print(f"\n-- Stage 3.5 [{label}] : LFP extraction and compression")

    fs_raw        = SESSION_CONFIG["lfp_source_rate"]
    fs_lfp        = SESSION_CONFIG["lfp_target_rate"]
    lowpass_hz    = SESSION_CONFIG["lfp_lowpass_hz"]
    compression   = SESSION_CONFIG["lfp_compression"]
    compress_opts = SESSION_CONFIG["lfp_compression_opts"]

    decimate_factor = int(round(fs_raw / fs_lfp))
    actual_fs_lfp   = fs_raw / decimate_factor

    lfp_bin_path = Path(probe_cfg["lfp_bin_path"])
    if not lfp_bin_path.exists():
        raise FileNotFoundError(
            f"LFP bin file not found: {lfp_bin_path}\n"
            "Check lfp_bin_path in PROBE_CONFIGS."
        )

    # Auto-detect stream name from filename
    _, available_streams = se.get_neo_streams("spikeglx", lfp_bin_path.parent)
    lf_streams = [s for s in available_streams if s.endswith(".lf")]
    if len(lf_streams) == 0:
        raise RuntimeError(
            f"No .lf stream found in {lfp_bin_path.parent}.\n"
            f"Available streams: {available_streams}"
        )
    matched = [s for s in lf_streams if s.split(".lf")[0] in lfp_bin_path.name]
    if len(matched) == 1:
        stream_name = matched[0]
    elif len(matched) == 0:
        stream_name = lf_streams[0]
        print(f"  Warning: could not match stream to filename, using {stream_name}")
    else:
        raise RuntimeError(f"Multiple .lf streams match filename: {matched}")

    print(f"  Stream: {stream_name}")

    lfp_recording = se.read_spikeglx(
        folder_path=lfp_bin_path.parent,
        stream_name=stream_name,
        load_sync_channel=False,
    )

    n_lfp_channels = lfp_recording.get_num_channels()
    n_lfp_samples  = lfp_recording.get_num_samples(segment_index=0)
    print(f"  Source: {n_lfp_channels} channels, {n_lfp_samples:,} samples at {fs_raw} Hz")

    # Chunked decimation to avoid out-of-memory crash
    CHUNK_SEC   = 60
    chunk_size  = int(CHUNK_SEC * fs_raw)
    n_out_samples = int(np.ceil(n_lfp_samples / decimate_factor))

    print(f"  Downsampling {fs_raw:.0f} -> {actual_fs_lfp:.0f} Hz "
          f"(factor {decimate_factor}, {CHUNK_SEC} s chunks)...")

    lfp_traces  = np.empty((n_out_samples, n_lfp_channels), dtype=np.float32)
    lfp_written = 0
    n_chunks    = int(np.ceil(n_lfp_samples / chunk_size))

    for chunk_idx in range(n_chunks):
        start     = chunk_idx * chunk_size
        end       = min(start + chunk_size, n_lfp_samples)
        raw_chunk = lfp_recording.get_traces(
            start_frame=start, end_frame=end, segment_index=0
        )
        dec_chunk = decimate(
            raw_chunk, q=decimate_factor, ftype="iir", axis=0, zero_phase=True
        ).astype(np.float32)

        out_end = lfp_written + dec_chunk.shape[0]
        lfp_traces[lfp_written:out_end] = dec_chunk
        lfp_written = out_end

        if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
            print(f"  Chunk {chunk_idx + 1}/{n_chunks} -- "
                  f"{100 * (chunk_idx + 1) / n_chunks:.0f}% done")

    lfp_traces = lfp_traces[:lfp_written]
    print(f"  LFP shape: {lfp_traces.shape}, "
          f"{lfp_traces.nbytes / 1e6:.1f} MB uncompressed")

    # Reference this probe's electrode rows
    probe_electrode_indices = list(range(electrode_offset,
                                         electrode_offset + n_channels))
    electrode_table_region = nwbfile.create_electrode_table_region(
        region=probe_electrode_indices,
        description=f"All Neuropixels channels inside brain for {label}",
    )

    chunk_samples  = int(actual_fs_lfp)
    compressed_lfp = H5DataIO(
        data=lfp_traces,
        compression=compression,
        compression_opts=compress_opts,
        chunks=(chunk_samples, n_lfp_channels),
        shuffle=True,
    )

    lfp_series = ElectricalSeries(
        name=f"LFP_{label}",
        description=(
            f"LFP from {label}: decimated from {fs_raw:.0f} Hz to "
            f"{actual_fs_lfp:.0f} Hz (zero-phase IIR decimate). "
            f"Compression: {compression} level {compress_opts}, shuffle=True. "
            f"Shape: (time, channel). Units: uV."
        ),
        data=compressed_lfp,
        electrodes=electrode_table_region,
        starting_time=0.0,
        rate=float(actual_fs_lfp),
        filtering=(
            f"Anti-aliasing IIR Chebyshev type I (scipy.signal.decimate, "
            f"zero-phase). Effective low-pass at {lowpass_hz} Hz."
        ),
        conversion=1e-6,
    )
    ecephys_module.add(LFP(name=f"LFP_{label}",
                           electrical_series=lfp_series))

    print(f"  LFP stored: {actual_fs_lfp:.0f} Hz, "
          f"{compression} level {compress_opts}")
    print(f"Stage 3.5 [{label}] done")

    # Advance electrode offset for next probe
    electrode_offset += n_channels


# ============================================================
# STAGE 4 -- Behavioral CSV -> Trials Table  (shared, once)
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
    n_t     = len(nwb_check.trials)
    blocks  = set(nwb_check.trials["block"].data[:])

    print(f"\n  Validation report")
    print(f"    Electrodes  : {n_el} total channels")
    print(f"    Trials      : {n_t} trials, blocks={blocks}")

    for probe_cfg in PROBE_CONFIGS:
        lbl = probe_cfg["probe_label"]
        mod = nwb_check.processing[f"units_{lbl}"]
        u   = mod[f"units_{lbl}"]
        n_u = len(u)
        n_s = sum(len(u["spike_times"][i]) for i in range(n_u))
        lfp_shape = nwb_check.processing["ecephys"][f"LFP_{lbl}"][f"LFP_{lbl}"].data.shape
        print(f"    {lbl} units  : {n_u} units, {n_s:,} spikes")
        print(f"    {lbl} LFP    : shape={lfp_shape}")

print(f"\nAll done -- NWB file ready at {output_path}")