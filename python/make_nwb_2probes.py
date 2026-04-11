"""
create_nwb_2probes.py
=====================
Builds a single NWB file from two simultaneously recorded Neuropixels probes.

Electrode table layout
----------------------
  probe0 channels : rows   0 .. N0-1   (N0 = n_sites_probe0 * 2)
  probe1 channels : rows  N0 .. N0+N1-1

Units table
-----------
  units.electrodes : DynamicTableRegion pointing to the electrode table row
                     of the peak channel for each unit. This is the standard
                     NWB linkage -- from a unit you can resolve its electrode,
                     then its ElectrodeGroup, then its Device (probe).
  units.probe_label: string label "probe0" / "probe1" for easy filtering

LFP
---
  processing/ecephys/LFP_probe0/ElectricalSeries_probe0
  processing/ecephys/LFP_probe1/ElectricalSeries_probe1
  Each ElectricalSeries.electrodes references the correct absolute row range.

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
    "behavior_csv": Path("/Volumes/T7/NWB_Joana/behavior/behavior_paired.csv"),
    "output_nwb":   Path("/Volumes/T7/NWB_Joana/NWB/999770_20251111_2probes.nwb"),

    "session_description":    "Adaptive sensorimotor task with PFC-STR recordings - Beginner session",
    "session_start_time":     datetime(2025, 11, 11, 14, 0, 0,
                                       tzinfo=ZoneInfo("Europe/Paris")),
    "experimenter":           ["Joana Catarino"],
    "lab":                    "Carlen Lab",
    "institution":            "Karolinska Institutet",
    "experiment_description": "Dual Neuropixels recording during an adaptive sensorimotor task",
    "session_id":             "999770_20251111_2probes",

    "subject_id":      "999770",
    "subject_species": "Mus musculus",
    "subject_sex":     "M",
    "subject_age":     "P60D",

    # AP band rate -- spike train conversion + IMEC sync timestamps
    "sampling_rate": 30000.0,

    # LFP settings shared for all probes
    "lfp_source_rate":     2500.0,   # Hz -- SpikeGLX LFP band native rate
    "lfp_target_rate":     1000.0,   # Hz -- rate after downsampling
    "lfp_lowpass_hz":       400.0,   # Hz -- anti-aliasing cutoff
    "lfp_compression":     "gzip",
    "lfp_compression_opts":     4,
}


# ============================================================
# PROBE CONFIGS  --  one dict per probe, ORDER MATTERS
# The first probe occupies electrode rows 0..N0-1,
# the second probe occupies rows N0..N0+N1-1, etc.
# ============================================================
PROBE_CONFIGS = [
    {
        "probe_label":     "probe0",
        "probe_name":      "Neuropixels 1.0 -- probe 0",
        "analyzer_path":   Path("/Volumes/T7/NWB_Joana/sorted/analyzer_probe0"),
        "anatomy_csv":     Path("/Volumes/T7/NWB_Joana/histology/999770_neuropixels_probe0.csv"),
        "lfp_bin_path":    Path("/Volumes/T7/NWB_Joana/lfp/999770_day1_1_g0_t0.imec0.lf.bin"),
        "curation_json":   None,   # None -> look in analyzer_path/curation.json
        "passing_qc_json": None,
        "bombcell_json":   None,
        "unitrefine_json": None,
    },
    {
        "probe_label":     "probe1",
        "probe_name":      "Neuropixels 1.0 -- probe 1",
        "analyzer_path":   Path("/Volumes/T7/NWB_Joana/sorted/analyzer_probe1"),
        "anatomy_csv":     Path("/Volumes/T7/NWB_Joana/histology/999770_neuropixels_probe1.csv"),
        "lfp_bin_path":    Path("/Volumes/T7/NWB_Joana/lfp/999770_day1_1_g0_t0.imec1.lf.bin"),
        "curation_json":   None,
        "passing_qc_json": None,
        "bombcell_json":   None,
        "unitrefine_json": None,
    },
]


# ============================================================
# HELPERS
# ============================================================
def load_json_if_exists(path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    print(f"    Note: {p.name} not found -- column will be skipped")
    return None


def resolve_json(probe_cfg, key, default_name):
    override = probe_cfg.get(key)
    if override is not None:
        return Path(override)
    return probe_cfg["analyzer_path"] / default_name


def parse_curation(raw):
    result = {}
    if raw is None:
        return result
    for entry in raw.get("manual_labels", []):
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


def decimate_chunked(recording, decimate_factor, chunk_sec=60):
    """Decimate an LFP recording in chunks to avoid loading it all into RAM."""
    fs_raw     = recording.get_sampling_frequency()
    n_samples  = recording.get_num_samples(segment_index=0)
    n_channels = recording.get_num_channels()
    chunk_size = int(chunk_sec * fs_raw)
    n_out      = int(np.ceil(n_samples / decimate_factor))
    n_chunks   = int(np.ceil(n_samples / chunk_size))

    out     = np.empty((n_out, n_channels), dtype=np.float32)
    written = 0

    for i in range(n_chunks):
        start = i * chunk_size
        end   = min(start + chunk_size, n_samples)
        chunk = recording.get_traces(start_frame=start, end_frame=end,
                                     segment_index=0)
        dec   = decimate(chunk, q=decimate_factor, ftype="iir",
                         axis=0, zero_phase=True).astype(np.float32)
        out[written:written + dec.shape[0]] = dec
        written += dec.shape[0]

        if (i + 1) % 10 == 0 or i == n_chunks - 1:
            print(f"    Chunk {i + 1}/{n_chunks} -- "
                  f"{100 * (i + 1) / n_chunks:.0f}% done")

    return out[:written]


def autodetect_lfp_stream(lfp_bin_path):
    """Return the correct .lf stream name by matching the bin filename."""
    _, available = se.get_neo_streams("spikeglx", lfp_bin_path.parent)
    lf_streams   = [s for s in available if s.endswith(".lf")]
    if not lf_streams:
        raise RuntimeError(f"No .lf stream found in {lfp_bin_path.parent}. "
                           f"Available: {available}")
    matched = [s for s in lf_streams if s.split(".lf")[0] in lfp_bin_path.name]
    if len(matched) == 1:
        return matched[0]
    if len(matched) == 0:
        print(f"    Warning: could not match stream to filename, using {lf_streams[0]}")
        return lf_streams[0]
    raise RuntimeError(f"Multiple .lf streams match {lfp_bin_path.name}: {matched}")


METRIC_COL_MAP = {
    "firing_rate":          ("firing_rate",          "Mean firing rate (Hz)"),
    "snr":                  ("snr",                  "Signal-to-noise ratio"),
    "isi_violations_ratio": ("isi_violations_ratio", "ISI violation ratio"),
    "presence_ratio":       ("presence_ratio",       "Fraction of recording with detectable spikes"),
    "amplitude_cutoff":     ("amplitude_cutoff",     "Estimated fraction of spikes below threshold"),
    "nn_hit_rate":          ("nn_hit_rate",           "Nearest-neighbour hit rate (0-1)"),
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

ecephys_module = nwbfile.create_processing_module(
    name="ecephys",
    description="Extracellular electrophysiology processed data",
)

print(f"  Session : {SESSION_CONFIG['session_id']}")
print(f"  Subject : {SESSION_CONFIG['subject_id']}")
print(f"  Probes  : {[p['probe_label'] for p in PROBE_CONFIGS]}")
print("Stage 1 done")


# ============================================================
# STAGE 2 -- ElectrodeTable  (all probes, sequential rows)
# ============================================================
print("\n-- Stage 2 : Probe anatomy -> ElectrodeTable")

# Custom columns declared once -- apply to every row regardless of probe
nwbfile.add_electrode_column("probe_label",       "Probe identifier (probe0, probe1, ...)")
nwbfile.add_electrode_column("channel_id",        "Physical channel index on probe (1-384)")
nwbfile.add_electrode_column("pad_side",          "Electrode pad side: left or right")
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

# probe_electrode_ranges[label] = (row_start, row_end)
# row indices are ABSOLUTE positions in the global electrode table.
# These are used later to build DynamicTableRegion for units and LFP.
probe_electrode_ranges = {}
electrode_row_counter  = 0

for probe_cfg in PROBE_CONFIGS:
    label = probe_cfg["probe_label"]
    print(f"\n  [{label}]")

    probe_df = pd.read_csv(probe_cfg["anatomy_csv"], index_col=0)
    probe_df = probe_df[probe_df["inside_brain"] == True].reset_index(drop=True)
    n_sites    = len(probe_df)
    n_channels = n_sites * 2   # two pads per site

    # One Device per probe
    device = nwbfile.create_device(
        name=probe_cfg["probe_name"],
        description="Neuropixels 1.0 silicon probe (IMEC)",
        manufacturer="IMEC",
    )

    # One ElectrodeGroup per probe
    electrode_group = nwbfile.create_electrode_group(
        name=f"{label}_shank0",
        description=f"Single shank Neuropixels probe ({label})",
        location=", ".join(probe_df["acronym"].unique().tolist()),
        device=device,
    )

    row_start = electrode_row_counter

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

        # Left pad  (Neuropixels 1.0: pads are +-16 um from shank midline)
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

    row_end = electrode_row_counter + n_channels
    probe_electrode_ranges[label] = (row_start, row_end)
    electrode_row_counter = row_end

    print(f"    {n_sites} sites -> {n_channels} channels")
    print(f"    Electrode table rows : {row_start} to {row_end - 1}")
    print(f"    Regions : {', '.join(probe_df['acronym'].unique())}")

print(f"\n  Total electrode rows : {electrode_row_counter}")
print("Stage 2 done")


# ============================================================
# STAGE 3 -- Units Table  (all probes, single global table)
#
# units.electrodes is a DynamicTableRegion -> electrode table.
# For each unit we store the ABSOLUTE row index of its peak channel
# so NWB can resolve: unit -> electrode row -> ElectrodeGroup -> Device.
#
# Absolute peak channel row =
#     probe_electrode_row_start
#     + (peak_channel_local_index * 2)     <- 2 rows per site (left + right pad)
#     + pad_offset (0=left, 1=right)
#
# Because the template peak channel index is the SpikeInterface channel index
# (0-based within the probe recording), and each SpikeInterface channel maps
# to one physical pad, we compute:
#     absolute_row = row_start + peak_ch_si_index
# This is valid as long as the electrode table rows are added in the same
# order as SpikeInterface's channel ordering (left pad first, right pad
# second, site by site from tip to top), which is what Stage 2 does.
# ============================================================
print("\n-- Stage 3 : Units table")

# Declare all unit columns once before the probe loop.
# electrodes : DynamicTableRegion linking each unit to its peak electrode row
nwbfile.add_unit_column(
    name="probe_label",
    description="Probe identifier (probe0, probe1, ...)",
)
# Quality metrics -- declared for all probes; sentinel -1.0 if not computed
for si_col, (nwb_col, desc) in METRIC_COL_MAP.items():
    nwbfile.add_unit_column(name=nwb_col, description=desc)

nwbfile.add_unit_column(
    name="waveform_mean",
    description="Mean waveform at peak channel (uV), 1-D array (n_samples,)",
)
# Curation labels -- sentinel values used when a file is absent
nwbfile.add_unit_column("quality",               "Manual quality label (good / MUA / noise)")
nwbfile.add_unit_column("passing_qc",            "Passes automated QC thresholds (bool)")
nwbfile.add_unit_column("bombcell_label",        "Bombcell automated label")
nwbfile.add_unit_column("unitrefine_prediction", "UnitRefine ML prediction (sua / mua / noise)")
nwbfile.add_unit_column("unitrefine_probability","UnitRefine ML confidence score [0-1]")

total_units_all = 0

for probe_cfg in PROBE_CONFIGS:
    label     = probe_cfg["probe_label"]
    row_start = probe_electrode_ranges[label][0]
    row_end   = probe_electrode_ranges[label][1]
    print(f"\n  [{label}] electrode rows {row_start}-{row_end - 1}")

    # Load curation files
    curation_quality              = parse_curation(
        load_json_if_exists(resolve_json(probe_cfg, "curation_json",   "curation.json")))
    passing_qc                    = parse_passing_qc(
        load_json_if_exists(resolve_json(probe_cfg, "passing_qc_json", "passing_qc.json")))
    bombcell                      = parse_bombcell(
        load_json_if_exists(resolve_json(probe_cfg, "bombcell_json",   "bombcell_labels.json")))
    unitrefine_pred, unitrefine_prob = parse_unitrefine(
        load_json_if_exists(resolve_json(probe_cfg, "unitrefine_json", "unitrefine_labels.json")))

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
        print(f"    Labels : {label_counts}")
        print(f"    Keeping {len(unit_ids)} / {len(all_unit_ids)} good units")
        if len(unit_ids) == 0:
            raise RuntimeError(f"No good units for {label}.")
    else:
        print(f"    Warning: no curation.json -- using all units")
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
        print("    Template operator : average")
    except Exception:
        templates = templates_ext.get_templates()
        print("    Template operator : default")
    if templates is None or np.all(templates == 0):
        raise RuntimeError(
            f"Templates for {label} are all zeros. "
            "Re-run: analyzer.compute('templates', operators=['average'])"
        )
    n_template_samples = templates.shape[1]

    all_unit_id_list = list(all_unit_ids)
    probe_spikes     = 0

    for uid in unit_ids:
        uid_int = int(uid)
        spikes  = sorting.get_unit_spike_train(uid, segment_index=0)
        times_s = spikes.astype(np.float64) / fs
        probe_spikes += len(spikes)

        tmpl_idx = all_unit_id_list.index(uid)
        template = templates[tmpl_idx]              # (n_samples, n_si_channels)
        # SpikeInterface channel index of the peak (0-based within this probe)
        peak_ch_si = int(np.argmax(np.ptp(template, axis=0)))
        waveform   = template[:, peak_ch_si]        # (n_samples,)

        if np.max(np.abs(waveform)) < 1e-6:
            print(f"    Warning: unit {uid_int} waveform is all zeros")

        # Absolute electrode table row for the peak channel.
        # SpikeInterface channel index maps directly to electrode table row
        # within this probe's block because Stage 2 adds electrodes in the
        # same order (channel 0 -> row row_start, channel 1 -> row_start+1, ...).
        abs_electrode_row = row_start + peak_ch_si

        # DynamicTableRegion: a single-element region pointing to the
        # electrode table row of this unit's peak channel
        peak_electrode_region = nwbfile.create_electrode_table_region(
            region=[abs_electrode_row],
            description=(
                f"Peak electrode for this unit on {label} "
                f"(absolute electrode table row {abs_electrode_row})"
            ),
        )

        row_kwargs = dict(
            spike_times            = times_s,
            electrodes             = peak_electrode_region,
            probe_label            = label,
            waveform_mean          = waveform,
            quality                = curation_quality.get(uid_int, "unlabelled"),
            passing_qc             = passing_qc.get(uid_int, False),
            bombcell_label         = bombcell.get(uid_int, "unknown"),
            unitrefine_prediction  = unitrefine_pred.get(uid_int, "unknown"),
            unitrefine_probability = unitrefine_prob.get(uid_int, -1.0),
        )

        for si_col, (nwb_col, _) in METRIC_COL_MAP.items():
            if si_col in metrics_df.columns:
                val = metrics_df.loc[uid, si_col]
                row_kwargs[nwb_col] = float(val) if not pd.isna(val) else -1.0
            else:
                row_kwargs[nwb_col] = -1.0

        nwbfile.add_unit(**row_kwargs)

    total_units_all += len(unit_ids)
    print(f"    {len(unit_ids)} units, {probe_spikes:,} spikes stored")

print(f"\n  Total units : {total_units_all}")
print("Stage 3 done")


# ============================================================
# STAGE 3.5 -- LFP  (per probe, into shared ecephys module)
#
# Each LFP ElectricalSeries.electrodes references the correct
# ABSOLUTE row range from the electrode table for that probe.
# ============================================================
print("\n-- Stage 3.5 : LFP extraction and compression")

fs_raw        = SESSION_CONFIG["lfp_source_rate"]
fs_lfp        = SESSION_CONFIG["lfp_target_rate"]
lowpass_hz    = SESSION_CONFIG["lfp_lowpass_hz"]
compression   = SESSION_CONFIG["lfp_compression"]
compress_opts = SESSION_CONFIG["lfp_compression_opts"]
decimate_factor = int(round(fs_raw / fs_lfp))
actual_fs_lfp   = fs_raw / decimate_factor

print(f"  Decimation : {fs_raw:.0f} Hz -> {actual_fs_lfp:.0f} Hz "
      f"(factor {decimate_factor})")

for probe_cfg in PROBE_CONFIGS:
    label        = probe_cfg["probe_label"]
    lfp_bin_path = Path(probe_cfg["lfp_bin_path"])
    row_start, row_end = probe_electrode_ranges[label]
    print(f"\n  [{label}] electrode rows {row_start}-{row_end - 1}")

    if not lfp_bin_path.exists():
        raise FileNotFoundError(
            f"LFP bin file not found: {lfp_bin_path}\n"
            "Check lfp_bin_path in PROBE_CONFIGS."
        )

    stream_name = autodetect_lfp_stream(lfp_bin_path)
    print(f"    Stream : {stream_name}")

    lfp_rec = se.read_spikeglx(
        folder_path=lfp_bin_path.parent,
        stream_name=stream_name,
        load_sync_channel=False,
    )

    n_lfp_ch  = lfp_rec.get_num_channels()
    n_lfp_raw = lfp_rec.get_num_samples(segment_index=0)
    print(f"    Source : {n_lfp_ch} channels, {n_lfp_raw:,} samples at {fs_raw:.0f} Hz")

    # Verify channel count matches electrode table block size
    n_electrode_rows = row_end - row_start
    if n_lfp_ch != n_electrode_rows:
        print(f"    Warning: LFP has {n_lfp_ch} channels but electrode table "
              f"block has {n_electrode_rows} rows. "
              f"Referencing first {min(n_lfp_ch, n_electrode_rows)} rows.")

    lfp_traces = decimate_chunked(lfp_rec, decimate_factor)
    print(f"    Output : {lfp_traces.shape}, "
          f"{lfp_traces.nbytes / 1e6:.1f} MB uncompressed")

    # Electrode table region: absolute rows row_start..row_start+n_lfp_ch-1
    lfp_electrode_indices = list(range(row_start,
                                       row_start + min(n_lfp_ch, n_electrode_rows)))
    electrode_table_region = nwbfile.create_electrode_table_region(
        region=lfp_electrode_indices,
        description=(
            f"All Neuropixels channels inside brain for {label} "
            f"(electrode table rows {row_start} to "
            f"{row_start + len(lfp_electrode_indices) - 1})"
        ),
    )

    chunk_samples  = int(actual_fs_lfp)   # 1 second of LFP samples
    compressed_lfp = H5DataIO(
        data=lfp_traces,
        compression=compression,
        compression_opts=compress_opts,
        chunks=(chunk_samples, n_lfp_ch),
        shuffle=True,
    )

    lfp_series = ElectricalSeries(
        name=f"ElectricalSeries_{label}",
        description=(
            f"LFP from {label}: {fs_raw:.0f} Hz -> {actual_fs_lfp:.0f} Hz "
            f"(zero-phase IIR decimate, factor {decimate_factor}). "
            f"Electrode table rows {row_start}-"
            f"{row_start + len(lfp_electrode_indices) - 1}. "
            f"Compression: {compression} level {compress_opts}, shuffle=True. "
            f"Shape: (time, channel). Units: uV."
        ),
        data=compressed_lfp,
        electrodes=electrode_table_region,
        starting_time=0.0,
        rate=float(actual_fs_lfp),
        filtering=(
            f"Anti-aliasing IIR Chebyshev type I (scipy.signal.decimate, "
            f"zero-phase). Effective low-pass: {lowpass_hz} Hz."
        ),
        conversion=1e-6,
    )

    lfp_container = LFP(
        name=f"LFP_{label}",
        electrical_series=lfp_series,
    )
    ecephys_module.add(lfp_container)
    print(f"    Stored as processing/ecephys/LFP_{label}/"
          f"ElectricalSeries_{label}")

print("Stage 3.5 done")


# ============================================================
# STAGE 4 -- Behavioral CSV -> Trials Table  (shared, once)
# ============================================================
print("\n-- Stage 4 : Behaviour -> Trials table")

beh_df = pd.read_csv(SESSION_CONFIG["behavior_csv"])
t0     = beh_df["session_start"].iloc[0]
fs_beh = SESSION_CONFIG["sampling_rate"]


def imec_to_seconds(val):
    return float(val) / fs_beh if pd.notna(val) else -1.0


def safe_float(val, ref=0.0):
    return float(val) - ref if pd.notna(val) else -1.0


def safe_bool(val):
    return bool(int(val)) if pd.notna(val) else False


nwbfile.add_trial_column("rw_start",            "Reward window start time (s from session start)")
nwbfile.add_trial_column("trial_duration",       "Total trial duration (s)")
nwbfile.add_trial_column("iti",                  "Inter-trial interval (s)")
nwbfile.add_trial_column("lick_time",            "First lick time (s from session start); -1 if absent")
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

# Readback validation
with NWBHDF5IO(output_path, mode="r") as io:
    nwb_check = io.read()

    n_el     = len(nwb_check.electrodes)
    n_t      = len(nwb_check.trials)
    n_u      = len(nwb_check.units)
    blocks   = set(nwb_check.trials["block"].data[:])
    probe_labels_stored = list(nwb_check.units["probe_label"].data[:])

    print(f"\n  Validation report")
    print(f"    Total electrodes : {n_el}")
    print(f"    Total units      : {n_u}")
    print(f"    Total trials     : {n_t}, blocks={blocks}")
    print()

    for probe_cfg in PROBE_CONFIGS:
        lbl       = probe_cfg["probe_label"]
        rs, re    = probe_electrode_ranges[lbl]
        n_el_p    = re - rs

        # Count units and spikes for this probe
        n_u_p = probe_labels_stored.count(lbl)
        n_s_p = sum(
            len(nwb_check.units["spike_times"][i])
            for i, pl in enumerate(probe_labels_stored) if pl == lbl
        )

        # Verify units electrode references fall within probe's row range
        elec_refs = [
            nwb_check.units["electrodes"][i][0]   # single-element region -> first row
            for i, pl in enumerate(probe_labels_stored) if pl == lbl
        ]
        bad_refs = [r for r in elec_refs if not (rs <= r < re)]

        # LFP shape
        lfp = nwb_check.processing["ecephys"][f"LFP_{lbl}"][f"ElectricalSeries_{lbl}"]

        print(f"    [{lbl}]")
        print(f"      Electrode rows : {rs} to {re - 1}  ({n_el_p} channels)")
        print(f"      Units          : {n_u_p} units, {n_s_p:,} spikes")
        print(f"      Electrode refs : all in range = {len(bad_refs) == 0}"
              + (f"  (BAD: {bad_refs[:5]})" if bad_refs else ""))
        print(f"      LFP            : shape={lfp.data.shape}, "
              f"rate={actual_fs_lfp:.0f} Hz")

print(f"\nAll done -- NWB file ready at {output_path}")
