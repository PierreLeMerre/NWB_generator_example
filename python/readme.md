# nwb-builder

Converts a SpikeInterface SortingAnalyzer + behavioral CSV + probe anatomy CSV
into a single NWB file ready for sharing or analysis.

---

## Project layout

```
nwb_builder/
├── pyproject.toml       ← dependencies & project metadata
├── README.md            ← this file
├── create_nwb.py        ← main script (edit SESSION_CONFIG at the top)
└── data/                ← put your input files here
    ├── analyzer/        ← SpikeInterface SortingAnalyzer folder
    ├── behavior_paired.csv
    └── probe_anatomy.csv
```

Output is written to `output/session.nwb` by default (configurable in `SESSION_CONFIG`).

---

## Requirements

- [uv](https://docs.astral.sh/uv/) — fast Python package manager
  ```bash
  # Install uv (one-time, system-wide)
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- Python 3.11 or newer (uv will download it automatically if needed)

---

## Setup — first time

```bash
# 1. Clone or copy this folder onto your machine, then enter it
cd nwb_builder

# 2. Create the virtual environment and install all dependencies
#    uv reads pyproject.toml and pins everything in uv.lock
uv sync

# That's it. uv creates .venv/ inside the project folder.
```

---

## Running the script

```bash
# Always run through uv so it uses the correct isolated environment
uv run python create_nwb.py
```

You will see a step-by-step progress report ending with a validation summary:

```
── Stage 1 : NWB file initialisation ──────────────────
  Session : session_001
  Subject : mouse_01
  Stage 1 done

── Stage 2 : Probe anatomy → ElectrodeTable ───────────
  191 channels inside brain, 12 brain regions
  Regions traversed: SI, GPe, PAL, STR, ...
  Stage 2 done

── Stage 3 : SpikeInterface → Units table ─────────────
  Loaded analyzer: 84 units at 30000 Hz
  Templates: (84, 82, 384)
  Stage 3 done

── Stage 4 : Behaviour → Trials table ─────────────────
  648 trials, 127 rewarded (19.6%)
  Blocks: ['sound', 'action-left']
  Stage 4 done

── Stage 5 : Write & validate ──────────────────────────
  Written to : output/session.nwb  (312.4 MB)

    Validation report
     Electrodes             191 channels, 12 regions
     Units                  84 units, 1,243,882 spikes
     Trials                 648 trials, blocks: {'sound', 'action-left'}

  All done — NWB file is ready at output/session.nwb
```

---

## Configuring a new session

Open `create_nwb.py` and edit the `SESSION_CONFIG` dictionary at the top:

```python
SESSION_CONFIG = {
    "analyzer_path":     Path("data/analyzer"),        # ← your analyzer folder
    "behavior_csv":      Path("data/behavior.csv"),    # ← your behavioral CSV
    "probe_anatomy_csv": Path("data/probe_anatomy.csv"),
    "output_nwb":        Path("output/session.nwb"),   # ← where to write

    "session_id":        "session_001",                # ← unique session label
    "session_start_time": datetime(2025, 6, 1, 14, 0, 0,
                                   tzinfo=ZoneInfo("Europe/Paris")),
    "experimenter":      ["Your Name"],
    "subject_id":        "mouse_01",
    ...
}
```

---

## Installing optional dev tools (Jupyter, matplotlib)

```bash
uv sync --extra dev
```

Then launch Jupyter in the project environment:

```bash
uv run jupyter lab
```

---

## Updating dependencies

```bash
# Upgrade all packages to their latest compatible versions
uv sync --upgrade

# Upgrade a single package
uv sync --upgrade-package pynwb
```

---

## Reproducing the exact environment on another machine

The `uv.lock` file (generated automatically by `uv sync`) pins every
transitive dependency to an exact version and hash. Simply copy the whole
`nwb_builder/` folder (including `uv.lock`) to the other machine and run:

```bash
uv sync
```

uv will recreate the byte-for-byte identical environment.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `quality_metrics extension not found` | Run `compute_quality_metrics()` on your analyzer first |
| `templates extension not found` | Run `compute_waveforms()` or `compute_templates()` first |
| ISI metric warning | Check column name: may be `isi_violations_ratio` or `isi_violations_count` depending on SI version |
| Large file size | Normal — waveform templates for 384 channels × N units are large; use `compression="gzip"` in NWBHDF5IO if needed |


## Alicante Matlab CSV export to run

```matlab
disp(histo.borders_table_channels.Properties.VariableNames)  % see exact column names
writetable(histo.borders_table_channels, 'borders_table_channels.csv')
writetable(histo.clust2chan_map, 'clust2chan_map.csv')
```
