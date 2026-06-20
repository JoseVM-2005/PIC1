from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field
import delays_db
import parameters_database
import importlib
from scipy.ndimage import zoom
importlib.reload(parameters_database)
importlib.reload(delays_db)
import mplhep as hep

DELAY_REGISTRY = delays_db.DELAY_REGISTRY
CFD_LEVELS = parameters_database.CFD_LEVELS


# =============================================================================
# Adjacency helpers
# =============================================================================

def _chebyshev_adjacent(p1: tuple, p2: tuple) -> bool:
    """True if two (row, col) pixels are Chebyshev-adjacent (share edge or corner)."""
    return abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1


def _is_connected(pixels: list[tuple]) -> bool:
    """
    True if all pixels form a single connected component under Chebyshev adjacency.
    Uses union-find.
    """
    n = len(pixels)
    if n <= 1:
        return True

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if _chebyshev_adjacent(pixels[i], pixels[j]):
                union(i, j)

    return len({find(i) for i in range(n)}) == 1

#=============================
# Diagnostics
#=============================
"""
mean_spread_diagnostic.py
--------------------------
Drop-in addon for delta_t_analysis.py / channel_pair_analysis.py.

Adds:
  1. Two small edits to plot_channel_pair_histograms (marked PATCH 1, PATCH 2)
     that record mu_pair / mu_pair_err per channel pair and print them in
     the summary table.
  2. A new function diagnose_mean_spread() that takes the per-pair (n, mu,
     mu_err, sigma) values and the pooled (global) resolution, and reports
     whether the pooled resolution is inflated by inter-channel-pair
     miscalibration of the mean delay.

Nothing here changes the existing return value or signature of
plot_channel_pair_histograms beyond adding two columns to the printed
table — the function still returns the same dict of figures.
"""



# =============================================================================
# PATCH 1 — inside plot_channel_pair_histograms, in the per-pair loop,
# right after computing mu50/mu50_err (CFD_plot/CFD_plot resolution block),
# nothing needs to change there — mu50 and mu50_err already exist.
# The only change is in what gets appended to summary_rows:
#
# BEFORE:
#   summary_rows.append((c1, c2, n_ev,
#                         res50, res50_err, res50_std, res50_std_err,
#                         best_k1, best_k2, best_res, best_err,
#                         best_k1_std, best_k2_std, best_res_std, best_err_std))
#
# AFTER:
#   summary_rows.append((c1, c2, n_ev,
#                         mu50, mu50_err,                     # <-- ADD
#                         res50, res50_err, res50_std, res50_std_err,
#                         best_k1, best_k2, best_res, best_err,
#                         best_k1_std, best_k2_std, best_res_std, best_err_std))
# =============================================================================


# =============================================================================
# PATCH 2 — summary table print loop, unpack + print the two new columns
#
# BEFORE:
#   for row in summary_rows:
#       (c1, c2, n_ev,
#        r50, e50, r50s, e50s,
#        bk1, bk2, br, be,
#        bk1s, bk2s, brs, bes) = row
#
# AFTER:
#   for row in summary_rows:
#       (c1, c2, n_ev,
#        mu50, mu50_err,                                       # <-- ADD
#        r50, e50, r50s, e50s,
#        bk1, bk2, br, be,
#        bk1s, bk2s, brs, bes) = row
#
# And add a column to the header / print f-string, e.g.:
#   f"{'mu (ps)':>14}"   ...   f"{mu50*1e3:.1f} ± {mu50_err*1e3:.1f}":>14
#
# A full ready-to-paste version of plot_channel_pair_histograms section 6
# is given in `summary_table_block.py` alongside this file if you'd rather
# copy-paste than hand-edit.
# =============================================================================


def diagnose_mean_spread(
    pair_results: List[Tuple[int, int, int, float, float, float, float]],
    pooled_resolution_ps: float,
    pooled_resolution_err_ps: Optional[float] = None,
    method_label: str = "Gaussian fit",
) -> dict:
    """
    Quantify whether the pooled (global) resolution is inflated by
    residual inter-channel-pair delay miscalibration.

    Parameters
    ----------
    pair_results : list of tuples
        (c1, c2, n_events, mu_ns, mu_err_ns, sigma_res_ps, sigma_res_err_ps)
        — one entry per channel pair. mu_ns is the fitted Δt centroid for
        that pair (ns); sigma_res_ps is that pair's own timing resolution
        in ps (i.e. sigma/sqrt(2)*1e3, NOT the raw Gaussian sigma).
    pooled_resolution_ps : float
        The resolution measured from the heatmap / plot_delta_t_histogram
        on the full pooled (all-pairs-combined) Δt distribution, in ps.
    pooled_resolution_err_ps : float, optional
        Uncertainty on the pooled resolution, for reporting only.
    method_label : str
        Just a label for the printout (e.g. "Gaussian fit" or "Std dev").

    Returns
    -------
    dict with keys:
        weighted_mean_mu_ns   : event-count-weighted average of mu_pair
        spread_of_means_ps    : weighted std of mu_pair values, in ps
        weighted_mean_sigma_ps: event-count-weighted average of per-pair
                                 resolutions, in ps
        quadrature_prediction_ps : sqrt(weighted_mean_sigma² + spread_of_means²)
        pooled_resolution_ps  : as given
        inflation_ps          : pooled_resolution_ps - weighted_mean_sigma_ps
        inflation_fraction    : inflation_ps / weighted_mean_sigma_ps
    """
    # Filter out pairs with non-finite values — can't use them
    clean = [
        (c1, c2, n, mu, mu_err, sig, sig_err)
        for (c1, c2, n, mu, mu_err, sig, sig_err) in pair_results
        if np.isfinite(mu) and np.isfinite(sig) and n > 0
    ]

    if len(clean) < 2:
        print("Not enough valid pairs (<2) to compute a mean-spread diagnostic.")
        return {}

    n_arr   = np.array([c[2] for c in clean], dtype=np.float64)
    mu_arr  = np.array([c[3] for c in clean], dtype=np.float64)   # ns
    sig_arr = np.array([c[5] for c in clean], dtype=np.float64)   # ps (resolution, not raw sigma)

    weights = n_arr  # event-count weighting

    # --- Weighted mean of the per-pair centroids ---
    weighted_mean_mu_ns = np.average(mu_arr, weights=weights)

    # --- Weighted spread of the centroids around that mean ---
    # This directly measures how mis-centered the pairs are relative to
    # each other — the quantity that inflates the pooled distribution.
    weighted_var_mu = np.average((mu_arr - weighted_mean_mu_ns) ** 2, weights=weights)
    spread_of_means_ps = np.sqrt(weighted_var_mu) * 1e3   # ns -> ps

    # --- Weighted mean of the per-pair *true* resolutions ---
    weighted_mean_sigma_ps = np.average(sig_arr, weights=weights)

    # --- Quadrature prediction for the pooled resolution ---
    # If pooling several Gaussians of similar width sigma, each offset
    # from a common mean by delta_i, the resulting pooled variance is
    # (to leading order, equal weights) sigma_true^2 + spread_of_means^2.
    quadrature_prediction_ps = np.sqrt(weighted_mean_sigma_ps ** 2 + spread_of_means_ps ** 2)

    inflation_ps = pooled_resolution_ps - weighted_mean_sigma_ps
    inflation_fraction = (
        inflation_ps / weighted_mean_sigma_ps if weighted_mean_sigma_ps > 0 else np.nan
    )

    # --- Report ---
    print()
    print("=" * 78)
    print(f"  MEAN-SPREAD DIAGNOSTIC  ({method_label})")
    print("=" * 78)
    print(f"  Pairs used                         : {len(clean)}")
    print(f"  Weighted mean of pair resolutions   : {weighted_mean_sigma_ps:6.1f} ps   "
          f"<- 'true' single-pair resolution estimate")
    print(f"  Weighted spread of pair means (μ)   : {spread_of_means_ps:6.1f} ps   "
          f"<- inter-channel-pair miscalibration")
    print(f"  Quadrature-predicted pooled res.    : {quadrature_prediction_ps:6.1f} ps   "
          f"<- sqrt(mean_sigma^2 + spread_of_means^2)")
    print("  " + "-" * 76)
    pooled_str = f"{pooled_resolution_ps:.1f}"
    if pooled_resolution_err_ps is not None and np.isfinite(pooled_resolution_err_ps):
        pooled_str += f" ± {pooled_resolution_err_ps:.1f}"
    print(f"  Actual pooled (heatmap) resolution  : {pooled_str:>8} ps")
    print(f"  Inflation (pooled - mean_sigma)     : {inflation_ps:6.1f} ps "
          f"({inflation_fraction*100:+.1f}%)" if np.isfinite(inflation_fraction) else "  Inflation: N/A")
    print("=" * 78)

    if spread_of_means_ps > 0.3 * weighted_mean_sigma_ps:
        print("  -> Mean spread is >30% of the single-pair resolution: ")
        print("     residual inter-channel delay miscalibration is likely")
        print("     contributing meaningfully to the pooled resolution.")
    else:
        print("  -> Mean spread is small relative to the single-pair resolution:")
        print("     pooled inflation from delay miscalibration appears minor.")
    print()

    return {
        "weighted_mean_mu_ns":       weighted_mean_mu_ns,
        "spread_of_means_ps":        spread_of_means_ps,
        "weighted_mean_sigma_ps":    weighted_mean_sigma_ps,
        "quadrature_prediction_ps":  quadrature_prediction_ps,
        "pooled_resolution_ps":      pooled_resolution_ps,
        "inflation_ps":              inflation_ps,
        "inflation_fraction":        inflation_fraction,
    }


# =============================================================================
# Event dataclass — one per coincidence window
# =============================================================================

     
@dataclass
class _LGADHit:
    """Representative time + metadata for one LGAD within an event."""
    lgad:         str
    n_pixels:     int
    is_cluster:   bool              # True if >1 pixel was averaged
    snr:          float
    times:        Dict[int, float]  # {cfd_level: corrected_time_ns}
    channel:      int = -1          # raw SAMPIC channel; -1 for clusters


# =============================================================================
# Core event builder
# =============================================================================

def build_events(
    corrdb_path: str,
    run_name: str,
    output_path: Optional[str] = None,
    coincidence_window_ns: float = 100.0,
    event_building_cfd: int = 50,
    cfd_levels: List[int] = CFD_LEVELS,
) -> str:
    """
    Group hits from the corrected-time DB into coincidence events, apply
    the single/cluster/discard logic per LGAD, and compute Δt for every
    (k1, k2) CFD pair.

    Output parquet — one row per valid event
    ----------------------------------------
    EventID          : int32
    LGAD1, LGAD2     : string (board names of the two sensors)
    LGAD1_NPixels    : int8   (1 = single hit, >1 = cluster)
    LGAD2_NPixels    : int8
    LGAD1_IsCluster  : bool
    LGAD2_IsCluster  : bool
    LGAD1_SNR        : float32
    LGAD2_SNR        : float32
    Dt_k1_k2         : float32  for all 9×9 = 81 pairs  (ns)

    Parameters
    ----------
    corrdb_path : str
        Path to the output of build_full_hit_database (contains
        CFD{k}Time_corr, SampicChannel, LGAD, PixelRow, PixelCol, SNR).
        SNR (along with AmpRatio and rise-time) is already filtered at this
        stage via the hit mask used to build corrdb, so no SNR cut is
        applied here — `snrs` is only used downstream for cluster
        SNR-weighting and the reported LGAD1_SNR/LGAD2_SNR columns.
    coincidence_window_ns : float
        Maximum time span from first to last hit in an event window.
    event_building_cfd : int
        CFD level used to sort and group hits into events.
    cfd_levels : list of int
        CFD levels stored in corrdb (must match what was built).
    """

    if output_path is None:
        p = Path(corrdb_path)
        output_path = str(p.with_stem(p.stem + "_events"))

    # ------------------------------------------------------------------
    # Load corrdb fully into memory — it's the compact DB, should be fine
    # ------------------------------------------------------------------
    print(f"Loading corrected DB: {corrdb_path}")
    table = pq.read_table(corrdb_path)

    hit_ids   = table["HITNumber"].to_pylist()
    lgads     = table["LGAD"].to_pylist()          # already decoded
    rows      = table["PixelRow"].to_pylist()
    cols      = table["PixelCol"].to_pylist()
    snrs      = table["SNR"].to_pylist()

    ordered_times = np.array(table["OrderedCell0Time"].to_pylist(), dtype=np.float64)

    cfd_offsets = {
        k: np.array(table[f"CFD{k}Offset"].to_pylist(), dtype=np.float64)
        for k in cfd_levels
    }
    channels = table["Channel"].to_pylist()

    delays = DELAY_REGISTRY[run_name]

    # Precompute per-hit delay as a float64 array — avoids dict lookup in inner loop
    delay_per_hit = np.array(
        [delays.channel_delays.get(int(ch), np.nan) for ch in channels],
        dtype=np.float64,
    )

    def get_corrected_time(i: int, k: int) -> float:
        offset = cfd_offsets[k][i]
        delay  = delay_per_hit[i]
        if np.isfinite(offset) and np.isfinite(delay):
            return ordered_times[i] + offset - delay
        return np.nan

    n_hits = len(hit_ids)
    print(f"  {n_hits:,} hits loaded")

    channels_arr = np.array(channels, dtype=np.int32)
    unique_channels = np.unique(channels_arr)

    # ------------------------------------------------------------------
    # Drop hits with invalid sorting time
    # (SNR/AmpRatio/rise-time were already filtered upstream via the hit
    # mask used to build corrdb — no quality cuts are reapplied here.)
    # ------------------------------------------------------------------
    cut_counts = {int(ch): {"invalid_time": 0} for ch in unique_channels}

    valid_mask = []
    for i in range(n_hits):
        t   = get_corrected_time(i, event_building_cfd)
        ch  = int(channels[i])

        passes_time = np.isfinite(t)
        if not passes_time:
            cut_counts[ch]["invalid_time"] += 1

        valid_mask.append(passes_time)

    idx_valid = [i for i, v in enumerate(valid_mask) if v]
    n_valid   = len(idx_valid)

    # Cutflow summary
    print(f"\n  {'Channel':<10} {'Total':>8} {'Bad time':>10} {'Passing':>10}")
    print(f"  {'-'*42}")
    ch_counts = {}
    for i, ch in enumerate(channels):
        ch_counts[int(ch)] = ch_counts.get(int(ch), 0) + 1

    for ch in sorted(unique_channels):
        total    = ch_counts.get(ch, 0)
        bad_time = cut_counts[ch]["invalid_time"]
        passing  = total - bad_time
        print(f"  {ch:<10} {total:>8,} {bad_time:>10,} {passing:>10,}")
    print(f"  {'-'*42}")
    print(f"  {'TOTAL':<10} {n_hits:>8,} "
          f"{sum(c['invalid_time'] for c in cut_counts.values()):>10,} "
          f"{n_valid:>10,}")

    # Sort hits by corrected CFD50 time for event building
    idx_valid.sort(key=lambda i: get_corrected_time(i, event_building_cfd))


    # ------------------------------------------------------------------
    # Coincidence grouping — fixed window from first hit in group
    # ------------------------------------------------------------------
    groups: list[list[int]] = []         # list of [hit_index, ...]
    if n_valid == 0:
        print("No valid hits — nothing to group.")
        return output_path

    current_group  = [idx_valid[0]]
    window_start_t = get_corrected_time(idx_valid[0], event_building_cfd)

    for i in idx_valid[1:]:
        t = get_corrected_time(i, event_building_cfd)
        if t is None or not np.isfinite(t):
            continue
        if t - window_start_t <= coincidence_window_ns:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group  = [i]
            window_start_t = t

    groups.append(current_group)
    print(f"  {len(groups):,} coincidence groups formed")

    # ------------------------------------------------------------------
    # Per-event processing
    # ------------------------------------------------------------------

    # Storage for output
    out_event_id        = []
    out_lgad1           = []
    out_lgad2           = []
    out_lgad1_npix      = []
    out_lgad2_npix      = []
    out_lgad1_cluster   = []
    out_lgad2_cluster   = []
    out_lgad1_snr       = []
    out_lgad2_snr       = []
    out_lgad1_ch  = []  
    out_lgad2_ch  = []  

    out_dt: Dict[Tuple[int,int], list] = {
        (k1, k2): [] for k1 in cfd_levels for k2 in cfd_levels
    }

    stats = {
        "total_groups":      len(groups),
        "discarded_lt2lgad": 0,
        "discarded_case3":   0,
        "kept":              0,
        "cluster_events":    0,
    }

    event_id = 0

    for group in groups:

        # --- Partition hits by LGAD ---
        lgad_hits: Dict[str, list[int]] = {}
        for i in group:
            lg = lgads[i]
            lgad_hits.setdefault(lg, []).append(i)

        # Require exactly two LGADs in coincidence
        if len(lgad_hits) != 2:
            stats["discarded_lt2lgad"] += 1
            continue

        lgad_names = sorted(list(lgad_hits.keys()))


        # --- Per-LGAD clustering ---
        lgad_representatives: Dict[str, Optional[_LGADHit]] = {}
        discard = False

        for lg, hit_indices in lgad_hits.items():
            pixels = [(rows[i], cols[i]) for i in hit_indices]

            if len(pixels) == 1:
                # Case 1 — single pixel
                i    = hit_indices[0]
                snr  = float(snrs[i]) if snrs[i] is not None else np.nan
                times = {k: get_corrected_time(i, k) for k in cfd_levels}
                lgad_representatives[lg] = _LGADHit(
                    lgad=lg, n_pixels=1, is_cluster=False,
                    snr=snr, times=times,
                    channel=int(channels[i]),
                )

            elif _is_connected(pixels):
                # Case 2 — connected cluster: SNR-weighted average
                snr_vals  = np.array([
                    float(snrs[i]) if snrs[i] is not None else 0.0
                    for i in hit_indices
                ])
                snr_total = snr_vals.sum()
                weights   = snr_vals / snr_total if snr_total > 0 else np.ones(len(hit_indices)) / len(hit_indices)
                rep_snr   = float(snr_total / len(hit_indices))

                times = {}
                # Case 2 — connected cluster, SNR-weighted average
                for k in cfd_levels:
                    raw = np.array([get_corrected_time(i, k) for i in hit_indices])
                    finite = np.isfinite(raw)
                    if finite.any():
                        w = weights.copy()
                        w[~finite] = 0.0
                        w /= w.sum()
                        times[k] = float(np.dot(w, raw))
                    else:
                        times[k] = np.nan

                lgad_representatives[lg] = _LGADHit(
                    lgad=lg,
                    n_pixels=len(hit_indices),
                    is_cluster=True,
                    snr=rep_snr,
                    times=times,
                )

            else:
                # Case 3 — disconnected: discard this event
                discard = True
                break

        if discard:
            stats["discarded_case3"] += 1
            continue

        # --- Check all CFD times are finite for at least k=50 as sanity ---
        # (Δt will be NaN for individual pairs if either time is NaN — that's fine)
        r1 = lgad_representatives[lgad_names[0]]
        r2 = lgad_representatives[lgad_names[1]]

        # --- Compute Δt matrix ---
        for k1 in cfd_levels:
            for k2 in cfd_levels:
                t1 = r1.times[k1]
                t2 = r2.times[k2]
                dt = t1 - t2 if (np.isfinite(t1) and np.isfinite(t2)) else np.nan
                out_dt[(k1, k2)].append(dt)

        out_event_id.append(event_id)
        out_lgad1.append(lgad_names[0])
        out_lgad2.append(lgad_names[1])
        out_lgad1_npix.append(r1.n_pixels)
        out_lgad2_npix.append(r2.n_pixels)
        out_lgad1_cluster.append(r1.is_cluster)
        out_lgad2_cluster.append(r2.is_cluster)
        out_lgad1_snr.append(r1.snr)
        out_lgad2_snr.append(r2.snr)
        out_lgad1_ch.append(r1.channel)    
        out_lgad2_ch.append(r2.channel)  
 
        if r1.is_cluster or r2.is_cluster:
            stats["cluster_events"] += 1

        event_id += 1
        stats["kept"] += 1

    # ------------------------------------------------------------------
    # Write output parquet
    # ------------------------------------------------------------------
    n_events = len(out_event_id)

    table_dict = {
        "EventID":        pa.array(out_event_id,      type=pa.int32()),
        "LGAD1":          pa.array(out_lgad1,          type=pa.string()).dictionary_encode(),
        "LGAD2":          pa.array(out_lgad2,          type=pa.string()).dictionary_encode(),
        "LGAD1_NPixels":  pa.array(out_lgad1_npix,    type=pa.int8()),
        "LGAD2_NPixels":  pa.array(out_lgad2_npix,    type=pa.int8()),
        "LGAD1_IsCluster":pa.array(out_lgad1_cluster, type=pa.bool_()),
        "LGAD2_IsCluster":pa.array(out_lgad2_cluster, type=pa.bool_()),
        "LGAD1_SNR":      pa.array(out_lgad1_snr,     type=pa.float32()),
        "LGAD2_SNR":      pa.array(out_lgad2_snr,     type=pa.float32()),
        "LGAD1_Channel": pa.array(out_lgad1_ch, type=pa.int32()),   
        "LGAD2_Channel": pa.array(out_lgad2_ch, type=pa.int32()),  
    }

    for (k1, k2), vals in out_dt.items():
        table_dict[f"Dt_{k1}_{k2}"] = pa.array(
            np.asarray(vals, dtype=np.float32), type=pa.float32()
        )

    table = pa.table(table_dict)
    pq.write_table(table, output_path, compression="zstd")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Event database built")
    print("=" * 60)
    print(f"  Total groups formed   : {stats['total_groups']:>10,}")
    print(f"  Discarded (<2 LGADs)  : {stats['discarded_lt2lgad']:>10,}")
    print(f"  Discarded (case 3)    : {stats['discarded_case3']:>10,}")
    print(f"  Valid events kept     : {stats['kept']:>10,}")
    print(f"  Of which cluster hits : {stats['cluster_events']:>10,}")
    print(f"  Output                : {output_path}")
    print("=" * 60 + "\n")

    return output_path

# =============================================================================
# Gaussian helper
# =============================================================================


def _gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fit_gaussian(
    values: np.ndarray,
    bin_edges: Optional[np.ndarray] = None,
):
    v = values[np.isfinite(values)]
    if len(v) < 10:
        return (np.array([np.nan, np.nan, np.nan]),
                np.full((3, 3), np.nan))

    bins = bin_edges if bin_edges is not None else "auto"
    counts, edges = np.histogram(v, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Poisson uncertainty per bin — exclude empty bins entirely
    filled   = counts > 0
    x_fit    = centres[filled]
    y_fit    = counts[filled].astype(float)
    sigma_y  = np.sqrt(y_fit)          # √N per bin, Poisson

    if len(x_fit) < 4:
        return (np.array([np.nan, np.nan, np.nan]),
                np.full((3, 3), np.nan))

    p0 = [np.median(v), np.std(v), float(counts[filled].max())]

    try:
        popt, pcov = curve_fit(
            _gaussian,
            x_fit, y_fit,
            p0=p0,
            sigma=sigma_y,
            absolute_sigma=True,
            maxfev=5000,
        )
        return popt, pcov

    except RuntimeError:
        return (np.array([np.nan, np.nan, np.nan]),
                np.full((3, 3), np.nan))


def plot_delta_t_histogram(
    events_path: str,
    k1: int = 50,
    k2: int = 50,
    cluster_filter: Optional[str] = None,
    n_sigma_window: float = 6.0,    # window half-width in units of kMAD
    n_bins: int = 100,              # bins within that window
    ax: Optional[plt.Axes] = None,
    label: str = "PPS2 Timing Preliminary",
    rlabel: str = "(H8 Test Beam, May 2025)",
    is_data: bool = True,
) -> plt.Figure:

    col = f"Dt_{k1}_{k2}"
    table = pq.read_table(events_path, columns=[col, "LGAD1_IsCluster", "LGAD2_IsCluster"])

    dt_arr       = table[col].to_pylist()
    is_cluster_1 = table["LGAD1_IsCluster"].to_pylist()
    is_cluster_2 = table["LGAD2_IsCluster"].to_pylist()

    dt_np = np.array(dt_arr, dtype=np.float64)
    all_valid = np.isfinite(dt_np)
    
    cluster_arr = np.array([
        (is_cluster_1[i] or is_cluster_2[i])
        for i in range(len(dt_arr))
    ], dtype=bool)

    if cluster_filter == "only":
        keep = all_valid & cluster_arr
    elif cluster_filter == "exclude":
        keep = all_valid & ~cluster_arr
    else:
        keep = all_valid

    vals = dt_np[keep]

    # ------------------------------------------------------------------
    # Robust window 
    # ------------------------------------------------------------------
    centre      = np.median(vals[np.isfinite(vals)])
    kMAD        = 1.4826 * np.median(np.abs(vals[np.isfinite(vals)] - centre))
    window_half = n_sigma_window * kMAD

    in_window   = vals[np.abs(vals - centre) < window_half]
    n_out       = len(vals) - len(in_window)

    bin_edges   = np.linspace(centre - window_half, centre + window_half, n_bins + 1)
    bin_width_ps = (bin_edges[1] - bin_edges[0]) * 1e3

    # ------------------------------------------------------------------
    # Fit Gaussian
    # ------------------------------------------------------------------
    popt, pcov = _fit_gaussian(in_window, bin_edges=bin_edges)
    mu, sigma, amp = popt
    sigma = abs(float(sigma))

    resolution_ps     = (sigma / np.sqrt(2)) * 1e3 if np.isfinite(sigma) else np.nan
    perr              = np.sqrt(np.diag(pcov))
    mu_err, sigma_err, amp_err = perr
    err_resolution_ps = (sigma_err / np.sqrt(2)) * 1e3 if np.isfinite(sigma_err) else np.nan

    # ------------------------------------------------------------------
    # Plot Setup
    # ------------------------------------------------------------------
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    else:
        fig = ax.get_figure()

    # Calculate histogram arrays
    counts, edges = np.histogram(in_window, bins=bin_edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    poisson_err = np.sqrt(counts.astype(float))

    # --- 1. Subtle, clean background fill underneath the data ---
    ax.stairs(
        counts, bin_edges,
        baseline=0, fill=True,
        color="#87DFAA", alpha=0.12,
    )

    # --- 2. Professional step outline (Unfilled edge) ---
    ax.stairs(
        counts, bin_edges,
        color="#2a8a50", lw=1.4,
        label="Data",
    )

    # --- 3. Centralized Poisson error bars (Only on filled bins) ---
    filled = counts > 0
    ax.errorbar(
        centres[filled], counts[filled],
        yerr=poisson_err[filled],
        fmt="none",
        ecolor="#2a8a50",
        elinewidth=1.0,
        capsize=2,
        capthick=0.8,
        zorder=3,
    )

    # --- Fit Overlay ---
    if np.isfinite(sigma):
        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)

        y_model = _gaussian(centres[filled], mu, sigma, amp)
        chi2    = float(np.sum(((counts[filled] - y_model) / np.sqrt(counts[filled])) ** 2))
        ndf     = filled.sum() - 3

        ax.plot(
            x_fit, _gaussian(x_fit, mu, sigma, amp),
            color="#3C3489", lw=2,
            label=rf"Gaussian fit  $\sigma={sigma*1e3:.1f}\pm{sigma_err*1e3:.1f}$ ps",
            zorder=4,
        )
        ax.axvline(mu, color="#3C3489", lw=1, linestyle="--", alpha=0.6, zorder=3)

    # --- Metadata / Labels Formatting ---
    ax.set_xlabel(r"$\Delta t$ (ns)", fontsize=12)
    ax.set_ylabel(f"Events / {bin_width_ps:.0f} ps", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    fig.subplots_adjust(top=0.82)

    ax.set_title(
        rf"$\Delta t$ distribution  (CFD{k2},CFD{k1})"
        + (f"\nTime resolution = {resolution_ps:.1f} ± {err_resolution_ps:.1f} ps"
           if np.isfinite(resolution_ps) else ""),
        fontsize=11, pad=16,
    )

    hep.cms.label(label, data=is_data, rlabel=rlabel, loc=0, ax=ax, fontsize=(14, 12, 12, 11))
    ax.legend(fontsize=10)

    ax.text(
        0.97, 0.97,
        f"μ = {mu:.4f} ± {mu_err:.4f} ns\n"
        f"σ = {sigma:.4f} ± {sigma_err:.4f} ns\n"
        f"χ²/ndf = {chi2:.1f}/{ndf} = {chi2/max(ndf,1):.2f}\n"
        f"Resolution = {resolution_ps:.1f} ± {err_resolution_ps:.1f} ps"
        + (f"\n{n_out:,} events outside window" if n_out > 0 else ""),
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    return fig

# =============================================================================
# 2D resolution heatmap
# =============================================================================

def plot_resolution_heatmap(
    events_path: str,
    cfd_levels: List[int] = CFD_LEVELS,
    cluster_filter: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    n_bins: int = 200,
    n_contours: int = 8,
    lgad1_label: str = "LGAD 1",
    lgad2_label: str = "LGAD 2",
    label: str = "PPS2 Timing Preliminary",
    rlabel: str = "(H8 Test Beam, May 2025)",
    is_data: bool = True,
) -> Tuple[plt.Figure, plt.Figure, float, float, float, float]:

    cols = [f"Dt_{k1}_{k2}" for k1 in cfd_levels for k2 in cfd_levels]
    cols += ["LGAD1_IsCluster", "LGAD2_IsCluster"]
    table = pq.read_table(events_path, columns=cols)

    cluster_arr = (
        np.array(table["LGAD1_IsCluster"].to_pylist(), dtype=bool)
        | np.array(table["LGAD2_IsCluster"].to_pylist(), dtype=bool)
    )

    if cluster_filter == "only":
        row_mask = cluster_arr
    elif cluster_filter == "exclude":
        row_mask = ~cluster_arr
    else:
        row_mask = np.ones(len(table), dtype=bool)

    n = len(cfd_levels)
    
    # 1. Matrices for Gaussian Fit
    res_matrix_fit = np.full((n, n), np.nan)
    err_matrix_fit = np.full((n, n), np.nan)
    
    # 2. Matrices for NumPy Std Dev
    res_matrix_std = np.full((n, n), np.nan)
    err_matrix_std = np.full((n, n), np.nan)

    for i, k1 in enumerate(cfd_levels):
        for j, k2 in enumerate(cfd_levels):
            raw  = np.array(table[f"Dt_{k1}_{k2}"].to_pylist(), dtype=np.float64)
            vals = raw[row_mask & np.isfinite(raw)]
            if len(vals) < 20:
                continue

            centre = np.median(vals)
            kMAD   = 1.4826 * np.median(np.abs(vals - centre))

            # Keep this for the Gaussian fit
            trimmed_fit = vals[np.abs(vals - centre) < 6 * kMAD]
            bin_edges = np.linspace(centre - 6 * kMAD, centre + 6 * kMAD, n_bins + 1)

            # Create a tighter trim strictly for the NumPy STD
            trimmed_std = vals[np.abs(vals - centre) < 3 * kMAD]  # <-- 3 sigma core

            if len(trimmed_fit) < 10 or len(trimmed_std) < 10:
                continue

            # --- Method 1: Gaussian Fit ---
            popt, pcov = _fit_gaussian(trimmed_fit, bin_edges=bin_edges)
            sigma_fit  = abs(float(popt[1]))
            sigma_fit_err = float(np.sqrt(pcov[1, 1])) if np.isfinite(pcov[1, 1]) else np.nan

            if np.isfinite(sigma_fit):
                res_matrix_fit[i, j] = sigma_fit / np.sqrt(2) * 1e3
                err_matrix_fit[i, j] = sigma_fit_err / np.sqrt(2) * 1e3

            # --- Method 2: NumPy Standard Deviation ---
            sigma_std = np.std(trimmed_std)
            sigma_std_err = sigma_std / np.sqrt(2 * len(trimmed_std))
            # Statistical error on sample standard deviation: sigma / sqrt(2N)
            sigma_std_err = sigma_std / np.sqrt(2 * len(trimmed_fit))
            
            res_matrix_std[i, j] = sigma_std / np.sqrt(2) * 1e3
            err_matrix_std[i, j] = sigma_std_err / np.sqrt(2) * 1e3

    # Print outs for benchmark pair
    idx_80 = cfd_levels.index(80)
    idx_60 = cfd_levels.index(60)
    print(f"Fit cfd(60,80): {res_matrix_fit[idx_80, idx_60]:.1f} ± {err_matrix_fit[idx_80, idx_60]:.1f} ps")
    print(f"Std cfd(60,80): {res_matrix_std[idx_80, idx_60]:.1f} ± {err_matrix_std[idx_80, idx_60]:.1f} ps")

    # ------------------------------------------------------------------
    # Reusable Plotting Helper to avoid copy-pasting the layout
    # ------------------------------------------------------------------
    def generate_heatmap(res_matrix, err_matrix, method_title, cbar_label_text):
        if np.all(np.isnan(res_matrix)):
            print(f"WARNING: Not enough data points ( < 20 ) for ANY CFD pair ({method_title}).")
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.text(0.5, 0.5, f"Insufficient Data\n({method_title} Matrix is all NaNs)",
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            return fig, np.nan, np.nan

        res_filled = res_matrix.copy()
        err_filled = err_matrix.copy()
        nan_mask = np.isnan(res_filled)
        if np.any(nan_mask):
            worst_res = np.nanmax(res_filled)
            worst_err = np.nanmax(err_filled)
            res_filled[nan_mask] = worst_res
            err_filled[nan_mask] = worst_err

        fig, ax = plt.subplots(figsize=(7, 6), tight_layout=True)

        flat_min_raw = np.nanargmin(res_matrix)
        i_min_idx, j_min_idx = np.unravel_index(flat_min_raw, res_matrix.shape)

        res_min = res_matrix[i_min_idx, j_min_idx]
        err_min = err_matrix[i_min_idx, j_min_idx]

        zoom_factor = 2
        res_smooth  = zoom(res_filled, zoom_factor, order=1)
        n_smooth = res_smooth.shape[0]

        X, Y = np.meshgrid(
            np.linspace(0, n - 1, n_smooth),
            np.linspace(0, n - 1, n_smooth),
        )

        plot_vmax = vmax if vmax is not None else np.nanpercentile(res_matrix, 90)
        plot_vmin = vmin if vmin is not None else np.nanmin(res_matrix)

        im = ax.imshow(
            res_smooth,
            origin="lower",
            cmap="viridis_r",
            vmin=plot_vmin,
            vmax=plot_vmax,
            aspect="auto",
            extent=[0, n-1, 0, n-1],
        )
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label_text, fontsize=11)
        cbar.ax.tick_params(labelsize=10)

        focus_levels = np.linspace(plot_vmin, plot_vmax, n_contours)
        cs = ax.contour(
            X, Y, res_smooth,
            levels=focus_levels,
            colors="black",
            linewidths=0.8,
            alpha=0.6,
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt=lambda v: f"{v:.1f}p")

        label_text = (f"Min: {res_min:.1f} ± {err_min:.1f} ps\n"
                      f"CFD{cfd_levels[j_min_idx]} / CFD{cfd_levels[i_min_idx]}")

        ax.errorbar(
            j_min_idx, i_min_idx,
            xerr=0.4, yerr=0.4,
            fmt="o",
            color="red",
            markersize=8,
            capsize=0,
            linewidth=1.5,
            label=label_text,
            zorder=5,
        )

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"{k}" for k in cfd_levels], rotation=45, ha="right")
        ax.set_yticklabels([f"{k}" for k in cfd_levels])
        ax.set_xlabel(f"k_{lgad2_label} (%)", fontsize=11)
        ax.set_ylabel(f"k_{lgad1_label} (%)", fontsize=11)
        ax.set_title(f"Detector Timing Resolution Matrix ({method_title})", fontsize=12, pad=16)
        ax.tick_params(axis="both", labelsize=11)
        
        
        hep.cms.label(label, data=is_data, rlabel=rlabel, loc=0, ax=ax, fontsize=(14, 12, 12, 11))
        ax.legend(fontsize=9, loc="lower left")
        legend = ax.legend(fontsize=9, loc="lower left")
        for text in legend.get_texts():
            text.set_color("white")
        return fig, res_min, err_min

    # Generate both figures seamlessly
    fig_fit, res_min_fit, err_min_fit = generate_heatmap(
        res_matrix_fit, err_matrix_fit, 
        method_title="Gaussian Fit", 
        cbar_label_text=r"Time Resolution $\sigma(\Delta t) / \sqrt{2}$ (ps)"
    )
    
    fig_std, res_min_std, err_min_std = generate_heatmap(
        res_matrix_std, err_matrix_std, 
        method_title="NumPy Std Dev", 
        cbar_label_text=r"Time Resolution $\text{STD}(\Delta t) / \sqrt{2}$ (ps)"
    )

    return fig_fit, fig_std, res_min_fit, err_min_fit, res_min_std, err_min_std

 
 
# =============================================================================
# Core analysis helper (shared with existing functions)
# =============================================================================
 
def _resolution_from_values(
    vals: np.ndarray,
    n_bins: int,
    n_sigma_window: float,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, float, int]:
    """
    Gaussian fit on kMAD-windowed data.
    Returns: mu, mu_err, sigma, sigma_err, bin_edges, counts, chi2, ndf
    All in ns.  Resolution = sigma / sqrt(2).
    """
    vals = vals[np.isfinite(vals)]
    if len(vals) < 20:
        return np.nan, np.nan, np.nan, np.nan, np.array([]), np.array([]), np.nan, 0

    centre    = np.median(vals)
    kMAD      = 1.4826 * np.median(np.abs(vals - centre))
    hw        = n_sigma_window * kMAD
    trimmed   = vals[np.abs(vals - centre) < hw]
    bin_edges = np.linspace(centre - hw, centre + hw, n_bins + 1)

    popt, pcov = _fit_gaussian(trimmed, bin_edges=bin_edges)
    mu, sigma, amp = popt
    sigma = abs(float(sigma))
    mu_err, sigma_err, _ = np.sqrt(np.diag(pcov))

    counts, _ = np.histogram(trimmed, bins=bin_edges)
    centres_b = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    filled    = counts > 0
    if filled.sum() >= 4 and np.isfinite(sigma):
        y_model = _gaussian(centres_b[filled], mu, sigma, amp)
        chi2    = float(np.sum(((counts[filled] - y_model) / np.sqrt(counts[filled])) ** 2))
        ndf     = int(filled.sum()) - 3
    else:
        chi2, ndf = np.nan, 0

    return mu, mu_err, sigma, sigma_err, bin_edges, counts, chi2, ndf
 
 
def _std_resolution_from_values(
    vals: np.ndarray,
    n_sigma_window: float = 3.0,
) -> Tuple[float, float]:
    """
    Unbinned std-dev resolution (3σ_kMAD core trim, matching heatmap).
    Returns (resolution_ps, error_ps) where resolution = std / sqrt(2).
    Error = std / sqrt(2N)  — statistical uncertainty on sample std.
    """
    vals = vals[np.isfinite(vals)]
    if len(vals) < 20:
        return np.nan, np.nan

    centre  = np.median(vals)
    kMAD    = 1.4826 * np.median(np.abs(vals - centre))
    trimmed = vals[np.abs(vals - centre) < n_sigma_window * kMAD]
    if len(trimmed) < 20:
        return np.nan, np.nan

    sigma     = np.std(trimmed)
    sigma_err = sigma / np.sqrt(2 * len(trimmed))
    return sigma / np.sqrt(2) * 1e3, sigma_err / np.sqrt(2) * 1e3


##----Helper to plot histograms----##
def _draw_histogram_with_errors(
    ax: plt.Axes,
    counts: np.ndarray,
    bin_edges: np.ndarray,
    mu: float,
    sigma: float,
    sigma_err: float,
    amp: float,
) -> None:
    """
    Draw a Δt histogram with:
      • Professional unfilled step outline with a faint background tint
      • Poisson √N error bars on every filled bin (no empty bin markers)
      • Gaussian fit curve (without uncertainty band)

    Parameters
    ----------
    ax         : matplotlib Axes object
    counts     : event counts per bin from np.histogram
    bin_edges  : edges used for histogramming and fit
    mu, sigma, amp : Gaussian fit parameters (ns)
    sigma_err  : uncertainty on sigma from covariance (ns)
    """
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    poisson_err = np.sqrt(counts.astype(float))   # √N per bin

    # --- 1. Subtle, clean background fill underneath the data ---
    ax.stairs(
        counts, bin_edges,
        baseline=0, fill=True,
        color="#87DFAA", alpha=0.12,
    )

    # --- 2. Professional step outline (Unfilled edge) ---
    ax.stairs(
        counts, bin_edges,
        color="#2a8a50", lw=1.4,
        label="Data",
    )

    # --- 3. Centralized Poisson error bars (Only where data exists) ---
    filled = counts > 0
    ax.errorbar(
        centres[filled], counts[filled],
        yerr=poisson_err[filled],
        fmt="none",
        ecolor="#2a8a50",
        elinewidth=1.0,
        capsize=2,
        capthick=0.8,
        zorder=3,
    )

    # Note: The 'if np.any(~filled):' empty bin block has been completely removed.

    if not np.isfinite(sigma):
        return

    # --- 4. Gaussian fit curve ---
    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)
    y_fit = _gaussian(x_fit, mu, sigma, amp)
    ax.plot(
        x_fit, y_fit,
        color="#3C3489", lw=2,
        label=rf"Gaussian fit  $\sigma={sigma*1e3:.1f}\pm{sigma_err*1e3:.1f}$ ps",
        zorder=4,
    )

    # --- 5. μ line ---
    ax.axvline(mu, color="#3C3489", lw=1, linestyle="--", alpha=0.55, zorder=3)


# =============================================================================
# Main function
# =============================================================================

def plot_channel_pair_histograms(
    events_path: str,
    corrdb_path: str,
    cfd_levels: List[int] = CFD_LEVELS,
    cfd_plot: int = 50,
    n_sigma_window: float = 6.0,
    n_sigma_window_std: float = 3.0,
    n_bins: int = 100,
    output_dir: Optional[str] = None,
    pooled_res_fit_ps: Optional[float] = None,
    pooled_res_fit_err_ps: Optional[float] = None,
    pooled_res_std_ps: Optional[float] = None,
    pooled_res_std_err_ps: Optional[float] = None,
    label: str = "PPS2 Timing Preliminary",
    rlabel: str = "(H8 Test Beam, May 2025)",
    is_data: bool = True,
) -> Dict[Tuple[int, int], plt.Figure]:
    """
    For every adjacent cross-board channel pair, produce a Δt histogram
    (CFD `cfd_plot`/`cfd_plot`) with Poisson error bars and a Gaussian fit.
    Prints a summary table with CFD50/50 and best-CFD resolutions.
    """

    # ------------------------------------------------------------------
    # 1. channel → (row, col) map from corrdb (one vectorised pass)
    # ------------------------------------------------------------------
    print("Building channel→(row,col) map from corrdb …")
    meta    = pq.read_table(corrdb_path, columns=["Channel", "PixelRow", "PixelCol"])
    ch_raw  = meta["Channel"].to_pylist()
    row_raw = meta["PixelRow"].to_pylist()
    col_raw = meta["PixelCol"].to_pylist()

    pixel_map: Dict[int, Tuple[int, int]] = {}
    n_unique = len(set(ch_raw))
    for ch, r, c in zip(ch_raw, row_raw, col_raw):
        ch = int(ch)
        if ch not in pixel_map:
            pixel_map[ch] = (int(r), int(c))
        if len(pixel_map) == n_unique:
            break
    del meta, ch_raw, row_raw, col_raw
    print(f"  {len(pixel_map)} unique channels mapped.")

    # ------------------------------------------------------------------
    # 2. Load events — single-pixel pairs only
    # ------------------------------------------------------------------
    needed_cols = (
        ["LGAD1_Channel", "LGAD2_Channel", "LGAD1_NPixels", "LGAD2_NPixels"]
        + [f"Dt_{k1}_{k2}" for k1 in cfd_levels for k2 in cfd_levels]
    )
    print("Loading events parquet …")
    tbl   = pq.read_table(events_path, columns=needed_cols)
    ch1_arr = np.array(tbl["LGAD1_Channel"].to_pylist(), dtype=np.int32)
    ch2_arr = np.array(tbl["LGAD2_Channel"].to_pylist(), dtype=np.int32)
    npix1   = np.array(tbl["LGAD1_NPixels"].to_pylist(),  dtype=np.int8)
    npix2   = np.array(tbl["LGAD2_NPixels"].to_pylist(),  dtype=np.int8)

    single_mask = (npix1 == 1) & (npix2 == 1) & (ch1_arr >= 0) & (ch2_arr >= 0)
    print(f"  {single_mask.sum():,} / {len(single_mask):,} events are single-pixel pairs.")

    # Load all Dt arrays once (float32 numpy, avoids repeated parquet reads)
    dt_arrays: Dict[Tuple[int, int], np.ndarray] = {
        (k1, k2): np.array(tbl[f"Dt_{k1}_{k2}"].to_pylist(), dtype=np.float32)
        for k1 in cfd_levels for k2 in cfd_levels
    }
    del tbl

    # ------------------------------------------------------------------
    # 3. Enumerate unique adjacent cross-board pairs
    # ------------------------------------------------------------------
    ch1_single = ch1_arr[single_mask]
    ch2_single = ch2_arr[single_mask]
    pair_keys  = set(zip(ch1_single.tolist(), ch2_single.tolist()))

    adjacent_pairs, skipped_pairs = [], []
    for (c1, c2) in sorted(pair_keys):
        p1, p2 = pixel_map.get(c1), pixel_map.get(c2)
        if p1 is None or p2 is None:
            skipped_pairs.append((c1, c2, "coord missing"))
        elif _chebyshev_adjacent(p1, p2):
            adjacent_pairs.append((c1, c2))
        else:
            skipped_pairs.append((c1, c2, f"not adjacent {p1}↔{p2}"))

    print(f"\n  Adjacent pairs : {len(adjacent_pairs)}")
    print(f"  Skipped pairs  : {len(skipped_pairs)}")
    for c1, c2, reason in skipped_pairs:
        print(f"    ch{c1}–ch{c2} : {reason}")

    if not adjacent_pairs:
        print("No adjacent pairs found — nothing to plot.")
        return {}

    # ------------------------------------------------------------------
    # 4. Per-pair boolean masks (built once, reused for every CFD pair)
    # ------------------------------------------------------------------
    pair_mask: Dict[Tuple[int, int], np.ndarray] = {
        (c1, c2): (ch1_single == c1) & (ch2_single == c2)
        for (c1, c2) in adjacent_pairs
    }

    # ------------------------------------------------------------------
    # 5. Per-pair analysis + figure
    # ------------------------------------------------------------------
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    figures: Dict[Tuple[int, int], plt.Figure] = {}
    summary_rows = []

    print(f"\nProcessing {len(adjacent_pairs)} adjacent pairs …\n")

    for (c1, c2) in adjacent_pairs:
        mask = pair_mask[(c1, c2)]
        n_ev = int(mask.sum())
        p1, p2 = pixel_map[c1], pixel_map[c2]

        print(f"  ch{c1}{p1} — ch{c2}{p2}  :  {n_ev:,} events", end="")

        if n_ev < 20:
            print("  [SKIP — too few events]")
            continue

        # --- CFD scan: best Gaussian-fit resolution ---
        best_res, best_err   = np.inf, np.nan
        best_k1, best_k2     = cfd_plot, cfd_plot
        # --- CFD scan: best unbinned-std resolution ---
        best_res_std, best_err_std = np.inf, np.nan
        best_k1_std, best_k2_std   = cfd_plot, cfd_plot

        for k1 in cfd_levels:
            for k2 in cfd_levels:
                v = dt_arrays[(k1, k2)][single_mask][mask].astype(np.float64)

                _, _, sigma, sigma_err, _, _, _, _ = _resolution_from_values(
                    v, n_bins=n_bins, n_sigma_window=n_sigma_window
                )
                if np.isfinite(sigma):
                    r = sigma / np.sqrt(2) * 1e3
                    if r < best_res:
                        best_res  = r
                        best_err  = sigma_err / np.sqrt(2) * 1e3 if np.isfinite(sigma_err) else np.nan
                        best_k1, best_k2 = k1, k2

                r_std, e_std = _std_resolution_from_values(v, n_sigma_window_std)
                if np.isfinite(r_std) and r_std < best_res_std:
                    best_res_std = r_std
                    best_err_std = e_std
                    best_k1_std, best_k2_std = k1, k2

        # --- CFD_plot / CFD_plot resolutions ---
        vals_50 = dt_arrays[(cfd_plot, cfd_plot)][single_mask][mask].astype(np.float64)
        mu50, mu50_err, sig50, sig50_err, bin_edges, counts, chi2, ndf = \
            _resolution_from_values(vals_50, n_bins=n_bins, n_sigma_window=n_sigma_window)

        res50     = sig50     / np.sqrt(2) * 1e3 if np.isfinite(sig50)     else np.nan
        res50_err = sig50_err / np.sqrt(2) * 1e3 if np.isfinite(sig50_err) else np.nan
        res50_std, res50_std_err = _std_resolution_from_values(vals_50, n_sigma_window_std)

        print(f"  CFD{cfd_plot}/{cfd_plot}: fit={res50:.1f} ps  std={res50_std:.1f} ps  |  "
              f"Best fit CFD{best_k1}/{best_k2}: {best_res:.1f} ps  "
              f"Best std CFD{best_k1_std}/{best_k2_std}: {best_res_std:.1f} ps")

         
        summary_rows.append((c1, c2, n_ev,
                              mu50, mu50_err,                       # <-- NEW
                              res50, res50_err, res50_std, res50_std_err,
                              best_k1, best_k2, best_res, best_err,
                              best_k1_std, best_k2_std, best_res_std, best_err_std))
 

        # ------------------------------------------------------------------
        # Figure
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        bin_width_ps = (bin_edges[1] - bin_edges[0]) * 1e3 if len(bin_edges) > 1 else np.nan

        amp50 = float(counts.max()) if len(counts) > 0 else 1.0

        # Pass counts and edges directly to guarantee 100% downstream alignment
        _draw_histogram_with_errors(
            ax=ax, 
            counts=counts, 
            bin_edges=bin_edges,
            mu=mu50, 
            sigma=sig50, 
            sigma_err=sig50_err,
            amp=amp50,
        )

        # n_out: events outside the kMAD window
        if np.isfinite(mu50):
            v_fin   = vals_50[np.isfinite(vals_50)]
            kMAD_50 = 1.4826 * np.median(np.abs(v_fin - np.median(v_fin)))
            n_out   = int(np.sum(np.abs(v_fin - mu50) >= n_sigma_window * kMAD_50))
        else:
            n_out = 0

        ax.set_xlabel(r"$\Delta t$ (ns)", fontsize=12)
        ax.set_ylabel(
            f"Events / {bin_width_ps:.0f} ps" if np.isfinite(bin_width_ps) else "Events",
            fontsize=12,
        )
        ax.tick_params(axis="both", labelsize=12)

        ax.set_title(
            rf"$\Delta t$ distribution  (CFD{cfd_plot}/CFD{cfd_plot})"
            + f"\nch{c1} {p1} — ch{c2} {p2}"
            + (f"\nTime resolution = {res50:.1f} ± {res50_err:.1f} ps"
               if np.isfinite(res50) else ""),
            fontsize=11, pad=16,
        )

        fig.subplots_adjust(top=0.82)
        hep.cms.label(label, data=is_data, rlabel=rlabel, loc=0, ax=ax,
                      fontsize=(14, 12, 12, 11))
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                  framealpha=0.85)

        textstr = (
            f"μ = {mu50:.4f} ± {mu50_err:.4f} ns\n"
            f"σ = {sig50:.4f} ± {sig50_err:.4f} ns\n"
            f"χ²/ndf = {chi2:.1f}/{ndf} = {chi2/max(ndf,1):.2f}\n"
            f"Res (fit) = {res50:.1f} ± {res50_err:.1f} ps\n"
            f"Res (std) = {res50_std:.1f} ± {res50_std_err:.1f} ps\n"
            f"Best fit: CFD{best_k1}/CFD{best_k2} → {best_res:.1f} ± {best_err:.1f} ps\n"
            f"Best std: CFD{best_k1_std}/CFD{best_k2_std} → {best_res_std:.1f} ± {best_err_std:.1f} ps"
            + (f"\n{n_out:,} events outside window" if n_out > 0 else "")
        )
        ax.text(
            0.97, 0.97, textstr,
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        figures[(c1, c2)] = fig

        if output_dir is not None:
            fname = Path(output_dir) / f"dt_pair_ch{c1}_ch{c2}.png"
            fig.savefig(fname, dpi=300, bbox_inches="tight")

    # ------------------------------------------------------------------
    # 6. Summary table
    # ------------------------------------------------------------------
    W = 142
    print()
    print("=" * W)
    print(f"  {'Pair':<10} {'N ev':>7}  "
          f"{'mu (ps)':>14}  "
          f"{'CFD50 fit (ps)':>16}  "
          f"{'CFD50 std (ps)':>16}  "
          f"{'Best fit CFD':>13}  {'Best fit res (ps)':>18}  "
          f"{'Best std CFD':>13}  {'Best std res (ps)':>18}")
    print("  " + "-" * (W - 2))
    for row in summary_rows:
        (c1, c2, n_ev,
         mu50, mu50_err,
         r50, e50, r50s, e50s,
         bk1, bk2, br, be,
         bk1s, bk2s, brs, bes) = row
 
        def _fmt(r, e):
            return f"{r:.1f} ± {e:.1f}" if np.isfinite(r) else "N/A"
 
        mu_str = f"{mu50*1e3:.1f} ± {mu50_err*1e3:.1f}" if np.isfinite(mu50) else "N/A"
 
        print(f"  {'ch'+str(c1)+'–ch'+str(c2):<10} {n_ev:>7,}  "
              f"{mu_str:>14}  "
              f"{_fmt(r50,  e50):>16}  "
              f"{_fmt(r50s, e50s):>16}  "
              f"{'CFD'+str(bk1)+'/'+str(bk2):>13}  {_fmt(br,  be):>18}  "
              f"{'CFD'+str(bk1s)+'/'+str(bk2s):>13}  {_fmt(brs, bes):>18}")
    print("=" * W)

    # ------------------------------------------------------------------
    # 7. Mean-spread diagnostic (only runs if pooled values were supplied)
    # ------------------------------------------------------------------
    if pooled_res_fit_ps is not None:
        pair_results_fit = [
            (c1, c2, n_ev, mu50, mu50_err, r50, e50)
            for (c1, c2, n_ev, mu50, mu50_err, r50, e50, r50s, e50s,
                 bk1, bk2, br, be, bk1s, bk2s, brs, bes) in summary_rows
        ]
        diagnose_mean_spread(
            pair_results=pair_results_fit,
            pooled_resolution_ps=pooled_res_fit_ps,
            pooled_resolution_err_ps=pooled_res_fit_err_ps,
            method_label="Gaussian fit",
        )

    if pooled_res_std_ps is not None:
        pair_results_std = [
            (c1, c2, n_ev, mu50, mu50_err, r50s, e50s)
            for (c1, c2, n_ev, mu50, mu50_err, r50, e50, r50s, e50s,
                 bk1, bk2, br, be, bk1s, bk2s, brs, bes) in summary_rows
        ]
        diagnose_mean_spread(
            pair_results=pair_results_std,
            pooled_resolution_ps=pooled_res_std_ps,
            pooled_resolution_err_ps=pooled_res_std_err_ps,
            method_label="Std dev",
        )

    return figures





# =============================================================================
# Execution Pipeline Example
# =============================================================================
if __name__ == "__main__":
    import numpy as np

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    RAW_PARQUET = "Run_0xx.parquet"

    # [!] Assume `my_delays` and `my_config` are instantiated earlier in your code
    # my_delays = Channel_Delays(...)
    # my_config = ConfigInformation(...)

    # Optional: A mask of verified good hit IDs from an earlier filtering step
    # valid_hit_mask = np.array([10, 11, 15, 102, ...])

    # -------------------------------------------------------------------------
    # 2. Build the hit-level database (1 row = 1 hit)
    # -------------------------------------------------------------------------
    # print(">>> STEP 1: Building unified hit database...")
    # corrdb_file = build_full_hit_database(
    #     raw_parquet_path=RAW_PARQUET,
    #     delays=my_delays,
    #     config=my_config,
    #     # hit_mask=valid_hit_mask,   # Uncomment to skip unverified hits instantly
    #     cfd_levels=CFD_LEVELS,
    #     batch_size=100_000,          # Adjust based on your RAM limits
    #     min_amplitude=0.01
    # )

    # -------------------------------------------------------------------------
    # 3. Build the event-level database (1 row = 1 coincidence event)
    # -------------------------------------------------------------------------
    print("\n>>> STEP 2: Grouping hits into coincidence events...")
    events_file = build_events(
        corrdb_path= "Run_0xx_full_hit_db.parquet",
        coincidence_window_ns= 1000.0, # 1 microsecond window
        event_building_cfd=50,        # Sort chronological groups using CFD 50%
        min_snr= 5                   # Aggressive cut: ignore noisy hits
    )

    # -------------------------------------------------------------------------
    # 4. Generate Visual Diagnostics
    # -------------------------------------------------------------------------
    print("\n>>> STEP 3: Generating diagnostic plots...")

    # Plot A: 1D Histogram for the 50% - 50% CFD pair, EXCLUDING clusters
    fig_hist = plot_delta_t_histogram(
        events_path=events_file,
        k1=50,
        k2=50,
        cluster_filter="exclude",
        bins=200
    )
    fig_hist.savefig("dt_histogram_CFD50.png", dpi=300)
    print("Saved -> dt_histogram_CFD50.png")

    # Plot B: 2D Heatmap of time resolution across ALL CFD pairs
    fig_heatmap = plot_resolution_heatmap(
        events_path=events_file,
        cluster_filter=None           # Include single AND cluster events
    )
    fig_heatmap.savefig("resolution_heatmap.png", dpi=300)
    print("Saved -> resolution_heatmap.png")

    # Render plots to screen if running interactively
    plt.show()

    print("\n>>> Pipeline complete.")