#!/usr/bin/env python3
"""Compare table-parameter hopping signs with matched starts and a separate expanded screen."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMPLETE = {"complete", "complete_ctm_recovered"}
START = ("InitState", "SpinOffset", "ChargePattern", "ChargeOffset", "LayerAlignment")
CONTROLS = ("Rows", "Cols", "Dmin", "TruncErr", "InverseCutoff", "CTMTolerance", "CTMTruncErr", "CTMSVDCutoff")
DIAGONALS = ("pi2_pi2", "pi2_minus_pi2")
AXES = ("0_pi", "pi_0")
GUARDS = {"0_0": "(0,0)", "pi_pi": "(π,π)", "pi2_0": "(π/2,0)", "0_pi2": "(0,π/2)"}
SCHEDULE = [{"Tau": .1, "Steps": 25}, {"Tau": .03, "Steps": 15}]


def cold_history(case):
    return (case.get("kind") == "evolve" and case.get("start_total_time", 0) == 0
            and not case.get("restart_checkpoint") and case["algorithm"].get("Schedule") == SCHEDULE)


def check_history(case, dimension, sign):
    if sign == -1 and cold_history(case):
        return None
    if (case["kind"] != "measure" or case.get("restart_checkpoint")
            or not math.isclose(case.get("start_total_time", 0), 2.95, abs_tol=1e-12, rel_tol=0)):
        raise ValueError(f"Expected cold or frozen T=2.95 history in {case['id']}")
    expected = case.get("expected_checkpoint_sha256")
    checkpoint = Path(case.get("checkpoint") or "")
    if sign == -1:
        source = case.get("source_cold_case")
        settings = (*START, "Rows", "Cols", "Dmin", "Dmax", "TruncErr", "InverseCutoff")
        if (dimension != 3 or not isinstance(source, dict) or not cold_history(source)
                or source.get("physics") != case["physics"]
                or not source.get("id") or checkpoint.parent.parent.name != source["id"]
                or any(source["algorithm"].get(key) != case["algorithm"].get(key) for key in settings)):
            raise ValueError(f"Frozen D3 case requires a matching original cold history in {case['id']}")
    if sign == -1 or expected is not None:
        if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
                or checkpoint.name != "checkpoint_step00000040"):
            raise ValueError(f"Frozen case requires a fingerprinted step-40 checkpoint in {case['id']}")
    return expected


def read_run(path, name, dimension, sign, count):
    raw = path.read_bytes()
    document = json.loads(raw)
    manifest = document.get("manifest", document)
    if manifest.get("schema") != "lno.cent.followup.v1" or len(manifest["cases"]) != count:
        raise ValueError(f"{name} requires a cent_followup input with {count} cases")
    cases = {case["id"]: case for case in manifest["cases"]}
    records = document.get("records", [])
    by_id = {record["case"]["id"]: record for record in records}
    if len(cases) != count or len(by_id) != len(records) or not by_id.keys() <= cases.keys():
        raise ValueError(f"{name} has duplicate or unexpected case IDs")
    first = manifest["cases"][0]
    physics = {key: value for key, value in first["physics"].items() if key != "TxzYSign"}
    controls = {key: first["algorithm"].get(key) for key in CONTROLS}
    chi = first["algorithm"]["Chi"]
    if type(chi) is not int or chi < 1:
        raise ValueError(f"{name} requires a positive integer χ")
    rows, binaries = [], set()
    for case_id, case in cases.items():
        algorithm = case["algorithm"]
        couplings = dict(case["physics"])
        actual_sign = couplings.pop("TxzYSign", 1)
        if actual_sign != sign or isinstance(actual_sign, bool) or couplings != physics:
            raise ValueError(f"Hamiltonian/sign differs in {case_id}")
        if [algorithm[key] for key in ("Rows", "Cols", "Dmax", "Chi")] != [4, 4, dimension, chi]:
            raise ValueError(f"Unexpected cell, D or χ in {case_id}")
        if {key: algorithm.get(key) for key in CONTROLS} != controls:
            raise ValueError(f"Numerical controls differ in {case_id}")
        expected = check_history(case, dimension, sign)
        record = by_id.get(case_id, {"case": case, "status": "not_started"})
        if record["case"] != case:
            raise ValueError(f"Record differs from manifest in {case_id}")
        row = {"dataset": name, "case_id": case_id, "TxzYSign": sign, "D": dimension, "time": 2.95, "chi": chi,
               "status": record["status"], **{key: algorithm[key] for key in START},
               "selected_d3_case": case.get("selected_d3_case"), "kind": case["kind"], "checkpoint_sha256": expected}
        if record["status"] in COMPLETE:
            m = record["measurement"]
            sector = {"particles_per_cell": 48, "twice_sz_per_cell": None, "symmetry": "fU1QN(N)"}
            if (m["bond_dimension_max"] != dimension or m["exact_symmetry_sector"] != sector
                    or m["chi"] != chi or m.get("ctm_converged") is not True
                    or not math.isclose(record["planned_total_time"], 2.95, abs_tol=1e-12, rel_tol=0)):
                raise ValueError(f"Incompatible measured D, sector, time or CTM in {case_id}")
            signs = m.get("txz_orientation_signs")
            if ("TxzYSign" in case["physics"] or signs is not None) and signs != {"horizontal": 1, "vertical": sign}:
                raise ValueError(f"Measurement does not confirm hopping signs in {case_id}")
            if expected and any((record.get(key) or {}).get("sha256") != expected
                                for key in ("source_checkpoint", "checkpoint_identity")):
                raise ValueError(f"Measurement differs from the frozen checkpoint in {case_id}")
            energy, density = m["physical_energy_per_ni"], m["means"]["n_total"] / 2
            q, layers = record["dominant_q"], tuple(record["layer_q"])
            if not all(map(math.isfinite, (energy, density))) or (q not in {"mixed_layers", "weak_moment"} and layers != (q, q)):
                raise ValueError(f"Invalid measured energy, density or phase in {case_id}")
            binaries.add(record["binary_sha256"])
            row.update(energy_per_ni=energy, n_per_ni=density, actual_D=m["bond_dimension_max"],
                       final_q=q, top_q=layers[0], bottom_q=layers[1])
        rows.append(row)
    if len(binaries) > 1 or len({start(row) for row in rows}) != count:
        raise ValueError(f"{name} mixes binaries or repeats an initializer")
    source = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
              "campaign": manifest["campaign"], "binary_sha256": next(iter(binaries), None), "chi": chi}
    return rows, physics, controls, source


def start(row):
    return tuple(row[key] for key in START)


def summarize(rows):
    finished = [row for row in rows if "energy_per_ni" in row]
    best = {}
    for row in sorted(finished, key=lambda item: (item["energy_per_ni"], item["case_id"])):
        best.setdefault(row["final_q"], row)
    axial = next((row for q, row in best.items() if q in AXES), None)
    gaps = {q: row["energy_per_ni"] - axial["energy_per_ni"] for q, row in best.items()} if axial else {}
    diag = {q: gaps.get(q) for q in DIAGONALS}
    return {"completed": len(finished), "planned": len(rows), "best_by_q": best, "axial_reference": axial,
            "diagonal_gaps": diag, "best_diagonal_gap": min((gap for gap in diag.values() if gap is not None), default=None),
            "guard_gaps": {q: gaps.get(q) for q in GUARDS}}


def plot(groups, output, chi):
    figure, (axis, guards) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    labels = []
    for x, (label, group) in enumerate(groups.items()):
        complete = group["completed"] == group["planned"]
        labels.append(f"{label}\n{group['completed']}/{group['planned']}")
        for q, offset, marker, color in zip(DIAGONALS, (-.12, .12), ("o", "^"), ("#4477aa", "#bb6633")):
            value = group["diagonal_gaps"][q]
            if value is not None:
                axis.scatter(x + offset, value, marker=marker, s=55, edgecolor=color, facecolor=color if complete else "none")
    for label, marker, color in [("(π/2,π/2)", "o", "#4477aa"), ("(π/2,−π/2)", "^", "#bb6633")]:
        axis.scatter([], [], marker=marker, color=color, label=label)
    axis.legend(frameon=False, loc="best")
    axis.set(xticks=range(len(labels)), xticklabels=labels, xlim=(-.5, len(labels)-.5), title="Diagonal branches")
    expanded = groups["Opposite −1\nD3 expanded"]
    selected_d4 = groups.get("Opposite −1\nD4 selected")
    has_d4_guard = selected_d4 and any(value is not None for value in selected_d4["guard_gaps"].values())
    for x, q in enumerate(GUARDS):
        value = expanded["guard_gaps"][q]
        if value is not None:
            face = "#555555" if expanded["completed"] == expanded["planned"] else "none"
            guards.scatter(x - .08 if has_d4_guard else x, value, s=50, edgecolor="#555555", facecolor=face)
        if has_d4_guard and selected_d4["guard_gaps"][q] is not None:
            color = "#228833"
            face = color if selected_d4["completed"] == selected_d4["planned"] else "none"
            guards.scatter(x + .08, selected_d4["guard_gaps"][q], marker="D", s=50,
                           edgecolor=color, facecolor=face)
    if has_d4_guard:
        guards.scatter([], [], s=50, color="#555555", label="D=3")
        guards.scatter([], [], marker="D", s=50, color="#228833", label="D=4 selected")
        guards.legend(frameon=False, loc="best")
    guards.set(xticks=range(len(GUARDS)), xticklabels=list(GUARDS.values()), xlim=(-.5, len(GUARDS)-.5),
               title="Opposite −1, other measured branches")
    if all(value is None for value in expanded["guard_gaps"].values()):
        guards.set(ylim=(-.05, .05), yticks=[0])
        guards.text(.5, .68, "No measured guard gaps", transform=guards.transAxes, ha="center")
    for panel in (axis, guards):
        panel.axhline(0, color="#555555", linewidth=.8)
        panel.grid(axis="y", alpha=.2)
        panel.set_ylabel("E/Ni − best axial E/Ni")
    figure.suptitle(f"Relative x-z hopping sign at table parameters (T=2.95, χ={chi})")
    figure.subplots_adjust(left=.09, right=.98, bottom=.23, top=.83, wspace=.31)
    figure.text(.5, .035, "Positive gaps favor the axial branch. Open markers use incomplete cohorts. Blank phases have no measured gap.",
                ha="center", fontsize=9)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"opposite_txz.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("legacy", "d3", "d4", "output"):
        parser.add_argument(f"--{option}", type=Path, required=option != "d4")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    legacy, physics, controls, legacy_source = read_run(args.legacy, "legacy", 3, 1, 8)
    d3, new_physics, new_controls, d3_source = read_run(args.d3, "opposite_d3", 3, -1, 32)
    chi = legacy_source["chi"]
    if d3_source["chi"] != chi:
        raise ValueError("Legacy and opposite cohorts must use the same χ")
    matched = {start(row) for row in legacy}
    if {start(row) for row in d3[:8]} != matched or (physics, controls) != (new_physics, new_controls):
        raise ValueError("Legacy and opposite cohorts differ in couplings, numerical controls or first-eight starts")
    sources = [legacy_source, d3_source]
    groups = {"Legacy +1\nD3 matched": summarize(legacy), "Opposite −1\nD3 matched": summarize(d3[:8]),
              "Opposite −1\nD3 expanded": summarize(d3)}
    rows = legacy + d3
    if args.d4:
        document = json.loads(args.d4.read_text())
        count = len(document.get("manifest", document)["cases"])
        if count not in (3, 4):
            raise ValueError("Selected D4 needs three or four branch starts")
        d4, d4_physics, d4_controls, d4_source = read_run(args.d4, "opposite_d4", 4, -1, count)
        if d4_source["chi"] != chi:
            raise ValueError("Selected D4 must use the same χ as D3")
        d3_starts = {row["case_id"]: start(row) for row in d3}
        if (physics, controls) != (d4_physics, d4_controls) or any(d3_starts.get(row["selected_d3_case"]) != start(row) for row in d4):
            raise ValueError("D4 must use the same couplings and its selected D3 initializers")
        groups["Opposite −1\nD4 selected"] = summarize(d4)
        sources.append(d4_source)
        rows += d4
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and any(args.output.glob("opposite_txz*")):
        parser.error("Output exists. Choose another directory or --overwrite.")
    plot(groups, args.output, chi)
    columns = ["dataset", "case_id", "matched_eight", *START, "selected_d3_case", "TxzYSign", "D", "actual_D", "time", "chi",
               "kind", "checkpoint_sha256", "status", "final_q", "top_q", "bottom_q", "energy_per_ni", "n_per_ni"]
    with (args.output / "opposite_txz_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(dict(row, matched_eight=row["D"] == 3 and start(row) in matched) for row in rows)
    old, new = list(groups.values())[:2]
    full_match = old["completed"] == new["completed"] == 8
    delta_old, delta_new = old["best_diagonal_gap"], new["best_diagonal_gap"]
    audit = {"sources": sources, "physics_without_sign": physics, "numerical_controls": controls, "chi": chi, "groups": groups,
             "matched_sign_effect_on_best_gap": delta_new - delta_old if full_match and None not in (delta_old, delta_new) else None}
    (args.output / "opposite_txz_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({label.replace("\n", " "): {key: group[key] for key in ("completed", "planned", "diagonal_gaps", "best_diagonal_gap")}
                      for label, group in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
