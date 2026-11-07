#!/usr/bin/env python3
"""Compare coupling-weighted energy terms for the best observed diagonal/axial states."""

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
DIAGONAL = {"pi2_pi2", "pi2_minus_pi2"}
AXIAL = {"0_pi", "pi_0"}
Q = {"pi2_pi2": "(π/2,π/2)", "pi2_minus_pi2": "(π/2,−π/2)", "0_pi": "(0,π)", "pi_0": "(π,0)"}
TOLERANCE = 1e-9
LABELS = {"hopping_x": r"$t_x$ hopping", "hopping_xz": r"$t_{xz}$ hopping",
          "exchange_x": r"$J_x$ exchange", "exchange_xz": r"$J_{xz}$ exchange",
          "orbital_energy": "Orbital energies", "density_v": r"$V$ density", "hund": "Hund",
          "interlayer_hopping": "Interlayer hopping", "interlayer_exchange": "Interlayer exchange",
          "chemical_potential": "Chemical potential", "total_physical": "Total physical"}


def checked(record):
    case, measurement = record["case"], record["measurement"]
    if case["physics"].get("Mu", 0) != 0:
        raise ValueError("Nonzero Mu is unsupported: grand-energy terms require an explicit chemical-potential correction.")
    components, closure = measurement["energy_components_per_ni"], measurement["energy_component_closure"]
    grand, physical = measurement["grand_energy_per_rung"] / 2, measurement["physical_energy_per_ni"]
    values = [*components.values(), *closure.values(), grand, physical]
    if not all(math.isfinite(value) for value in values) or not measurement.get("ctm_converged"):
        raise ValueError("Nonfinite energy terms or unconverged CTM: " + case["id"])
    if measurement["chi"] != case["algorithm"]["Chi"]:
        raise ValueError("Measurement Chi differs from manifest: " + case["id"])
    q = record["dominant_q"]
    if q in Q and tuple(record["layer_q"]) != (q, q):
        raise ValueError("Selected phase disagrees with measured layer phases: " + case["id"])
    residuals = [closure["grand_residual_per_ni"], closure["physical_residual_per_ni"],
                 closure["chemical_potential_correction_per_ni"], sum(components.values()) - grand,
                 sum(components.values()) - physical, closure["grand_sum_per_ni"] - grand,
                 closure["physical_sum_per_ni"] - physical]
    if any(abs(value) >= TOLERANCE for value in residuals):
        raise ValueError("Component grand/physical energy closure exceeds 1e-9/Ni: " + case["id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    raw = args.input.read_bytes(); document = json.loads(raw)
    manifest = document.get("manifest", document)
    if manifest.get("schema") != "lno.cent.followup.v1":
        raise ValueError("Expected a cent_followup collection or manifest")
    records = {record["case"]["id"]: record for record in document.get("records", [])}
    groups = {}
    for case in manifest["cases"]:
        algorithm = case["algorithm"]
        schedule = [] if case["kind"] == "measure" else algorithm["Schedule"]
        duration = case.get("start_total_time", 0) + sum(stage["Steps"] * stage["Tau"] for stage in schedule)
        key = (case["point"], algorithm["Dmax"], round(duration, 12), algorithm["Chi"])
        record = records.get(case["id"], {"case": case, "status": "not_started"})
        if record["case"] != case:
            raise ValueError("Result differs from manifest: " + case["id"])
        if record["status"] in COMPLETE:
            checked(record)
            if abs(record.get("planned_total_time", duration) - duration) >= TOLERANCE:
                raise ValueError("Result time differs from manifest: " + case["id"])
        groups.setdefault(key, []).append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and any(args.output.glob("energy_components*")):
        parser.error("Output exists; choose another directory or --overwrite")
    audit, rows = [], []
    for index, ((point, dimension, duration, chi), group) in enumerate(sorted(groups.items()), 1):
        if len({json.dumps(item["case"]["physics"], sort_keys=True) for item in group}) != 1:
            raise ValueError("Different Hamiltonians within a comparison group: " + point)
        physics = group[0]["case"]["physics"]
        sign = physics.get("TxzYSign", 1)
        if isinstance(sign, bool) or sign not in {-1, 1}:
            raise ValueError("TxzYSign must be -1 or +1: " + point)
        finished = [item for item in group if item["status"] in COMPLETE]
        if len({item["binary_sha256"] for item in finished}) > 1:
            raise ValueError("Different numerical binaries within a comparison group: " + point)
        if len({item["case"]["algorithm"]["CTMTolerance"] for item in group}) > 1:
            raise ValueError("Different CTM tolerances within a comparison group: " + point)
        choose = lambda phases: min((item for item in finished if item["dominant_q"] in phases),
                                    key=lambda item: item["measurement"]["physical_energy_per_ni"], default=None)
        diagonal, axial = choose(DIAGONAL), choose(AXIAL)
        info = {"point": point, "D": dimension, "time": duration, "chi": chi, "TxzYSign": sign,
                "physics": physics, "sign_from_legacy_default": "TxzYSign" not in physics,
                "completed": len(finished), "planned": len(group), "status": "complete_pair" if diagonal and axial else "missing_pair"}
        for name, item in [("diagonal", diagonal), ("axial", axial)]:
            info[name] = None if item is None else {"case_id": item["case"]["id"], "q": item["dominant_q"],
                "energy_per_ni": item["measurement"]["physical_energy_per_ni"], "binary_sha256": item["binary_sha256"],
                "checkpoint": {key: (item.get("checkpoint_identity") or {}).get(key) for key in ["path", "sha256"]},
                "closure": item["measurement"]["energy_component_closure"]}
        audit.append(info)
        base = {key: info[key] for key in ["point", "D", "time", "chi", "TxzYSign", "completed", "planned", "status"]}
        if not (diagonal and axial):
            rows.append(dict(base, term="total_physical", diagonal=None, axial=None, delta=None)); continue
        left, right = diagonal["measurement"]["energy_components_per_ni"], axial["measurement"]["energy_components_per_ni"]
        if left.keys() != right.keys():
            raise ValueError("Different component definitions in paired measurements")
        delta = {term: left[term] - right[term] for term in left}
        diagonal_energy, axial_energy = info["diagonal"]["energy_per_ni"], info["axial"]["energy_per_ni"]
        total = diagonal_energy - axial_energy
        if abs(sum(delta.values()) - total) >= TOLERANCE:
            raise ValueError("Energy-gap component closure exceeds 1e-9/Ni")
        component_order = [term for term in LABELS if term in left] + [term for term in left if term not in LABELS]
        for term in [*component_order, "total_physical"]:
            rows.append(dict(base, term=term, diagonal=left.get(term, diagonal_energy),
                             axial=right.get(term, axial_energy), delta=delta.get(term, total)))
        terms = [term for term in component_order if max(abs(left[term]), abs(right[term])) > 1e-14] + ["total_physical"]
        values = [delta.get(term, total) for term in terms]
        figure, axis = plt.subplots(figsize=(8.1, max(4.5, .36 * len(terms) + 1.6)))
        figure.subplots_adjust(left=.23, right=.96, bottom=.19, top=.84)
        bars = axis.barh(range(len(terms)), values, color=["#333333" if term == "total_physical" else "#dd8844" if value > 0 else "#4477aa" for term, value in zip(terms, values)])
        axis.bar_label(bars, fmt="%+.5f", padding=3, fontsize=8)
        axis.margins(x=.2)
        axis.set_yticks(range(len(terms)), [LABELS.get(term, term.replace("_", " ")) for term in terms]); axis.invert_yaxis()
        axis.axvline(0, color="#444444", linewidth=.8); axis.grid(axis="x", alpha=.2)
        axis.set_xlabel("δE/Ni = diagonal − axial   (positive favors axial)")
        sign_label = "Legacy same-sign" if sign == 1 else "Opposite-sign"
        figure.suptitle(f"{sign_label} x-z hopping: {point}, D={dimension}, T={duration:g}, χ={chi}\n"
                        f"qdiag={Q[diagonal['dominant_q']]} vs qaxis={Q[axial['dominant_q']]}   ΔE/Ni={total:+.6f}", fontsize=11)
        figure.text(.5, .035, "Frozen-state energy decomposition; no ground-state convergence claim.", ha="center", fontsize=9)
        stem = f"energy_components_{index:02d}_" + re.sub(r"[^A-Za-z0-9_-]", "_", point)
        for extension in ["png", "pdf"]: figure.savefig(args.output / f"{stem}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(figure)
    with (args.output / "energy_components.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["point", "D", "time", "chi", "TxzYSign", "completed", "planned", "status", "term", "diagonal", "axial", "delta"])
        writer.writeheader(); writer.writerows(rows)
    summary = {"input": str(args.input.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "closure_tolerance_per_ni": TOLERANCE,
               "selection": "Lowest measured final-q diagonal among ±, versus lowest measured (0,pi)/(pi,0); no missing values imputed.", "groups": audit}
    (args.output / "energy_components_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"groups": len(audit), "complete_pairs": sum(item["status"] == "complete_pair" for item in audit), "missing_pairs": sum(item["status"] == "missing_pair" for item in audit)}))


if __name__ == "__main__":
    main()
