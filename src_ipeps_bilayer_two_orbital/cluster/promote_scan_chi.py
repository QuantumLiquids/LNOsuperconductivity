#!/usr/bin/env python3
"""Promote a stopped Chi8 scan to matched Chi16 measurements, preserving SU.

First withdraw only pending original Slurm tasks. `cold` records which original
cases already exist and prepares the remaining cold starts. Once those existing
cases reach their final checkpoint, `reuse` validates their history and freezes
their checkpoint hashes. The first point is covered by a separate rescue run.
No Slurm jobs are cancelled or submitted by this helper.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

from cent_followup import checksum, checkpoint_identity, effective_algorithm, read_json, validate_manifest


def write_new(path, value):
    if path.exists():
        raise FileExistsError("Refusing to replace " + str(path))
    path.write_text(json.dumps(value, indent=2)+"\n")


def new_case(old, chi, deployment, campaign, reuse):
    case = copy.deepcopy(old)
    case["id"] += f"_chi{chi}"
    case["algorithm"]["Chi"] = chi
    case["algorithm"]["CTMTolerance"] = 1e-6
    case["algorithm"]["MaxCTMSweeps"] = 200
    if not reuse:
        return case
    source = deployment/"runs"/campaign/old["id"]
    algorithm = effective_algorithm(old)
    source_record = read_json(source/"result.json")
    binary_hash = checksum(deployment/"build/ipeps_bilayer_simple_update_u1")
    if read_json(source/"physics.json") != {"CaseParams": old["physics"]} or \
            read_json(source/"algorithm.json") != {"CaseParams": algorithm} or \
            source_record["case"] != old or source_record.get("binary_sha256") != binary_hash:
        raise ValueError("Original input/history differs: " + old["id"])
    latest = read_json(source/"state/latest_checkpoint.json")
    checkpoint = source/"state/checkpoint_step00000040"
    match = re.search(r"SU step=40 tau=([0-9eE+.\-]+)", (source/"solver.log").read_text())
    if latest["step"] != 40 or Path(latest["checkpoint"]).resolve() != checkpoint.resolve() or \
            not match or abs(float(match[1])-.03) > 1e-12:
        raise ValueError("Original has not finished all 40 SU steps: " + old["id"])
    identity = checkpoint_identity(checkpoint, algorithm)
    case.update(kind="measure", checkpoint=str(checkpoint), start_total_time=2.95,
                source_last_tau=.03, expected_checkpoint_sha256=identity["sha256"],
                source_history={"root": str(source), "original_case_id": old["id"],
                                "effective_algorithm": algorithm, "binary_sha256": binary_hash,
                                "final_step": 40, "final_tau": .03})
    case["algorithm"]["Schedule"] = []
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["cold", "reuse"])
    parser.add_argument("deployment", type=Path)
    parser.add_argument("original_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chi", type=int, default=16)
    args = parser.parse_args()
    original = validate_manifest(read_json(args.original_manifest))
    for c in original["cases"]:
        if c["kind"] != "evolve" or c.get("start_total_time", 0) != 0 or \
                c["algorithm"]["Dmax"] != 3 or c["algorithm"]["Schedule"] != [
                    {"Tau": .1, "Steps": 25}, {"Tau": .03, "Steps": 15}]:
            raise ValueError("Expected the original common cold D3 T2.95 protocol")
    if args.chi < 1:
        raise ValueError("Chi must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    selection_path = args.output/"chi_promotion_selection.json"
    original_hash = hashlib.sha256(args.original_manifest.read_bytes()).hexdigest()
    if args.phase == "cold":
        cold, reuse = [], []
        for c in original["cases"]:
            if c["point"] == "near_lam000_s100":
                continue  # Separate all-eight-state Chi16 rescue manifest.
            target = reuse if (args.deployment/"runs"/original["campaign"]/c["id"]).exists() else cold
            target.append(c["id"])
        selection = {"original_manifest_sha256": original_hash, "chi": args.chi,
                     "cold_ids": cold, "reuse_ids": reuse}
        write_new(selection_path, selection)
        ids = cold
    else:
        selection = read_json(selection_path)
        if selection["original_manifest_sha256"] != original_hash or selection["chi"] != args.chi:
            raise ValueError("Original manifest or target Chi differs from saved promotion")
        ids = selection["reuse_ids"]
    lookup = {c["id"]: c for c in original["cases"]}
    cases = [new_case(lookup[identifier], args.chi, args.deployment, original["campaign"], args.phase == "reuse") for identifier in ids]
    manifest = {"schema": "lno.cent.followup.v1",
                "campaign": f"nonly_constrained_j_d3_chi{args.chi}_{args.phase}_20261105",
                "description": "Common Chi16 readout; original cold SU history remains .1x25+.03x15. First point supplied by separate rescue.",
                "original_manifest_sha256": original_hash, "cases": cases}
    validate_manifest(manifest)
    path = args.output/f"chi{args.chi}_{args.phase}_manifest.json"
    write_new(path, manifest)
    print(json.dumps({"manifest": str(path), "cases": len(cases),
                      "cold": len(selection["cold_ids"]), "reuse": len(selection["reuse_ids"])}))


if __name__ == "__main__":
    main()
