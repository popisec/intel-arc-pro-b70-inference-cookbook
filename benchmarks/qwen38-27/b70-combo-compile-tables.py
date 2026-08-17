#!/usr/bin/env python3
"""Single-mode (MTP4 COMBO) compile + tables for the Qwen3.8 phase-separated
matrix. Mirrors the cookbook 4-mode compiler/render for the adopted COMBO
stack. Fails closed on wrong tokens, finish reason, or prefix-cache contamination.

Usage: python3 b70-combo-compile-tables.py RUN_ROOT
Outputs RUN_ROOT/summary.json and RUN_ROOT/tables.md
"""
import argparse, hashlib, json, statistics
from pathlib import Path

COORDS = {
    'prefill-p512': (512, 1, 'prefill'),
    'prefill-p2048': (2048, 1, 'prefill'),
    'prefill-p4096': (4096, 1, 'prefill'),
    'prefill-p6144': (6144, 1, 'prefill'),
    'prefill-p8192': (8192, 1, 'prefill'),
    'decode-p512-g32': (512, 32, 'decode_small_prompt'),
    'decode-p512-g128': (512, 128, 'decode_small_prompt'),
    'decode-p512-g256': (512, 256, 'decode_small_prompt'),
    'decode-p512-g512': (512, 512, 'decode_small_prompt'),
    'decode-p8192-g32': (8192, 32, 'decode'),
    'decode-p8192-g128': (8192, 128, 'decode'),
    'decode-control-p9445-g128': (9445, 128, 'decode_historical_control'),
    'decode-p8192-g256': (8192, 256, 'decode'),
    'decode-p8192-g512': (8192, 512, 'decode'),
    'prefill-full-p131071': (131071, 1, 'prefill_full_context'),
    'decode-full-p130944-g128': (130944, 128, 'decode_full_context'),
    'decode-full-p130560-g512': (130560, 512, 'decode_full_context'),
}
PREFILL = ['prefill-p512', 'prefill-p2048', 'prefill-p4096', 'prefill-p6144', 'prefill-p8192']


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"n": len(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "min": min(values), "max": max(values),
            "pstdev": statistics.pstdev(values) if len(values) > 1 else 0.0}


def text_hash(rec):
    payload = (rec.get("reasoning_text") or "") + "\0" + (rec.get("content_text") or "")
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", type=Path)
    args = ap.parse_args()
    root = args.run_root.resolve()
    manifest = dict(line.split("=", 1) for line in (root / "manifest.txt").read_text().splitlines()
                    if "=" in line)
    rows = []
    for coord, (ptok, otok, cls) in COORDS.items():
        path = root / coord / "results.json"
        if not path.exists():
            raise RuntimeError(f"missing cell: {coord}")
        data = json.loads(path.read_text())
        records = data["records"]
        if len(records) != 5:
            raise RuntimeError(f"{coord}: n={len(records)}")
        for rec in records:
            if rec["prompt_tokens"] != ptok:
                raise RuntimeError(f"{coord}: prompt {rec['prompt_tokens']} != {ptok}")
            if rec["completion_tokens"] != otok:
                raise RuntimeError(f"{coord}: output {rec['completion_tokens']} != {otok}")
            if rec["finish_reason"] != "length":
                raise RuntimeError(f"{coord}: finish={rec['finish_reason']}")
            if rec["prefix_cache_hits_delta"] != 0:
                raise RuntimeError(f"{coord}: cache contamination delta={rec['prefix_cache_hits_delta']}")
        proposed = sum(r.get("mtp_proposed_tokens") or 0 for r in records)
        accepted = sum(r.get("mtp_accepted_tokens") or 0 for r in records)
        post = [r.get("client_post_first_tps") for r in records]
        rows.append({
            "mode": "mtp4", "coordinate": coord, "class": cls,
            "prompt_tokens": ptok, "requested_output_tokens": otok,
            "samples": 5, "ignore_eos": data.get("ignore_eos", False),
            "ttft_s": stats([r["ttft_s"] for r in records]),
            "ttfc_s": stats([r.get("ttfc_s") for r in records]),
            "e2e_s": stats([r["total_s"] for r in records]),
            "input_tokens_per_ttft_s": stats([r["input_tokens_per_ttft_s"] for r in records]),
            "client_post_first_tps": stats(post),
            "client_post_first_tpot_ms": stats([1000.0 / v for v in post if v]),
            "prefix_cache_hits_delta": sum(r["prefix_cache_hits_delta"] for r in records),
            "prefix_cache_queries_delta": sum(r["prefix_cache_queries_delta"] for r in records),
            "mtp_proposed_tokens": proposed, "mtp_accepted_tokens": accepted,
            "mtp_acceptance_pct": (100.0 * accepted / proposed) if proposed else None,
            "output_sha256_by_rep": [text_hash(r) for r in records],
        })
    summary = {
        "schema": 1, "run_id": root.name,
        "status": "E2_PROVISIONAL_SELF_REPORTED_NOT_INDEPENDENTLY_REPRODUCED",
        "scope": "C1 phase-separated vLLM serving; medians of five measured requests per cell; single mode MTP4 COMBO",
        "timing_source": "client monotonic SSE timestamps",
        "prefill_metric_warning": "input_tokens_per_ttft_s includes scheduling + first-token work; not engine-native prefill",
        "decode_metric_warning": "client post-first rate is request-side, not engine-native",
        "cache": "prefix caching enabled; unique entropy-first cold prompts; zero hit delta",
        "configured_power_cap_W": int(manifest.get("cap_W", 230)),
        "scheduler_budget": int(manifest.get("scheduler_budget", 4096)),
        "max_model_len": int(manifest.get("max_model_len", 131328)),
        "excluded": [],
        "mode": "mtp4", "num_speculative_tokens": 4,
        "stack": {
            "image": manifest.get("image"),
            "model": manifest.get("model_id", "unknown"),
            "vllm": "0.26.1rc1.dev457+gc810e5ee9.xpu",
            "vllm_xpu_kernels": "0.1.12",
            "patches": "patch_mtp_nightly.py,patch_mtp_boundary.py,patch_mtp_ptr_wrap.py,patch_gdn_split_mixed.py,patch_draft_lmhead_int4.py,patch_draft_mtp_int4.py",
        },
        "rows": rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    row = {r["coordinate"]: r for r in rows}
    def med(c, f):
        v = row[c][f]
        if isinstance(v, dict):
            return v["median"] if v else None
        return v
    def fmt(v, d=2): return "n/a" if v is None else f"{v:,.{d}f}"
    L = [
        "## Phase-separated vLLM benchmark — Qwen3.8-27B COMBO (gptqmodel quant on the 5-patch stack)",
        "",
        f"Tested stack: vLLM `{summary['stack']['vllm']}`, `vllm-xpu-kernels {summary['stack']['vllm_xpu_kernels']}`, "
        f"C1, `n=5`, scheduler budget {summary['scheduler_budget']}, context {summary['max_model_len']}, fp8 KV, "
        f"prefix cache enabled with zero hit delta, configured {summary['configured_power_cap_W']} W cap, "
        f"MTP4 with S+M1 draft-INT4 patches. Status: E2 self-reported.",
        "",
        "### Cold input rate (actual input tokens / TTFT, tok/s)",
        "",
        "| Mode | p512 | p2048 | p4096 | p6144 | p8192 | Full p131071 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    vals = [fmt(med(c, "input_tokens_per_ttft_s"), 0) for c in PREFILL]
    vals.append(fmt(med("prefill-full-p131071", "input_tokens_per_ttft_s"), 0))
    L.append("| MTP4 COMBO | " + " | ".join(vals) + " |")
    for p in (512, 8192):
        L += ["", f"### Decode at p{p} (client post-first tok/s)", "",
              "| Mode | g32 | g128 | g256 | g512 |", "|---|---:|---:|---:|---:|"]
        v = [fmt(med(f"decode-p{p}-g{o}", "client_post_first_tps")) for o in (32, 128, 256, 512)]
        L.append("| MTP4 COMBO | " + " | ".join(v) + " |")
    L += ["", "### Matched historical control (p9445/g128)", "", "| Mode | Client post-first median (tok/s) |", "|---|---:|"]
    L.append(f"| MTP4 COMBO | {fmt(med('decode-control-p9445-g128', 'client_post_first_tps'))} |")
    L += ["", "### Full-context decode", "",
          "| Mode | p130944/g128 (tok/s) | MTP accept | p130560/g512 (tok/s) | MTP accept |", "|---|---:|---:|---:|---:|"]
    g1, g2 = row["decode-full-p130944-g128"], row["decode-full-p130560-g512"]
    a1 = f"{g1['mtp_acceptance_pct']:.1f}%" if g1["mtp_acceptance_pct"] is not None else "n/a"
    a2 = f"{g2['mtp_acceptance_pct']:.1f}%" if g2["mtp_acceptance_pct"] is not None else "n/a"
    L.append(f"| MTP4 COMBO | {fmt(med('decode-full-p130944-g128', 'client_post_first_tps'))} | {a1} | {fmt(med('decode-full-p130560-g512', 'client_post_first_tps'))} | {a2} |")
    L += ["",
          "Input rate includes request scheduling and first-token work; decode is client-observed, not engine-native. "
          "Exact output rows use the requested completion length.",
          "",
          "### Detailed per-cell (medians, n=5)",
          "",
          "| Cell | TTFT (s) | E2E (s) | input/TTFT (tok/s) | post-first (tok/s) | TPOT (ms) | MTP accept % |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for c in COORDS:
        r = row[c]
        L.append(f"| {c} | {fmt(med(c, 'ttft_s'))} | {fmt(med(c, 'e2e_s'))} | "
                 f"{fmt(med(c, 'input_tokens_per_ttft_s'), 0)} | {fmt(med(c, 'client_post_first_tps'))} | "
                 f"{fmt(med(c, 'client_post_first_tpot_ms'), 1)} | {fmt(med(c, 'mtp_acceptance_pct'))}% |")
    (root / "tables.md").write_text("\n".join(L) + "\n")
    print("summary.json + tables.md written to", root)


if __name__ == "__main__":
    main()