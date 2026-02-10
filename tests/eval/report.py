"""Standalone HTML report generator for WeatherWise.ai evaluation results.

Produces a self-contained .html file (inline CSS, no external deps)
that can be opened in any browser and shared via Slack/email.

Usage (standalone)::

    from report import generate_html_report
    generate_html_report(all_results, "reports/eval_report.html")

Or via CLI::

    python tests/eval/model_comparison.py --html reports/eval_report.html
"""

from __future__ import annotations

import html
import os
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from model_comparison import ModelResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """HTML-escape user-facing text."""
    return html.escape(str(text))


def _badge(passed: bool, label: str = "") -> str:
    """Return a colored PASS/FAIL badge."""
    if passed:
        color, text = "#16a34a", label or "PASS"
    else:
        color, text = "#dc2626", label or "FAIL"
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f'font-size:0.75rem;font-weight:600;color:#fff;background:{color}">'
        f"{_esc(text)}</span>"
    )


def _bar(score: int, max_score: int = 5) -> str:
    """Render a small horizontal score bar."""
    pct = int(score / max_score * 100)
    hue = int(score / max_score * 120)  # 0=red, 120=green
    return (
        f'<div style="display:flex;align-items:center;gap:6px">'
        f'<div style="flex:1;height:8px;background:#e5e7eb;border-radius:4px;overflow:hidden">'
        f'<div style="width:{pct}%;height:100%;background:hsl({hue},70%,45%);border-radius:4px"></div>'
        f"</div>"
        f'<span style="font-size:0.8rem;font-weight:600;min-width:28px">{score}/{max_score}</span>'
        f"</div>"
    )


def _tag(label: str, is_must: bool) -> str:
    """Render a MUST/SHOULD tag."""
    bg = "#7c3aed" if is_must else "#6366f1"
    return (
        f'<span style="display:inline-block;padding:1px 6px;border-radius:3px;'
        f'font-size:0.65rem;font-weight:600;color:#fff;background:{bg};margin-right:4px">'
        f"{_esc(label)}</span>"
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_summary_table(all_results: Dict[str, list], models: List[str], buckets: List[str]) -> str:
    """Section 1: Score summary table with PASS/FAIL badges."""
    cols = "".join(f"<th>{_esc(m)}</th>" for m in models)
    rows = ""

    model_totals: Dict[str, List[float]] = {m: [] for m in models}
    for i, bucket in enumerate(buckets):
        cells = ""
        for m in models:
            r = all_results[m][i]
            score = r.verdict.overall_score
            pf = r.verdict.pass_fail == "pass"
            model_totals[m].append(score)
            cells += f"<td>{score:.2f} {_badge(pf)}</td>"
        rows += f"<tr><td><strong>{_esc(bucket)}</strong></td>{cells}</tr>"

    # Average row
    avg_cells = ""
    for m in models:
        avg = sum(model_totals[m]) / len(model_totals[m]) if model_totals[m] else 0
        avg_cells += f"<td><strong>{avg:.2f}</strong></td>"
    rows += f'<tr style="border-top:2px solid #374151"><td><strong>AVERAGE</strong></td>{avg_cells}</tr>'

    return (
        f"<h2>Summary</h2>"
        f'<table><thead><tr><th>Bucket</th>{cols}</tr></thead>'
        f"<tbody>{rows}</tbody></table>"
    )


def _build_dimensions(all_results: Dict[str, list], models: List[str], buckets: List[str]) -> str:
    """Section 2: Dimension breakdown with score bars."""
    sections = ""
    for m in models:
        inner = ""
        for i, bucket in enumerate(buckets):
            r = all_results[m][i]
            dim_rows = ""
            for ds in r.verdict.dimension_scores:
                dim_rows += (
                    f"<tr><td>{_esc(ds.dimension)}</td>"
                    f"<td style='width:200px'>{_bar(ds.score)}</td>"
                    f"<td style='color:#6b7280;font-size:0.8rem'>{_esc(ds.reasoning)}</td></tr>"
                )
            inner += (
                f"<details><summary><strong>{_esc(bucket)}</strong></summary>"
                f"<table>{dim_rows}</table></details>"
            )
        sections += f"<h3>{_esc(m)}</h3>{inner}"
    return f"<h2>Dimension Breakdown</h2>{sections}"


def _build_assertions(all_results: Dict[str, list], models: List[str], buckets: List[str]) -> str:
    """Section 3: Code assertion results."""
    sections = ""
    for m in models:
        inner = ""
        for i, bucket in enumerate(buckets):
            r = all_results[m][i]
            if not r.assertions:
                continue
            items = "".join(
                f"<div style='margin:4px 0'>{_badge(a.passed)} "
                f"<code>{_esc(a.name)}</code>: {_esc(a.detail)}</div>"
                for a in r.assertions
            )
            inner += (
                f"<details><summary><strong>{_esc(bucket)}</strong></summary>"
                f"<div style='padding:8px 0'>{items}</div></details>"
            )
        sections += f"<h3>{_esc(m)}</h3>{inner}"
    return f"<h2>Code Assertions</h2>{sections}"


def _build_binary(all_results: Dict[str, list], models: List[str], buckets: List[str]) -> str:
    """Section 4: Binary criteria pass/fail."""
    sections = ""
    for m in models:
        inner = ""
        for i, bucket in enumerate(buckets):
            r = all_results[m][i]
            bv = r.binary_verdict
            if bv is None:
                inner += (
                    f"<details><summary><strong>{_esc(bucket)}</strong> "
                    f"{_badge(False, 'SKIPPED')}</summary></details>"
                )
                continue
            summary_badge = _badge(bv.overall_pass)
            must_text = f"Must: {bv.must_pass_count} pass / {bv.must_fail_count} fail"
            should_text = f"Should: {bv.should_pass_count}/{bv.should_total_count}"
            criteria_rows = ""
            for cr in bv.criteria_results:
                tag_label = "MUST" if cr.is_must else "SHOULD"
                criteria_rows += (
                    f"<tr><td>{_tag(tag_label, cr.is_must)} {_badge(cr.passed)}</td>"
                    f"<td>{_esc(cr.criterion)}</td>"
                    f"<td style='color:#6b7280;font-size:0.8rem'>{_esc(cr.reasoning)}</td></tr>"
                )
            inner += (
                f"<details><summary><strong>{_esc(bucket)}</strong> {summary_badge} "
                f"<span style='font-size:0.8rem;color:#6b7280'>{must_text} | {should_text}</span>"
                f"</summary><table>{criteria_rows}</table></details>"
            )
        sections += f"<h3>{_esc(m)}</h3>{inner}"
    return f"<h2>Binary Criteria</h2>{sections}"


def _build_pairwise(pairwise_results: List[Dict]) -> str:
    """Section 5: Pairwise comparison table."""
    if not pairwise_results:
        return ""

    model_a = pairwise_results[0]["model_a"]
    model_b = pairwise_results[0]["model_b"]
    wins: Dict[str, int] = {model_a: 0, model_b: 0, "tie": 0}

    rows = ""
    for pr in pairwise_results:
        v = pr["verdict"]
        if v.winner == "model_a":
            winner_name = model_a
            wins[model_a] += 1
        elif v.winner == "model_b":
            winner_name = model_b
            wins[model_b] += 1
        else:
            winner_name = "Tie"
            wins["tie"] += 1
        rows += (
            f"<tr><td>{_esc(pr['bucket'])}</td>"
            f"<td><strong>{_esc(winner_name)}</strong></td>"
            f"<td>{v.confidence:.1f}</td>"
            f"<td style='color:#6b7280;font-size:0.8rem'>{_esc(v.reasoning)}</td></tr>"
        )

    tally = (
        f"<strong>{_esc(model_a)}</strong>: {wins[model_a]} &nbsp;|&nbsp; "
        f"<strong>{_esc(model_b)}</strong>: {wins[model_b]} &nbsp;|&nbsp; "
        f"Ties: {wins['tie']}"
    )
    return (
        f"<h2>Pairwise: {_esc(model_a)} vs {_esc(model_b)}</h2>"
        f"<p style='font-size:1.1rem'>{tally}</p>"
        f"<table><thead><tr><th>Bucket</th><th>Winner</th>"
        f"<th>Confidence</th><th>Reasoning</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #0f172a; color: #e2e8f0; padding: 32px; max-width: 1100px; margin: auto; }
h1 { font-size: 1.6rem; margin-bottom: 4px; }
h2 { font-size: 1.2rem; margin: 28px 0 12px; padding-bottom: 6px; border-bottom: 1px solid #334155; }
h3 { font-size: 1rem; margin: 16px 0 8px; color: #94a3b8; }
table { width: 100%; border-collapse: collapse; margin: 8px 0; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #1e293b; }
th { background: #1e293b; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; }
details { margin: 4px 0; }
summary { cursor: pointer; padding: 6px 0; }
summary:hover { color: #60a5fa; }
code { background: #1e293b; padding: 2px 6px; border-radius: 3px; font-size: 0.8rem; }
.meta { color: #64748b; font-size: 0.85rem; margin-bottom: 24px; }
"""


def generate_html_report(
    all_results: Dict[str, list],
    output_path: str,
    pairwise_results: Optional[List[Dict]] = None,
) -> str:
    """Generate a self-contained HTML evaluation report.

    Args:
        all_results: Mapping of model_key -> List[ModelResult].
        output_path: File path to write the HTML report.
        pairwise_results: Optional pairwise comparison results.

    Returns:
        The absolute path of the written report file.
    """
    models = list(all_results.keys())
    # Derive buckets from the first model's results
    buckets = [r.golden.bucket for r in all_results[models[0]]]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    body = (
        f"<h1>WeatherWise.ai Evaluation Report</h1>"
        f'<p class="meta">Generated {timestamp} &nbsp;|&nbsp; '
        f"Models: {', '.join(_esc(m) for m in models)} &nbsp;|&nbsp; "
        f"{len(buckets)} buckets</p>"
    )
    body += _build_summary_table(all_results, models, buckets)
    body += _build_dimensions(all_results, models, buckets)
    body += _build_assertions(all_results, models, buckets)
    body += _build_binary(all_results, models, buckets)
    if pairwise_results:
        body += _build_pairwise(pairwise_results)

    html_doc = (
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>WeatherWise.ai Eval Report — {timestamp}</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    abs_path = os.path.abspath(output_path)
    print(f"HTML report written to: {abs_path}")
    return abs_path
