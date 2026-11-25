"""Local evidence viewer with review notes kept outside the immutable bundle."""

from __future__ import annotations

import html
import json
import secrets
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer, ThreadingMixIn
from urllib.parse import parse_qs, unquote, urlsplit

from .pipeline import verify
from .review import MAX_REVIEW_FORM_BYTES, append_review, read_reviews


class LocalHTTPServer(ThreadingMixIn, TCPServer):
    daemon_threads = True
    allow_reuse_address = True

    @property
    def server_port(self) -> int:
        return self.server_address[1]


def build_server(bundle: Path, host: str = "127.0.0.1", port: int = 8765) -> LocalHTTPServer:
    research_manifest = bundle / "bundle_manifest.json"
    if research_manifest.is_file():
        candidate = json.loads(research_manifest.read_text(encoding="utf-8"))
        if isinstance(candidate, dict) and candidate.get("schema_version") in ("0.4", "0.5-algorithm", "0.6-algorithm", "0.7-project"):
            from .research_view import build_research_server
            return build_research_server(bundle, host, port)
    if host not in ("127.0.0.1", "::1"):
        raise ValueError("review viewer binds to loopback only")
    bundle = bundle.resolve()
    errors = verify(bundle)
    if errors:
        raise ValueError("bundle verification failed: " + "; ".join(errors))
    manifest_name = "bundle_manifest.json" if (bundle / "bundle_manifest.json").exists() else "manifest.json"
    manifest = json.loads((bundle / manifest_name).read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in ("0.2", "0.3"):
        raise ValueError("viewer requires an integrated schema 0.2 or 0.3 bundle")
    if manifest.get("schema_version") == "0.3":
        read_reviews(bundle)
    csrf_token = secrets.token_urlsafe(32)
    allowed = set(manifest["artifacts"])
    finance = json.loads((bundle / "finance_memo.json").read_text(encoding="utf-8"))
    benchmark = json.loads((bundle / "benchmark.json").read_text(encoding="utf-8"))
    validation = json.loads((bundle / "validation.json").read_text(encoding="utf-8"))
    governance = json.loads((bundle / "governance.json").read_text(encoding="utf-8"))
    replay = json.loads((bundle / "stress_replay.json").read_text(encoding="utf-8"))
    graph = (json.loads((bundle / "graphsage_evidence.json").read_text(encoding="utf-8"))
             if (bundle / "graphsage_evidence.json").exists() else None)
    import_profile = (json.loads((bundle / "import_profile.json").read_text(encoding="utf-8"))
                      if (bundle / "import_profile.json").exists() else None)
    impact = (json.loads((bundle / "decision_impact.json").read_text(encoding="utf-8"))
              if (bundle / "decision_impact.json").exists() else None)
    release_status = (json.loads((bundle / "release_gate.json").read_text(encoding="utf-8"))["status"]
                      if manifest_name == "bundle_manifest.json" else "not evaluated")

    def page() -> bytes:
        reviews = read_reviews(bundle) if manifest_name == "bundle_manifest.json" else []
        review_rows = "".join(
            f"<li><strong>{html.escape(item['decision'].replace('_', ' '))}</strong> by "
            f"{html.escape(item['reviewer'])} <time>{html.escape(item['reviewed_utc'])}</time>"
            f"<p>{html.escape(item['rationale'])}</p></li>"
            for item in reviews
        ) or "<li>No review recorded for this bundle.</li>"
        rows = "".join(
            "<tr><th scope='row'>" + html.escape(name) + "</th><td>" +
            f"{model['metrics']['pr_auc']:.4f}</td><td>{model['metrics']['f1_at_0_5']:.4f}</td><td>" +
            f"{model['metrics']['observed_fpr']:.4f}</td><td>" +
            f"{validation['error_analysis'][name]['false_positives']}</td></tr>"
            for name, model in benchmark["models"].items()
        )
        stress = "".join(
            "<li><strong>" + html.escape(name.replace("_", " ")) + "</strong>: " +
            html.escape(item["status"]) + f" (AP drop {item['observed_pr_auc_drop']:.4f})</li>"
            for name, item in replay["scenarios"].items()
        )
        links = "".join(
            f"<li><a href='/api/artifacts/{html.escape(name, quote=True)}'>{html.escape(name)}</a></li>"
            for name in sorted(allowed)
        )
        graph_panel = (f"""<section><h2>Optional GraphSAGE experiment</h2><p>Held-out average precision: <strong>{graph['metrics']['pr_auc']:.4f}</strong>; F1 at 0.5: <strong>{graph['metrics']['f1_at_0_5']:.4f}</strong> on {len(graph['predictions'])} synthetic transactions.</p><p>Uses the same test IDs but a stricter training-only graph context than the history-feature baseline. This is not a controlled model ranking or real-world fraud result. <a href="/api/artifacts/graphsage_evidence.json">Inspect raw edge scores and training trace</a>.</p></section>""") if graph else ""
        input_panel = (f"""<section><h2>Imported transaction source</h2><p>{html.escape(import_profile['source_name'])}: {import_profile['row_count']} synthetic transactions, {import_profile['positive_count']} positive labels. Declared license: <code>{html.escape(import_profile['license_id_operator_declared'])}</code>.</p><p>Source and reuse rights were declared by the operator and have not been independently authenticated. <a href="/api/artifacts/import_profile.json">Inspect the input profile</a>.</p></section>""") if import_profile else ""
        stability_note = ("Five-seed synthetic stability is saved in validation.json." if validation["stability_runs"] else
                          "No repeated-seed estimate applies to this imported file; see validation.json.")
        impact_rows = "".join(
            "<tr><th scope='row'>" + html.escape(name) + "</th>" +
            (f"<td>{model['heldout_tpr']:.3f}</td><td>{model['heldout_fpr']:.3f}</td>"
             f"<td>{model['scenarios'][0]['expected_alerts_per_period']:.1f}</td>"
             f"<td>{model['scenarios'][0]['expected_false_alerts_per_period']:.1f}</td>"
             f"<td>{model['scenarios'][0]['expected_missed_positives_per_period']:.1f}</td>"
             f"<td>{html.escape(model['scenarios'][0]['capacity_status'].replace('_', ' '))}</td></tr>"
             if model['availability'] == 'available' else "<td colspan='6'>Unavailable: single-class holdout</td></tr>")
            for name, model in impact["models"].items()
        ) if impact else ""
        impact_panel = (f"""<section><h2>Decision impact at a fixed threshold</h2><p>The threshold was selected on training data. The table uses held-out TPR/FPR and the <strong>illustrative 1% prevalence, 1,000 transactions, 100-review capacity</strong> scenario. It is not a real-world cost forecast or release approval.</p><div style="overflow-x:auto"><table><thead><tr><th>Model</th><th>Held-out TPR</th><th>Held-out FPR</th><th>Expected alerts</th><th>False alerts</th><th>Missed cases</th><th>Capacity</th></tr></thead><tbody>{impact_rows}</tbody></table></div><p><a href="/api/artifacts/decision_impact.json">Inspect all three assumptions, confusion counts, uncertainty intervals and calculations</a>.</p></section>""") if impact else ""
        review_panel = (f"""<section class="review"><h2>Record a review</h2><p>Decide whether the saved demonstration evidence is usable for review. This record does not authorize public or production release.</p>
<form method="post" action="/api/reviews"><input type="hidden" name="csrf" value="{csrf_token}">
<label>Reviewer name<input name="reviewer" required minlength="2" maxlength="80"></label>
<label>Decision<select name="decision"><option value="request_changes">Request changes</option><option value="accept_demo_evidence">Accept demo evidence</option><option value="reject_evidence">Reject evidence</option></select></label>
<label>Reason<textarea name="rationale" required minlength="10" maxlength="2000" rows="4" placeholder="Explain the evidence checked and any limits."></textarea></label>
<button type="submit">Save review record</button></form><h3>Review history</h3><ol>{review_rows}</ol><p><a href="/api/reviews">Download review records</a>. Records are stored beside this run and linked to its manifest hash.</p></section>""") if manifest_name == "bundle_manifest.json" else ""
        document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Governance Evidence Bundle — {html.escape(manifest['run_id'])}</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;background:#eaf0f3;color:#143046;margin:0}}main{{max-width:1080px;margin:auto;padding:1.5rem}}header{{background:#102f46;color:white;border-bottom:6px solid #1d6863}}h1,h2,h3{{font-family:Georgia,serif;line-height:1.18}}h1{{font-size:clamp(2rem,5vw,3.5rem);max-width:18ch;margin:.35rem 0}}.eyebrow{{font-size:.85rem;color:#b7d9d2}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}}section{{background:#fff;border-top:3px solid #1d6863;padding:1.4rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #cad7dc;text-align:left;padding:.5rem}}.warn{{color:#9e3f32;font-weight:700}}a{{color:#075b87}}a:focus-visible,button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{{outline:3px solid #9a6312;outline-offset:2px}}code{{overflow-wrap:anywhere}}footer{{margin:2rem 0}}form{{display:grid;gap:1rem;max-width:680px}}label{{display:grid;gap:.3rem;font-weight:600}}input,select,textarea{{font:inherit;padding:.65rem;border:1px solid #8da7b1;background:white}}button{{font:inherit;background:#1d6863;color:white;border:0;padding:.7rem 1rem;justify-self:start;cursor:pointer}}.review ol{{padding-left:1.5rem}}.review li{{border-bottom:1px solid #cad7dc;padding:.5rem 0}}summary{{cursor:pointer;font-weight:700}}@media(max-width:600px){{main{{padding:1rem}}section{{padding:1rem}}table{{font-size:.84rem}}}}</style></head>
<body><header><main><div class="eyebrow">Verified local run</div><h1>Governance Evidence Bundle</h1><p>Run <code>{html.escape(manifest['run_id'])}</code> · Version <code>{html.escape(manifest['project_version'])}</code></p></main></header><main>
<p class="warn">Synthetic benchmark. No real fraud-performance or regulatory approval claim.</p>
<div class="grid"><section><h2>Financial source</h2><p>{html.escape(finance['answer'])}</p><p>SEC filing accession <code>{html.escape(finance['accession'])}</code>. <a href="/api/artifacts/finance_memo.json">Open exact citations</a>.</p></section>
<section><h2>Governance decision</h2><p>Data-quality gate: <strong>{html.escape(governance['gate']['status'].upper())}</strong></p><p>Overall release gate: <strong>{html.escape(release_status.upper())}</strong>. This demonstration does not authorize production release.</p></section></div>
{input_panel}<section><h2>Synthetic benchmark</h2><div style="overflow-x:auto"><table><thead><tr><th>Model</th><th>Average precision</th><th>F1 at 0.5</th><th>Held-out FPR</th><th>False alerts @0.5</th></tr></thead><tbody>{rows}</tbody></table></div><p>All predictions are available as raw JSON. {stability_note}</p></section>
{impact_panel}
<section><h2>Frozen-model stress replay</h2><ul>{stress}</ul><p>Both scenarios rescore the same fitted graph-feature logistic model. Thresholds are illustrative.</p></section>
{graph_panel}
{review_panel}<section><h2>Evidence artifacts</h2><p><a href="/api/manifest">Open manifest</a>. Files and supported cross-file relations are checked for each request.</p><details><summary>Browse all {len(allowed)} files</summary><ul>{links}</ul></details></section>
<footer>Local evidence viewer · {html.escape(manifest['source_classification'])}</footer></main></body></html>"""
        return document.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def valid_host(self) -> bool:
            port = self.server.server_port
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def verified(self) -> bool:
            if verify(bundle):
                self.send_error(409, "bundle or journal verification failed")
                return False
            return True

        def do_GET(self) -> None:
            if not self.valid_host():
                self.send_error(403, "invalid local host")
                return
            if not self.verified():
                return
            path = unquote(urlsplit(self.path).path)
            if path == "/":
                payload, content_type = page(), "text/html; charset=utf-8"
            elif path == "/api/manifest":
                payload, content_type = (bundle / manifest_name).read_bytes(), "application/json; charset=utf-8"
            elif path == "/api/reviews" and manifest_name == "bundle_manifest.json":
                payload, content_type = json.dumps(read_reviews(bundle), ensure_ascii=False, indent=2).encode(), "application/json; charset=utf-8"
            elif path.startswith("/api/artifacts/"):
                name = path.removeprefix("/api/artifacts/")
                if name not in allowed or "/" in name or name in (".", ".."):
                    self.send_error(404)
                    return
                target = bundle / name
                payload = target.read_bytes()
                content_type = "application/json; charset=utf-8" if name.endswith(".json") else "text/plain; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/reviews" or manifest_name != "bundle_manifest.json":
                self.send_error(404)
                return
            if not self.valid_host():
                self.send_error(403, "invalid local host")
                return
            expected_host = f"127.0.0.1:{self.server.server_port}"
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://{expected_host}", f"http://localhost:{self.server.server_port}"):
                self.send_error(403, "invalid origin")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_REVIEW_FORM_BYTES:
                    self.send_error(413, f"review form exceeds {MAX_REVIEW_FORM_BYTES}-byte transport limit")
                    return
                if self.headers.get("Content-Type", "").split(";")[0] != "application/x-www-form-urlencoded" or length <= 0:
                    raise ValueError("invalid form size or media type")
                fields = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
                if any(len(values) != 1 for values in fields.values()) or fields.get("csrf") != [csrf_token]:
                    raise ValueError("invalid review form token")
                errors = verify(bundle)
                if errors:
                    raise ValueError("bundle verification failed: " + "; ".join(errors))
                append_review(bundle, fields.get("reviewer", [""])[0], fields.get("decision", [""])[0], fields.get("rationale", [""])[0])
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                self.send_error(400, str(exc))
                return
            self.send_response(303)
            self.send_header("Location", "/")
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    return LocalHTTPServer((host, port), Handler)
