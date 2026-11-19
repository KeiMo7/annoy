# Copyright (c) 2013 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# This module is a thin wrapper around the underlying C++ module.
import csv
import os
from concurrent.futures import ThreadPoolExecutor

from .annoylib import Annoy as AnnoyIndex
from .annoylib import read_index_metadata as _read_index_metadata

__all__ = [
    "AnnoyIndex",
    "audit_indexes",
    "check_indexes",
    "export_report_csv",
    "format_report",
    "index_metadata",
    "inspect_index",
    "read_index_metadata",
]


def read_index_metadata(filename, f=None, metric=None):
    """Read index metadata from the 4 KiB header without loading the body."""
    filename = os.fspath(filename)
    kwargs = {}
    if f is not None:
        kwargs["f"] = f
    if metric is not None:
        kwargs["metric"] = metric
    return _read_index_metadata(filename, **kwargs)


inspect_index = read_index_metadata
index_metadata = read_index_metadata


def _rebuild_reason(item):
    status = item.get("status")
    if item.get("error"):
        return item["error"]
    if status == "legacy":
        return "legacy metadata is missing; pass f and metric to validate it"
    if status == "unsupported":
        return "index format is newer than this reader"
    return "index is not loadable with the requested parameters"


def check_indexes(paths, f=None, metric=None, n_jobs=None):
    """Audit many indexes concurrently and split them into loadable/rebuild lists."""
    paths = list(paths)
    if n_jobs is None or n_jobs < 1:
        n_jobs = min(32, max(2, (os.cpu_count() or 2) * 2))

    def inspect_one(path):
        path = os.fspath(path)
        try:
            item = read_index_metadata(path, f=f, metric=metric)
        except Exception as exc:
            item = {
                "path": path,
                "status": "unreadable",
                "format": None,
                "format_version": None,
                "metadata_present": False,
                "dimension": f,
                "metric": metric,
                "node_size": None,
                "item_count": None,
                "tree_count": None,
                "warning": None,
                "error": "%s: %s" % (type(exc).__name__, exc),
            }
        loadable = item.get("status") in ("ok", "current", "compatible", "legacy")
        if item.get("status") == "legacy" and not item.get("dimension"):
            loadable = False
        item["classification"] = "loadable" if loadable else "rebuild"
        item["reason"] = None if loadable else _rebuild_reason(item)
        return item

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        entries = list(executor.map(inspect_one, paths))

    loadable = [item for item in entries if item["classification"] == "loadable"]
    rebuild = [item for item in entries if item["classification"] == "rebuild"]
    return {
        "loadable": loadable,
        "rebuild": rebuild,
        "summary": {
            "total": len(entries),
            "loadable": len(loadable),
            "rebuild": len(rebuild),
        },
    }


audit_indexes = check_indexes


def format_report(result):
    """Render a check_indexes result as a compact text report."""
    summary = result["summary"]
    lines = [
        "Annoy index audit: total=%d loadable=%d rebuild=%d"
        % (summary["total"], summary["loadable"], summary["rebuild"])
    ]
    for heading, key in (("Loadable", "loadable"), ("Rebuild", "rebuild")):
        lines.append("")
        lines.append("%s (%d):" % (heading, summary[key]))
        if not result[key]:
            lines.append("  (none)")
        for item in result[key]:
            dimension = item.get("dimension")
            metric = item.get("metric")
            node_size = item.get("node_size")
            detail = "dimension=%s metric=%s node_size=%s status=%s" % (
                "-" if dimension is None else dimension,
                "-" if metric is None else metric,
                "-" if node_size is None else node_size,
                item.get("status"),
            )
            if item.get("warning"):
                detail += " warning=%s" % item["warning"]
            if item.get("reason"):
                detail += " reason=%s" % item["reason"]
            lines.append("  %s | %s" % (item["path"], detail))
    return "\n".join(lines)


def export_report_csv(result, filename):
    """Export an audit result as a CSV table suitable for release hand-off."""
    fieldnames = [
        "classification",
        "path",
        "status",
        "format",
        "format_version",
        "dimension",
        "metric",
        "node_size",
        "item_count",
        "tree_count",
        "reason",
        "warning",
        "error",
    ]
    with open(filename, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in result["loadable"] + result["rebuild"]:
            writer.writerow(item)
    return filename
