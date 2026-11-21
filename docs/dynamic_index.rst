# Dynamic index operations (delete / update / append without rebuilding)

This document describes the in-place mutation layer added on top of the
existing Annoy forests. It covers the data model, persistence format,
concurrency guarantees, operating thresholds, and the measured cost curves.

## 1. Problem and design constraints

Annoy builds immutable random-projection trees. The historical workflow is
"add all vectors, build once, query read-only"; any removal required a full
nightly rebuild (~40 min during which new items are unsearchable).

The operations team needs removals to take effect within five minutes, plus
in-place vector replacement and immediate searchability of newly added items,
while keeping every external item id stable for callers that have stored them.

Hard constraints taken from the requirements:

* External ids must never drift: deleting or updating item N must not change
  the id of any other item.
* A mutation only affects its target: distances and neighbors of untouched
  items are unchanged.
* In-memory and on-disk behavior must match; save/reload preserves exactly
  which items were deleted or updated.
* Legacy index files keep loading; new files opened by an old reader fail
  with an explicit error rather than returning scrambled results.
* Concurrent queries never crash, never return a deleted id or a mismatched
  vector.
* A crash during save leaves either the pre-save or post-save file, never a
  half-written one.
* Read-only loads keep their original performance and entry points.

## 2. Data model

Annoy already reserves a fixed node slot per external id in `[0, n_items)`
and never moves item nodes, so ids are inherently stable. The mutation layer
adds four side structures; the built trees themselves are never edited.

| Structure | Meaning |
|---|---|
| `_deleted` | runtime bitmap of dead in-span slots = legacy holes + explicit tombstones |
| `_explicit_del` | only the ids passed to `delete_item`; this is what gets persisted |
| `_shadowed` | in-span slot whose vector was replaced; the authoritative copy now lives in the append region |
| `_appended_del` | tombstone bitmap over append-region nodes |
| `_appended_ids` | external id of each append-region node |

### Delete

`delete_item(i)` only sets a tombstone bit; no tree node is touched, moved,
or renumbered. Query traversal filters tombstones both at leaf nodes and at
multi-item leaf lists. Cost is O(1) and independent of index size.

* `must_exist=False` (default): deleting an unknown or already-deleted id is
  idempotent and returns normally.
* `must_exist=True`: raises `KeyError` naming the first missing id;
  `delete_items([...], must_exist=True)` is a single locked batch.

### Update

`update_item(i, v)` keeps id `i` but writes the new vector into a fresh
append-region node and marks the original in-span slot *shadowed*. The old
tree may still route toward the old slot, but traversal skips shadowed slots,
while the append-region copy is scanned on every query. When an item already
lives in the append region (a prior update or re-add), its node is overwritten
in place. Active count is unchanged.

### Append

`add_item(j, v)` after build appends a node at slot
`tree_node_count + n_appended` and records external id `j`. Appended nodes are
outside every tree, so each query scans them explicitly at O(n_appended · f)
cost, then merges them with tree candidates by exact distance. This keeps new
items searchable immediately with no rebuild. A previously deleted id can be
re-added: it gets a new append node under the same external id.

`tree_node_count` (the value of `_n_nodes` at build time) is stored
explicitly so the append boundary survives reloads even though `_n_nodes`
keeps growing.

### Why append/update scan rather than re-split

Editing a tree node in place would invalidate split hyperplanes and the
`n_descendants` counters that Annoy relies on for traversal and root
detection, and would risk id drift. Appending + linear scanning keeps the
trees byte-stable, makes every mutation O(1)-ish, and bounds the cost to the
number of post-build changes, which is exactly what `compact()` resets.

## 3. Query semantics

* Tombstone/shadow filtering happens inside `_get_all_nns`, so both
  `get_nns_by_item` and `get_nns_by_vector` honor it, with or without
  distances.
* If fewer than `n` live candidates exist, the result is a shorter list;
  deleted ids are never used as padding. Querying a deleted id by item
  returns an empty list; `get_distance` against a dead id returns NaN.
* Results return stable external ids; append-region slots are mapped back to
  their recorded external id before sorting.
* Queries before `build()` keep the legacy behavior (empty results), guarded
  so the new bookkeeping cannot dereference uninitialized bitmaps.

## 4. Persistence format

### Legacy (version 1, unchanged)

A bare sequence of fixed-size nodes; file size is a multiple of the node
size. These files still load through both `load()` and `load_mutable()`, and
tombstones are derived from the existing zeroed-slot ("hole") convention.

### Dynamic (version 2)

```
[ node records ][ deleted section ][ shadowed section ]
                [ appended-deleted section ][ appended external ids ]
                [ footer ][ 8-byte magic "ANNOYDYN" ]
```

Each bitmap section is sparse and self-describing:

* kind 0: empty (4-byte header only)
* kind 1: int64 count + sorted int32 ids
* kind 2: int64 span + raw bitmap bytes

The writer chooses list vs. dense encoding by byte size, so a handful of
deletions in a huge id space costs a few dozen bytes instead of a multi-MB
bitmap. Legacy holes are never persisted — only explicit tombstones — which
keeps sparse indexes small.

The packed footer carries format version, dimensions, metric tag (FNV-1a of
the metric name), node counts, item span, append count, exact trailer size,
and a CRC-32 over the footer.

* Dimension or metric mismatch → explicit `IOError`, never a silent
  reinterpretation.
* Unknown format version → explicit `IOError`.
* CRC mismatch, truncated section, or size/metadata disagreement → explicit
  `IOError`.
* The trailing 8-byte magic makes the file size non-node-aligned, so an old
  Annoy reader rejects it through its existing "size is not a multiple of
  vector size" check instead of misreading the trailer.

### Atomic writes

`save()` writes `name.tmp-annoy`, `fsync`s, then `rename(2)`s over the
target. A crash or power loss leaves either the complete old file or the
complete new one; the temp file may remain but the target is never torn.
On-disk builds follow the same temp+rename path for the trailer. The crash
test in `artifacts/crash_mid_save.py` kills the writer at 40 different points
and confirms the target always reloads.

## 5. Concurrency

A `shared_timed_mutex` (compiled in via the existing
`ANNOYLIB_MULTITHREADED_BUILD` flag) guards all dynamic state and node
writes:

* queries take the shared lock and run concurrently with each other;
* delete / update / append / compact take the exclusive lock.

The GIL is released around long-running query and compact calls in the
Python binding, so real parallelism is preserved for read-heavy serving.
Read-only `load()` indexes reject mutations with a clear message directing
the caller to `load_mutable()`.

## 6. Observability and compaction

`MonitoredAnnoyIndex` (Python wrapper, zero impact on the C++ read path)
exposes `stats()`:

```
n_active, n_deleted, n_appended, deleted_fraction,
query_count, query_avg_ms, query_max_ms, compact_recommended
```

The C++ object itself provides `get_n_active_items()`,
`get_n_appended_items()`, `get_deleted_fraction()`, and `item_exists()`.

`compact(fn, n_trees, n_jobs)` rebuilds a clean forest from live items only,
preserving every external id, and writes it atomically. After compaction the
append scan list and all tombstones are gone, so query cost returns to the
baseline.

### Threshold / operating policy

* Single delete/update/append latency is flat versus library size, so there
  is no size-based trigger.
* The linear append scan grows with `n_appended`; high delete fractions also
  waste traversal on dead leaves. The default `compact_recommended` threshold
  is **deleted_fraction ≥ 1/3**, which operators can override by reading
  `stats()` directly.
* Empirically (20k-item, f=32, 20 trees), queries stay within ~1.5× of
  baseline through 75% deletion and only degrade materially (~12×) at 99%
  deletion, so 1/3 is a conservative, well-margin trigger.

## 7. Measured cost curves

From `artifacts/bench_dynamic.py` (f=32, angular, 20 trees). Full data in
`artifacts/bench_results.json`.

Per-operation latency vs. index size — flat, as required:

| n_items | delete (ms) | update (ms) | append (ms) |
|---:|---:|---:|---:|
| 1,000 | 0.0007 | 0.0024 | 0.0037 |
| 5,000 | 0.0005 | 0.0024 | 0.0045 |
| 20,000 | 0.0003 | 0.0029 | 0.0031 |
| 50,000 | 0.0003 | 0.0024 | 0.0033 |

Query latency and file size vs. deleted fraction (20k items):

| deleted | active | query slowdown | trailer overhead |
|---:|---:|---:|---:|
| 0% | 20,000 | 0.99× | ~0 |
| 10% | 18,000 | 1.04× | ~2.5 KB |
| 50% | 10,000 | 1.29× | ~2.5 KB |
| 75% | 5,000 | 1.53× | ~2.5 KB |
| 90% | 2,000 | 2.39× | ~2.5 KB |
| 99% | 200 | 11.56× | ~2.5 KB |

Recall vs. a full rebuild (brute-force ground truth, 10k items):

| scenario | dynamic recall | rebuild recall | gap |
|---|---:|---:|---:|
| delete 10% | 0.5285 | 0.5290 | 0.0005 |
| delete 50% | 0.5345 | 0.5345 | 0.0000 |

Appended-node scan cost grows roughly linearly with `n_appended` (5k
appends on a 5k-item base roughly quintupled query time), which is the
quantity `compact()` resets to zero.

## 8. API surface

```python
from annoy import AnnoyIndex, MonitoredAnnoyIndex

idx = AnnoyIndex(f, "angular")
idx.add_item(i, v); idx.build(n_trees)

idx.delete_item(i)                       # idempotent
idx.delete_item(i, must_exist=True)      # raises KeyError if missing
idx.delete_items([i, j, k])              # single locked batch
idx.update_item(i, new_v)                # same id, new vector
idx.add_item(new_id, v)                  # post-build append, immediately live
idx.item_exists(i)
idx.get_n_active_items(); idx.get_n_appended_items(); idx.get_deleted_fraction()
idx.save("x.ann")                        # atomic, version-2 file
idx.load_mutable("x.ann")                # read/write reopen (v1 or v2)
idx.load("x.ann")                        # read-only, unchanged performance
idx.compact("clean.ann", n_trees=20)     # rebuild from live items only

mon = MonitoredAnnoyIndex(f, "angular")
mon.stats()                              # counts, fractions, query timing
```

All five metrics (angular, euclidean, manhattan, dot, hamming) support the
full dynamic API. Dot-product indexes get the correct `built` flag on
appended/updated nodes so post-build ranking uses raw inner products.

## 9. Known inherent limits

Annoy preallocates a contiguous node array sized to the largest id, so an
extremely sparse id space (e.g. ids near 2^31 with only a few items) exhausts
memory at `build()` time. This predates the mutation layer and is unchanged;
the sparse trailer only governs on-disk metadata, not the in-memory node
array. Keep ids reasonably dense, or compact to renumber into a dense range.
