# Demo: 30 tenants sharing one memory budget, with LRU eviction,
# automatic remount, and a final per-tenant report (JSON on stdout).
#
# Run from the repository root:
#   PYTHONPATH=. python output/tenant_manager_demo.py > output/sample_report.json
import json
import os
import random
import tempfile

from annoy import AnnoyIndex, TenantIndexManager, estimate_resident_bytes

DIM = 16
N_TENANTS = 30


def build_index(path, seed):
    rng = random.Random(seed)
    idx = AnnoyIndex(DIM, "angular")
    for i in range(50):
        idx.add_item(i, [rng.uniform(-1, 1) for _ in range(DIM)])
    idx.build(4)
    idx.save(path)
    idx.unload()


def main():
    tmp = tempfile.mkdtemp(prefix="tenancy-demo-")
    mgr = TenantIndexManager(budget_bytes=None)  # set after we know sizes
    for t in range(N_TENANTS):
        p = os.path.join(tmp, "tenant-%02d.ann" % t)
        build_index(p, seed=t)
        mgr.register("tenant-%02d" % t, p, DIM)

    # Budget: room for 10 of the 30 identical tenants.
    est = estimate_resident_bytes(os.path.join(tmp, "tenant-00.ann"))
    mgr.set_budget(10 * est)

    rng = random.Random(7)
    # Hot tenants get queried a lot; the tail is touched once. With only
    # 10 slots for 30 tenants, the cold ones are evicted and remounted.
    for step in range(600):
        if step % 3:
            tid = "tenant-%02d" % rng.randrange(10)
        else:
            tid = "tenant-%02d" % rng.randrange(N_TENANTS)
        v = [rng.uniform(-1, 1) for _ in range(DIM)]
        mgr.query(tid, v, 5)

    # Shrink the budget online, no restart: usage drops immediately.
    mgr.set_budget(5 * est)
    after_shrink = mgr.report()["used_bytes"]

    report = mgr.report()
    report["demo"] = {
        "tenants": N_TENANTS,
        "queries": 600,
        "budget_bytes_after_online_shrink": 5 * est,
        "used_bytes_after_online_shrink": after_shrink,
        "estimated_resident_bytes_per_index": est,
    }
    mgr.close()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
