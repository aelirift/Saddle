"""Memory-aware process pool for LLM subprocess callers.

Batch-admits subprocesses with memory pacing:
  - Hard cap: host-aware (~3 slots/GiB RAM, clamped [8, 98]; see
    _default_max_procs / RAYXI_MAX_PROCS). Was a flat 98 — too high for a
    14 GiB box, where a full content_pack fan-out OOM-rebooted the host.
  - Batch admission with adaptive sizing:
      active < 50 → batch 3–5, cooldown 2.5s
      active >= 50 → batch 1–2, cooldown 5s
  - Memory ceiling: 80%. No new procs above this.
    Resumes below 70% (hysteresis prevents start-stop thrashing).
  - Collects per-label duration stats (caller type, phase, etc.).

Usage:
    from saddle.llm.pool import get_pool

    async with get_pool(label="ClaudeCLI/HLR"):
        result = await some_llm_call()

    # After a run, inspect stats:
    get_pool().print_stats()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from saddle import supervise
from saddle.context import Context
from saddle.trace import get_trace

_log = logging.getLogger("saddle.llm.pool")


def _total_memory_gb() -> float:
    """Total system RAM in GiB (0.0 if it can't be read — non-Linux)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def _pool_policy() -> dict:
    """Saddle's process-pool policy (config/llm_policy.json -> "pool").

    Lazy import so pool.py stays importable even if policy can't load;
    returns {} on any failure (callers fall back to host-aware defaults).
    """
    try:
        from saddle.llm.policy import pool_settings
        return pool_settings()
    except Exception:
        return {}


def _default_max_procs() -> int:
    """Host-aware concurrency ceiling.

    The old hardcoded 98 was blind to host RAM: on a 14 GiB box a full
    98-slot content_pack fan-out drove system memory to 95.6% (peak observed
    2026-06-20, run 171926) and the host OOM-rebooted. The admission-time
    memory ceiling (``_MEM_CEILING_PCT``) can only block NEW acquires — it
    can't claw back the calls already in flight — so the CAP is the real
    lever. Derive it from total RAM at ~3 slots/GiB (≈340 MiB budget per
    in-flight streaming call), clamped to [8, 98] so small hosts stay safe
    and large hosts keep the prior ceiling. ``RAYXI_MAX_PROCS`` overrides for
    ops tuning without a code change.
    """
    env = os.environ.get("SADDLE_MAX_PROCS") or os.environ.get("RAYXI_MAX_PROCS")
    if env and env.strip():
        try:
            return max(1, int(env))
        except ValueError:
            _log.warning("SADDLE_MAX_PROCS=%r not an int — using policy/host default", env)
    # Saddle policy (config/llm_policy.json -> pool.max_procs) wins over the
    # host-derived default; the env var above still overrides policy.
    _policy_mp = _pool_policy().get("max_procs")
    if _policy_mp:
        try:
            return max(1, int(_policy_mp))
        except (TypeError, ValueError):
            pass
    total_gb = _total_memory_gb()
    if total_gb <= 0:
        return 24  # unknown host: conservative middle, well under any ceiling
    # ~4 slots/GiB (lifted 3->4 2026-06-20): gives the per-provider DYNAMIC cap
    # (MiniMax floors at 36) room to explore toward the ~45 429-onset instead of
    # being masked by the global cap. ~0.77% mem/slot → 14 GiB host (~57) ≈ 64%
    # mem, under the 80% admission ceiling (the hard mem backstop).
    return max(8, min(98, int(total_gb * 4)))


_MAX_PROCS = _default_max_procs()
_BATCH_SIZE_LOW = 30      # batch size when active < _ACTIVE_THRESHOLD (API calls are lightweight)
_BATCH_SIZE_HIGH = 10     # batch size when active >= _ACTIVE_THRESHOLD
_ACTIVE_THRESHOLD = 50    # switch from large to small batches
_COOLDOWN_LOW_S = 0.5     # minimal pause between batches for API calls
_COOLDOWN_HIGH_S = 1.0    # slightly longer at high load
_MEM_CEILING_PCT = float(_pool_policy().get("mem_ceiling_pct", 80.0))   # hard stop — no new procs above this
_MEM_RESUME_PCT = float(_pool_policy().get("mem_resume_pct", 70.0))    # resume admitting below this
_MEM_POLL_S = 3           # how often to recheck when paused on memory

# Upper bound on how long admission will WAIT for memory to fall below the
# ceiling. The ceiling can only pause NEW acquires; it cannot reclaim in-flight
# calls, so if something ELSE on the host holds RAM above the ceiling and never
# frees it, the wait loop would spin forever — a silent hang with only a periodic
# warning, exactly the "I never see saddle return" failure. Past this budget the
# ONE blocked acquire raises supervise.DeadlineExceeded (a typed signal the caller
# fails the call / fails over on) instead of wedging the whole run. 0 disables the
# bound (wait forever) — an EXPLICIT opt-out, never the silent default.
try:
    _MEM_WAIT_DEADLINE_S = float(
        os.environ.get("SADDLE_POOL_MEM_WAIT_DEADLINE_S")
        or os.environ.get("RAYXI_POOL_MEM_WAIT_DEADLINE_S")
        or _pool_policy().get("mem_wait_deadline_s", 600.0)
    )
except (TypeError, ValueError):
    _MEM_WAIT_DEADLINE_S = 600.0


def _memory_usage_percent() -> float:
    """Return memory usage as a percentage (0-100)."""
    try:
        with open("/proc/meminfo") as f:
            total = available = 0
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1])
                if total and available:
                    break
            if total == 0:
                return 0.0
            return ((total - available) / total) * 100
    except OSError:
        return 0.0  # non-Linux: assume fine


class _ProcStats:
    """Tracks per-label timing stats."""

    def __init__(self) -> None:
        # label → list of durations (seconds)
        self._durations: dict[str, list[float]] = {}

    def record(self, label: str, duration: float) -> None:
        self._durations.setdefault(label, []).append(duration)

    def summary(self) -> dict[str, dict[str, float]]:
        """Return {label: {count, total_s, avg_s, min_s, max_s}}."""
        out: dict[str, dict[str, float]] = {}
        for label, durations in sorted(self._durations.items()):
            out[label] = {
                "count": len(durations),
                "total_s": round(sum(durations), 1),
                "avg_s": round(sum(durations) / len(durations), 1),
                "min_s": round(min(durations), 1),
                "max_s": round(max(durations), 1),
            }
        return out

    def format(self) -> str:
        lines = ["Pool stats by label:"]
        lines.append(f"  {'label':<40} {'count':>5}  {'total':>7}  {'avg':>6}  {'min':>6}  {'max':>6}")
        lines.append(f"  {'-'*40} {'-'*5}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}")
        for label, s in self.summary().items():
            lines.append(
                f"  {label:<40} {s['count']:>5}  {s['total_s']:>6.1f}s  {s['avg_s']:>5.1f}s  {s['min_s']:>5.1f}s  {s['max_s']:>5.1f}s"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        self._durations.clear()


class ProcessPool:
    """Memory-paced subprocess pool with batch admission and stats tracking."""

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(_MAX_PROCS)
        self._admission = asyncio.Lock()
        self._active = 0
        self._batch_admitted = 0
        self.stats = _ProcStats()
        # Per-tenant fairness gates (multi-tenant). Acquired OUTSIDE the
        # global memory admission so a single tenant holds at most its
        # configured cap of concurrent slots — one tenant's burst can't
        # starve another. The global memory pool stays the host-safety
        # ceiling; these gates only add per-tenant fairness on top.
        self._tenant_sems: dict[str, asyncio.Semaphore] = {}
        self._tenant_caps: dict[str, int] = {}
        self._tenant_inflight: dict[str, int] = {}

    @property
    def active(self) -> int:
        return self._active

    def _batch_size(self) -> int:
        return _BATCH_SIZE_LOW if self._active < _ACTIVE_THRESHOLD else _BATCH_SIZE_HIGH

    def _cooldown(self) -> float:
        return _COOLDOWN_LOW_S if self._active < _ACTIVE_THRESHOLD else _COOLDOWN_HIGH_S

    async def acquire(self, label: str = "") -> float:
        """Acquire a slot. Returns the wall-clock start time for stats.

        Raises :class:`supervise.DeadlineExceeded` if memory stays above the
        ceiling for longer than ``_MEM_WAIT_DEADLINE_S`` — admission waits, but
        it does NOT wait forever.
        """
        # Hard cap — blocks if _MAX_PROCS already active (host-aware; see
        # _default_max_procs)
        await self._semaphore.acquire()

        # The hard-cap slot is now HELD. Everything below can wait (memory
        # gate), sleep (cooldown), be cancelled, or raise the memory deadline —
        # any exit that is NOT the normal return must hand the slot back, or the
        # pool permanently shrinks by one (and eventually deadlocks). release()
        # only runs via PoolSlot.__aexit__, which never fires if acquire raises,
        # so the slot return is this method's own responsibility.
        admitted = False
        try:
            # Serialize admission decisions so batching works
            async with self._admission:
                # Hard stop while memory is above ceiling — BOUNDED so sustained
                # external pressure can't wedge the acquire forever.
                usage = _memory_usage_percent()
                waited = 0.0
                while usage >= _MEM_CEILING_PCT:
                    if _MEM_WAIT_DEADLINE_S > 0 and waited >= _MEM_WAIT_DEADLINE_S:
                        raise supervise.DeadlineExceeded(
                            f"pool admission [{label or 'unlabeled'}]: memory "
                            f"{usage:.0f}% stayed >= ceiling {_MEM_CEILING_PCT:.0f}% "
                            f"for {waited:.0f}s — refusing to wait forever"
                        )
                    _log.warning(
                        "Pool: memory %.0f%% >= ceiling %d%%, %d active — "
                        "paused until %.0f%% (waited %.0fs/%.0fs)",
                        usage, _MEM_CEILING_PCT, self._active, _MEM_RESUME_PCT,
                        waited, _MEM_WAIT_DEADLINE_S,
                    )
                    trace = get_trace()
                    if trace:
                        trace.pool_pause(f"memory_ceiling_{usage:.0f}pct",
                                         self._active, usage)
                    await asyncio.sleep(_MEM_POLL_S)
                    waited += _MEM_POLL_S
                    usage = _memory_usage_percent()
                    if usage < _MEM_RESUME_PCT:
                        _log.info("Pool: memory %.0f%% < resume %d%% — resuming",
                                  usage, _MEM_RESUME_PCT)
                        break

                self._active += 1
                admitted = True
                self._batch_admitted += 1
                batch_sz = self._batch_size()
                _log.debug(
                    "Pool: acquired [%s] (memory %.0f%%, %d active, batch %d/%d)",
                    label, usage, self._active, self._batch_admitted, batch_sz,
                )
                trace = get_trace()
                if trace:
                    trace.pool_acquire(label, self._active, usage)

                # After a full batch, cooldown to let memory settle
                if self._batch_admitted >= batch_sz:
                    self._batch_admitted = 0
                    cd = self._cooldown()
                    _log.debug(
                        "Pool: batch full — cooling down %.1fs before next batch",
                        cd,
                    )
                    if trace:
                        trace.pool_batch_cooldown(batch_sz, cd, self._active)
                    await asyncio.sleep(cd)

            return time.monotonic()
        except BaseException:
            # Memory-deadline raise, cancellation, or any error after the slot
            # was taken — undo the admission counter (if we got that far) and
            # release the hard-cap slot so a transient failure never strands it.
            if admitted:
                self._active -= 1
            self._semaphore.release()
            raise

    def release(self, label: str = "", start_time: float = 0.0) -> None:
        self._active -= 1
        self._semaphore.release()
        duration = time.monotonic() - start_time if start_time > 0 else 0.0
        if start_time > 0:
            self.stats.record(label or "unlabeled", duration)
            _log.debug("Pool: released [%s] after %.1fs (%d active)",
                        label, duration, self._active)
        else:
            _log.debug("Pool: released (%d active)", self._active)
        trace = get_trace()
        if trace:
            trace.pool_release(label, self._active,
                               _memory_usage_percent(), duration)

    # -- per-tenant fairness ---------------------------------------------

    def _tenant_sem(self, tenant: str, cap: int) -> asyncio.Semaphore:
        """Get-or-create a tenant's fairness semaphore. Race-free in one
        asyncio loop; the cap is fixed at first creation for the tenant."""
        sem = self._tenant_sems.get(tenant)
        if sem is None:
            cap = max(1, int(cap))
            sem = asyncio.Semaphore(cap)
            self._tenant_sems[tenant] = sem
            self._tenant_caps[tenant] = cap
        return sem

    async def acquire_tenant(self, tenant: str, cap: int) -> None:
        """Block until this tenant has a free fairness slot (cap concurrent)."""
        await self._tenant_sem(tenant, cap).acquire()
        self._tenant_inflight[tenant] = self._tenant_inflight.get(tenant, 0) + 1

    def release_tenant(self, tenant: str) -> None:
        sem = self._tenant_sems.get(tenant)
        if sem is None:
            return
        self._tenant_inflight[tenant] = max(
            0, self._tenant_inflight.get(tenant, 0) - 1
        )
        sem.release()

    def tenant_inflight(self, tenant: str) -> int:
        """Concurrent slots a tenant currently holds (introspection/tests)."""
        return self._tenant_inflight.get(tenant, 0)

    def print_stats(self) -> None:
        _log.info("\n%s", self.stats.format())

    async def __aenter__(self) -> ProcessPool:
        await self.acquire()
        return self

    async def __aexit__(self, *args) -> None:
        self.release()


class PoolSlot:
    """Context manager that carries a label and tracks timing.

    Usage:
        async with PoolSlot(get_pool(), "ClaudeCLI/MLR-fsm"):
            ...
    """

    def __init__(self, pool: ProcessPool, label: str = "") -> None:
        self._pool = pool
        self._label = label
        self._start: float = 0.0

    async def __aenter__(self) -> PoolSlot:
        self._start = await self._pool.acquire(self._label)
        return self

    async def __aexit__(self, *args) -> None:
        self._pool.release(self._label, self._start)


class _NullGate:
    """No-op async gate — used when no per-tenant cap is configured, so the
    single-tenant / unconfigured path stays byte-for-byte as before."""

    async def __aenter__(self) -> "_NullGate":
        return self

    async def __aexit__(self, *args) -> None:
        return None


class TenantGate:
    """Per-tenant fairness gate. Wrap a tenant's unit of LLM work in it
    (OUTSIDE the per-caller memory PoolSlot) so the tenant holds at most
    ``cap`` concurrent in-flight calls.

        async with TenantGate(get_pool(), ctx.tenant, cap):
            await caller(system, prompt, ...)
    """

    def __init__(self, pool: ProcessPool, tenant: str, cap: int) -> None:
        self._pool = pool
        self._tenant = tenant
        self._cap = cap

    async def __aenter__(self) -> "TenantGate":
        await self._pool.acquire_tenant(self._tenant, self._cap)
        return self

    async def __aexit__(self, *args) -> None:
        self._pool.release_tenant(self._tenant)


def tenant_gate(ctx: Context, pool: "ProcessPool | None" = None):
    """Return the fairness gate for ``ctx``'s tenant.

    Reads ``pool.tenant_max_concurrency`` from the context-resolved policy.
    When unset, non-numeric, or >= the global cap (the memory pool is
    already the tighter constraint), returns a no-op gate so the
    unconfigured path is unchanged.
    """
    cap = None
    try:
        from saddle.llm.policy import pool_settings
        cap = pool_settings(ctx).get("tenant_max_concurrency")
    except Exception:  # noqa: BLE001 — policy unavailable -> no fairness bound
        cap = None
    if not cap:
        return _NullGate()
    try:
        cap = int(cap)
    except (TypeError, ValueError):
        return _NullGate()
    if cap >= _MAX_PROCS:
        return _NullGate()
    return TenantGate(pool or get_pool(), ctx.tenant, cap)


# Singleton — shared across all callers in one process
_pool: ProcessPool | None = None


def get_pool() -> ProcessPool:
    """Return the global ProcessPool singleton."""
    global _pool
    if _pool is None:
        _pool = ProcessPool()
    return _pool
