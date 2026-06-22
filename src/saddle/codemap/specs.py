"""The node specs the codemap reasons about — the canonical *things*.

There are five — ValueSpec, IdentitySpec, BoundarySpec, ReferenceSpec,
PersistenceSpec — and that set is held consistent end to end by
``tests/test_codemap_coverage.py`` (no spec can exist without its impact, check,
exporter, and manifest field), so this list can't drift from the code.

Split out of checks.py so the fan-out derivations (impact.py for the three
dataflow kinds, substrate.py for references + persistence) and the GATE
(checks.py, the gaps) can both depend on them without a circular import. A spec
names ONE thing the user cares about (a value, an identity, a server→client
boundary, a cross-substrate reference, a persisted field); the matching impact
derives every site that touches it from the real AST / real files, and checks.py
projects the unsatisfied subset of that fan-out as findings. One definition, two
readers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValueSpec:
    """A value that varies. `field` is the key/attr read in code; `accessor` is
    the function (or functions) that returns `base (+) modifiers`. Consumers must
    call a resolver — reading `field` raw means the modifier never reaches them.

    The ideal is ONE accessor; a tuple is accepted because real systems some-
    times resolve through more than one entry point (e.g. a base resolver plus a
    talent-modifier applier). Each is a valid way to put the effective value in
    scope; more than one is a smell the value isn't single-sourced, but the gate
    still verifies coverage against all of them rather than crying false.

    `producers` are functions that legitimately read the base to CONSTRUCT the
    value (the def builder, the deserializer) — they are the source of `base`, not
    consumers of the effective value, so their raw reads are exempt just like a
    resolver's own base read is."""
    name: str
    field: str
    accessor: str | tuple[str, ...]
    producers: tuple[str, ...] = ()

    @property
    def resolvers(self) -> tuple[str, ...]:
        return (self.accessor,) if isinstance(self.accessor, str) else tuple(self.accessor)

    @property
    def exempt_funcs(self) -> frozenset[str]:
        """Functions whose raw base read is legitimate: the resolvers themselves
        (their base read IS the resolution) plus the producers (they build base)."""
        return frozenset(self.resolvers) | frozenset(self.producers)


@dataclass
class IdentitySpec:
    """An identity namespace. `canonical` is the one true member set; the code
    that declares it lives at `source_symbol`; `carriers` are the var/attr/key
    names that HOLD a value of this identity (so a literal compared/assigned to a
    carrier is being used AS this identity and must be a member)."""
    name: str
    canonical: set
    source_symbol: str
    carriers: set


@dataclass
class BoundarySpec:
    """A value that crosses the server→client trust boundary. Written
    authoritatively on the server, it must be read into `replication_func` (so it
    ships in the snapshot) and read by at least one client-domain module (so it
    actually lands on screen)."""
    name: str
    key: str
    replication_func: str


@dataclass
class ReferenceSpec:
    """A name that must be REGISTERED across non-code substrates — config files,
    docs, a DB schema — not just defined in code. This is the "code-substitute
    artifact" axis CLAUDE.md calls out: a feature isn't complete until it appears
    everywhere it must be declared (the cooldown that lived in the engine but
    never in the skill DESCRIPTION is this class of gap, off the AST).

    Each `substrate` is a glob relative to the project root. The `key` must appear
    (as a whole token) in at least one file the glob matches, or that substrate is
    a gap. Substrates are AND'd — every listed place must carry it; files within
    one substrate glob are OR'd — registered in any one of them counts."""
    name: str
    key: str
    substrates: tuple[str, ...]


@dataclass
class PersistenceSpec:
    """A value that must SURVIVE a save/load round-trip. `key` is the field; it
    must be referenced in BOTH `save_func` (written out) and `load_func` (read
    back). Referenced on save but not load means the value silently resets every
    session; referenced on load but not save means it restores a default/garbage
    value. The meta-system completeness axis, derived from code (reads ∪ writes of
    the key inside each named function), so it can't drift from a declaration."""
    name: str
    key: str
    save_func: str
    load_func: str
