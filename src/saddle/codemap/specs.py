"""The node specs the codemap reasons about — the canonical *things*.

There are eight — ValueSpec, IdentitySpec, BoundarySpec, ReferenceSpec,
LifecycleSpec, PersistenceSpec, AuthoritySpec, BindingSpec — and that set is held
consistent end to end by ``tests/test_codemap_coverage.py`` (no spec can exist
without its impact, check, exporter, and manifest field), so this list can't
drift from code.

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
class LifecycleSpec:
    """A declared symbol that must be READ by some code to mean anything — an
    exported setting, a constant, a signal, a script-scope field. `symbol` is the
    declared identifier; the impact derives its declaration site(s) and every read
    of it from the real AST. A declaration with zero reads anywhere in the project
    is a DEAD knob: it looks adjustable (a designer can set it) but changes
    nothing, because no code consumes it. This is the completeness axis ORTHOGONAL
    to value propagation — propagation asks whether a read sees the modifier, this
    asks whether the declaration has any read at all. The status-effect
    `max_stacks_per_kind` / `bleed_ignores_armor` / cc-flag exports and the
    loadout `encoding` export — all declared, all read by nobody — are this gap."""
    name: str
    symbol: str


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


@dataclass
class AuthoritySpec:
    """A set of MUTATORS that change server-authoritative state and must each call
    an authority guard before they write, or a client can invoke them and desync /
    cheat. `guard` is the authority-check function (or functions) — e.g.
    `is_server`, `is_multiplayer_authority`, a project `_is_server` — that must be
    CALLED inside each mutator's body; `mutators` are the functions that perform
    the authoritative write. This is the trust-boundary axis on the WRITE side
    (BoundarySpec covers the read/replication side): RayXI's talent/loadout
    mutators shipped with no such gate, so the server applied whatever a client
    sent. A defined mutator with no guard call in its body is the gap; a mutator
    the code doesn't define is named by nobody, so it stays silent (a spec error,
    not an authority gap)."""
    name: str
    guard: str | tuple[str, ...]
    mutators: tuple[str, ...]

    @property
    def guards(self) -> tuple[str, ...]:
        return (self.guard,) if isinstance(self.guard, str) else tuple(self.guard)


@dataclass
class BindingSpec:
    """An input keymap that must be UNAMBIGUOUS and fully REACHABLE — the axis
    RayXI's gates never had, so a build shipped the number row firing BOTH a
    combat ability AND a store/social panel: pressing `1` opened Collections
    instead of casting, and every menu key did the same, because two binding
    mechanisms (hotbar-slot assignment and panel assignment) each claimed the same
    physical keys and nothing reconciled them.

    Off the AST — the bindings live in an engine keymap file (a Godot
    ``project.godot`` ``[input]`` section), not in code — so this sits beside the
    other substrate axes. ``keymap`` is that file, relative to root.

    The completeness rule has two halves, both derived from the ACTUAL file (never
    a second declaration that can drift):

      COLLISION — group every action by the physical trigger that fires it (a
        keyboard keycode, a mouse button, a gamepad button/axis). A trigger that
        fires two DISTINCT actions is a collision (pressing it does two things at
        once / the wrong thing) UNLESS the two actions' intent FAMILIES are
        declared compatible. ``families`` maps a family name to the action-name
        prefixes/exact-names in it (an action takes the family of its LONGEST
        matching prefix; unmatched → ``"other"``). ``compatible`` lists the
        unordered family pairs allowed to co-bind one trigger — a context-exclusive
        pair that can never both fire (e.g. a class-gated card slot vs an ability
        slot on the same number key), or a deliberate synonym cluster declared by
        pairing a family with ITSELF (e.g. ``("dismiss","dismiss")`` so
        cancel/pause/ui_cancel may all share Esc). Same-family co-binds are NOT
        allowed by default: two different abilities on one key is a real gap, so it
        must be opted into explicitly.

      REACHABILITY — an action declared in the keymap with NO trigger at all (an
        empty events list) is a DEAD action: it exists but can never be invoked.
        ``programmatic`` lists action names exempt because code fires them directly
        (``Input.is_action_pressed`` from a script, never a physical input).

    Every field serialises to plain JSON so the design step can hand the policy
    over in ``Design.meta`` exactly like the other specs."""
    name: str
    keymap: str
    families: dict
    compatible: tuple = ()
    programmatic: tuple = ()

    @property
    def compatible_set(self) -> frozenset[frozenset[str]]:
        """The compatible family pairs as a set of unordered pairs, for O(1)
        membership tests. A pair listed as ``(f, f)`` permits same-family co-bind."""
        return frozenset(frozenset(p) for p in self.compatible)

    def family_of(self, action: str) -> str:
        """The intent family of an action: the family whose declared prefix is the
        LONGEST match for the action name; ``"other"`` when none matches. Exact
        action names are just full-length prefixes, so a family may list either."""
        best: str = "other"
        best_len = -1
        for fam, prefixes in self.families.items():
            for pre in prefixes:
                if action.startswith(pre) and len(pre) > best_len:
                    best, best_len = fam, len(pre)
        return best
