<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Autonomous clean-checkout reproduction review

Verdict: **PASS for the authorized CPU reproduction, evidence integrity, and
reached A–C stopping boundary. No remaining material defect found in that
scope. Terminal state remains INCONCLUSIVE_MEASUREMENT; stop this run.**
This is a fresh agent context, local checkout and set of Python processes,
on the same machine, day and operator ownership. It is not independent
performance, accelerator qualification, or learning replication.

## Checkout and runtime

Reviewed frozen result commit:
`48d02b26caac2fa24bd95eac4745b334e8d32af7`.
Historical baseline:
`9d8ad069c51b4d0e6c13e9c6e465579e9b44a339`.
The source checkout was cloned with local `git clone --no-hardlinks --no-checkout`
into `/tmp/dw-sampling-review-2ydqirjv/checkout`, then detached at the exact
frozen commit. This is a full, non-shallow clone. Initial porcelain status was
empty. `/tmp` resolves to `/private/tmp` on this host.

Actual interpreter:
`/Users/artemlegotin/Documents/git_alter/optimizer-aware-residual-auditing/.venv/bin/python`,
native arm64 CPython 3.13.5, macOS 26.7.1, Torch 2.15.0. The existing pinned
environment was reused without installation, updates or downloads.
`PYTHONPATH=/tmp/dw-sampling-review-2ydqirjv/checkout/src` selected the cloned
source; both `oar.__file__` and `oar.sampled_linear.__file__` explicitly resolve
under that clone. All Python review subprocesses had fallback, fast math and
prefer-metal set to `0` before import.

CPU execution was actually used. Torch reports MPS built but not visible in
the sandbox. Platform and architecture were read directly; fresh `sysctl`
brand/model/RAM queries were denied by the sandbox (all three exited 1).
Therefore Apple M3 Max / Mac15,11 / 38,654,705,664 bytes are retained host
identities from the sealed original environment evidence, not freshly
re-measured values in this review. No host-visible rerun or escalation was
needed: the CPU suite succeeded despite those visibility restrictions.

## Executed checks

All commands, absolute cwd, duration and exit codes are in `commands.json`;
stdout/stderr files are separate. Commands used the interpreter and clone
above.

| Check | Exit | Observed result |
| --- | --- | --- |
| Local clone / detached checkout | 0 / 0 | exact frozen commit, full history |
| Fresh process runtime/import identity | 0 | cloned source, original pinned dependencies |
| `python -m pytest -q -m 'not mps'` | 0 | **470 passed, 11 deselected, 5.60s**; no failures or skips |
| `python analysis/run_dw_sampling.py --help` | 0 | required stage/config/out CLI visible |
| Historical `python -m oar.cli report ...` | 0 | new `/tmp/dw-sampling-review-2ydqirjv/historical-rendered.md` |
| `python analysis/run_dw_sampling.py --stage summarize --config configs/dw_sampling.yaml --out analysis/dw_sampling/20261031T105213Z` | 0 | new `summarize/attempt-0002`, status pass |
| Final integrity audit, `audit_checkout_v2.py` | 0 | counts and checks below |
| Post-execution checkout status | 0 | only the newly appended summary directory untracked; tracked diff empty |

The help-side-effect regression also ran as part of the CPU suite. No
prepare, feasibility, correctness, cost, pilot or confirm stage was launched
by this review. No benchmark, model training, corpus body acquisition, or
performance session was run. Ordinary CPU test fixtures are the authorized
suite, not scientific measurement trials.

## Integrity and stop checks

- **184/184 historical tracked paths** match their initial SHA256 inventory
  and the actual baseline Git blobs; the baseline tracked path set also
  equals the inventory set.
- **190/190 current amendment-bound files** verify; strict config canonical
  hashing, reviewed commit existence and the exact historical WP02 stop
  record verify through the driver's own validators.
- **7/7 sealed stage records and 84/84 associated artifacts** verify. This
  comprises the six frozen records with 81 artifacts and the new summary
  record with three artifacts. Exact artifact inventories were checked in
  addition to listed hashes. Source-manifest hashes, measurement maps,
  frozen amendment identities and their bound files all agree.
- For all seven manifests, every one of their 190 source/history bindings
  was also checked against the appropriate source Git snapshot. Actual
  measured prepare2, prepare3, feasibility and correctness snapshots match
  their recorded actual HEADs, retaining v1/v2/v3 distinctions.
- The newly generated summary includes all **five non-summary stage
  records**, reports `INCONCLUSIVE_MEASUREMENT`, sets
  `new_performance_claim=false` and `primary_time_to_target=null`.
- Direct read-only prerequisite calls reject current cost continuation:
  prepare3 is stale against the current driver hash, and feasibility1 is
  `INCONCLUSIVE_MEASUREMENT`. No new stage directory was allocated for these
  checks. Cost, pilot and confirm directories remain absent.
- Existing tracked files are unchanged in the clone. The original checkout
  received only new review files; no source or Git mutation was performed
  by this reviewer.

The first audit (`09-audit`, exit 1) assumed every stage's source already
existed at its recorded HEAD. That reviewer assumption failed for
prepare1, the explicitly inventory-only `not_run` record made before the
first prospective commit. Its manifest correctly records the baseline HEAD
and untracked prospective files. The corrected audit verifies those exact
190 bound bytes at subsequent commit `afa51cb`; it does not rewrite their
original identity or treat the dry run as measurement. Both original script
and failure log remain beside the successful v2 audit.

The newly rendered historical report has different byte hashes solely
because relative links depend on output location. Resolving each Markdown
link to its repository-relative destination produces **identical complete
text**, normalized SHA256
`3f3e840f64de4ced970b90b1d412e9b1d003774866201ac49adf83a305f93ea0`.
The retained copy of the renderer output intentionally keeps its original
links and bytes; those links were generated for the `/tmp` output location.

## Source/report assessment and limitations

Inspection covered the standalone sampler, strict configuration, stage
allocation/sealing, source/amendment validation, reached preparation and
native selection, fused counter repair, prerequisite rejection, and final
report claims. The CPU suite exercises actual gather/weight algebra,
adjoints, accumulation, stream state, extremes, ownership exclusions and
native optimizer behavior, plus protocol rejection and evidence integrity.
The retained B records contain the initial unstable session and exactly one
replacement; later source repair and correctness evidence do not upgrade
that failed qualification. The report correctly separates this stop from a
cost defeat and retains the independent asset block. No scientific result
needs alteration on this review.

The report's completion-addendum placeholder is accurate for frozen commit
48d02b2; the owner can append this actual review result without changing
sealed historical attempts. Some retained commands and qualification fields
name original absolute paths; these are provenance records, not commands
executed by this reproduction. The current source-hash and failed-B checks
already reject continued measurement. Portability of a future resumed
measurement stage, full-shape corrected native qualification, cost/model
stages, checkpoint recovery and learning replication are **not qualified**
by this CPU review. Any future study needs fresh prospective authorization
and fresh qualified prerequisites.

No package-file RECORD inventory was rerun here; original sealed dependency
inventories remain verified as artifacts. Reusing the installed pinned
environment is not a clean dependency rebuild or fresh wheel verification.
No accelerator timing, power/thermal isolation, corpus license resolution,
or independence from the original machine/operator is claimed.

## Key SHA256 identities

| Artifact | SHA256 |
| --- | --- |
| Current sampler | `11398f9340a48b0dcabdc733c74dfa52b53663ea2005989f2ad7b53079827ca3` |
| Current driver | `227f6977d4713e84d01cf2ebeb61d390d769f391da6fe14e65704daac47df663` |
| Config | `afbff17b101ffc45963b0bb8132574c405ba821f196704ff254e721e9e7ad4c3` |
| Scope | `d91d9350dc117f83e39c648e48d435de75ab9ea5a3b6bc197b216c4b73c11181` |
| Current amendment | `7c269df87b3c726def0dccebf245e978da00f1835def79482ecaaf425d613c27` |
| Historical inventory | `10fc155f316905ba509ee1f76b4b8bd8d6a4c438537b153e026100d346e666fc` |
| New summary record | `29977f162f032a7536ee2403373f8ac5eb17915848f04188eb786dcf4861c2e2` |
| New historical rendering | `adccda057cf2b9a6d231190f03754bf9c73c6cf08cbbf404fe40d53050d97c6c` |

Full maps and per-stage source identities are in `10-audit.stdout.txt`.
Copied new summary artifacts are in `summary-attempt-0002/`. This review's
own file inventory excludes itself to avoid a hash cycle.
