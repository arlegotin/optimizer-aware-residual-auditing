# SDD ledger — plan: decisions/dw_sampling_reopening.md

Owner authorization: latest request explicitly reopens the old measurement stop, removes arbitrary caps, and prioritizes obtaining a useful positive result. Exact prior source HEAD40fd56f and clean checkout recorded in new run initial inventory.

Preflight:
| Task/interface | Check | Result |
| --- | --- | --- |
| Timing driver/tests | duration calibration, warmup diagnostic, uncapped resource semantics | strict new schema2; no sampler changes |
| Protocol/config -> driver | minimum .5s, uncapped updates, target-only A/A floor0 | overrides old timing statements prospectively; fixed math/data protections preserved |
| Freeze -> measurements | old config/amendment snapshots, direct source hashes | before comparative measurements; new run root |
| Timing -> learning | actual cost gate, asset permission | remains conditional; positive throughput alone cannot establish quality |
| Evidence -> reviews | raw blocks/attempts/failed states kept | no selecting observations to force favorable result |

Ruling: the owner's explicit rewrite/fix/continue instruction supplies approval for this new prospective timing design and local execution — no repeated confirmation — incorrect scope would require affected evidence withdrawal, not historical edits.
Ruling: warmup max32 remains only a finite diagnostic observation window, not a stop; no update-count/whole-stage cap prevents reaching the timing duration — median/paired evidence handles noise — wrong interpretation risks spending extra local compute, not a fabricated positive claim.

Task1 dispatched /root/reopen_timing (gpt-6.1-sol/high), owns only driver/protocol tests. Parent owns new scope/config/amendment/evidence; no simultaneous hardware/tests once measurements start.
