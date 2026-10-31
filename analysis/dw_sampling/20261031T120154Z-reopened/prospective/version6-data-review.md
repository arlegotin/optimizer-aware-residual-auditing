# Task 1 independent byte-data review

Recommendation: **PASS** for the invented-fixture implementation scope. No actionable correctness or code-quality findings were identified.

Reviewed `data-brief.md`, `data-fixtures-plan.md`, `data-report.md`, `learning-plan.md`, all of `src/oar/sampling_data.py`, and the appended data-specific fixtures in `tests/test_dw_sampling_protocol.py`. The parent supplied the completed cost/attempt PASS and sealed record SHA256 `d52f2062995810f4541022d02ed21e8f33ad72257675f66f8bed773c34fa9b18`; this review did not repeat qualification or acquire data.

## Requirement evidence

- `src/oar/sampling_data.py:35–56` normalizes CRLF/CR per row, uses each row's `splitlines(keepends=True)`, strips only for the exact heading regex, retains whitespace preambles and title lines, inserts no separator, rejects nonstring rows, and assigns emission indices before filtering. Hashes cover only the emitted normalized UTF8 bytes. Literal fixtures at `tests/test_dw_sampling_protocol.py:746–774` include a heading beginning after an unterminated row, a heading split across rows, and CR/LF split across rows.
- `src/oar/sampling_data.py:59–85` encodes BOS 256 / EOS 257 once, partitions by integer SHA modulo 10, and excludes evaluation hashes present in the supplied complete training sequence while retaining source indices and within-split multiplicity. Tests at `tests/test_dw_sampling_protocol.py:777–810` cover UTF8 bytes, empty explicit articles, frozen records, duplicates, and exclusions from both fitting and development.
- `src/oar/sampling_data.py:88–120` partitions training targets into consecutive chunks of at most 256 and evaluation targets into stride-128 windows with input start `max(0, target_end - 256)`. Integer/range validation excludes BOS targets, reversed/empty bounds, overlong prefixes, and targets beyond EOS. Literal 257-target training and 300-target evaluation fixtures are at `tests/test_dw_sampling_protocol.py:813–832`; full training collation coverage is at `tests/test_dw_sampling_protocol.py:955–962`.
- `src/oar/sampling_data.py:123–140` selects the lowest SHA256 of ASCII `article_hash + ':' + target_start`, applies the prescribed hash/start/source-index tie breaks, preserves duplicate article windows, and restores provided article/target order. Independent expected rankings and reordered input-window fixtures appear at `tests/test_dw_sampling_protocol.py:835–853` and `946–952`.
- `src/oar/sampling_data.py:143–183` shifts each scored target to `target - input_start - 1`, pads every actual row to 256 using PAD 258, masks unscored targets with -100, returns a CPU Boolean scored-target mask, and reports exact valid-target and physical-position counts. It validates selected token sections plus the article boundaries without rescanning whole articles, and imports Torch only inside collation. Tests at `tests/test_dw_sampling_protocol.py:856–923`, `938–943`, and `965–969` cover partial batches, a one-target row, contextual score offsets, target-weighted counting, lazy import, and malformed bounds/tokens.
- `src/oar/sampling_data.py:186–205` emits only provenance, hashes, boundaries, and counts in public manifests; no text or token arrays are included. Literal key/value assertions are at `tests/test_dw_sampling_protocol.py:926–935`.

## Verification and limits

Independent scoped CPU rerun:

```text
PYTHONPATH=src .venv/bin/python -m pytest tests/test_dw_sampling_protocol.py -k 'test_data_' -q
43 passed, 87 deselected in 0.64s
```

No implementation, tests, driver, configuration, or corpus assets were changed. No downloads, hardware measurements, broad test suite, or commit were performed. This PASS covers data-helper behavior on invented fixtures; it does not establish the pinned real-source row interpretation, batch-16 runner integration, numerical evaluation aggregation, manifest freeze, or learning qualification. Those remain with the later authorized integration and source-validation stages.

<!-- SPDX-License-Identifier: CC-BY-4.0 -->
