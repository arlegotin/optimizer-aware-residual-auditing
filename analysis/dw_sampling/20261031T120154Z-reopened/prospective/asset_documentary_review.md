<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# WikiText documentary review: proposal for a limited local use

Review observed 2026-10-31, approximately 12:06–12:08 UTC. This is a
documentary proposal for the controlling review, not corpus approval, an
authoritative license-version determination, or legal advice. No corpus
archive, parquet split, dataset viewer, dataset loader, or training code was
requested or executed. Publisher documentation can contain illustrative text;
no such examples are reproduced or used here. No hardware work or tests ran.

## Proposed finding

There is primary-source support for resolving the **narrow repository-use
question** without resolving the conflicting version notices: the creator's
official article offers WikiText under CC BY-SA, and both candidate CC
versions permit reproduction and creating/reproducing adaptations. The
documented proposed use consists of local copying, labeled preprocessing,
and private research, with corpus material and potentially derived artifacts
kept local. Treating a transformation as an adaptation when that is uncertain
does not require a different permission for those private acts.

That is an inference from the grants, not a statement that the metadata is
wrong, that the prose controls, that model weights are legally adaptations,
or that the version conflict has disappeared. The GFDL metadata remains part
of the notice record. This review neither treats GFDL as revoked nor relies
on a GFDL-only permission theory or an exception such as fair use.

The documentary evidence supports a **prospective limited-use decision** by
the parent/controller if the experiment reaches its asset gate. No new creator
communication is necessary merely to establish that both candidate CC grants
contain the relevant copying/adaptation permissions. A different finding
would be needed for public sharing, a claim to have identified the governing
version, or a use beyond this limited permission analysis. This file does not
itself change the blocked manifest or satisfy byte/split/preprocessing gates.

## Primary sources and retrieval facts

All URLs below are publisher/creator or Creative Commons sources. Search
results from third-party mirrors were not used as evidence. Retrieval used
the web tool's parsed responses; no exact remote response bytes, HTTP headers,
or remote-document SHA256 were captured. A tool crawl age is not a claim of
an independently verified live server response or a publication revision.

1. **Pinned official dataset card**, revision
   `b08601e04326c79dfdd32d625aee71d232d685c3`:
   [Salesforce/wikitext README](https://huggingface.co/datasets/Salesforce/wikitext/raw/b08601e04326c79dfdd32d625aee71d232d685c3/README.md).
   The tool returned plain text (322 parsed lines; crawl marked today).
   YAML lines 7–9 identify `cc-by-sa-3.0` and `gfdl`; the licensing paragraph
   at parsed line 306 names and links CC BY-SA 4.0. The summary separately
   states the CC BY-SA family without a version. This independently
   reproduces the recorded conflict, rather than resolving it by selecting
   a newer card. The README's point of contact is Stephen Merity and its
   homepage is the old Einstein article URL. The local archived README
   SHA256 below matches the manifest's notice-evidence hash; equivalence to
   fetched remote bytes was not separately established.

2. **Creator's official article**, attributed to Stephen Merity and dated
   September 26, 2016:
   [The WikiText Long Term Dependency Language Modeling Dataset](https://www.salesforce.com/blog/the-wikitext-long-term-dependency-language-modeling-dataset/?bc=OTH).
   This exact query-bearing URL returned the article (206 parsed lines;
   tool crawl marked two weeks ago), including the unversioned statement:
   “The dataset is available under the Creative Commons Attribution-ShareAlike
   License.” Its link points to Wikipedia's CC BY-SA explanation, not a
   version-specific legal code. The article also distinguishes word-level
   and raw character-level downloads and describes the latter as appropriate
   for character work or newly derived datasets. No download links were
   followed. This supplies creator provenance and a permission-family
   statement, not a version selection.

3. **Availability qualification for that article**:
   [Salesforce's Stephen Merity author page](https://www.salesforce.com/blog/author/stephen-merity/)
   returned a link to the dataset article (93 parsed lines; crawl marked two
   weeks ago). Following that queryless article link returned a redirect to
   [Salesforce's AI blog index](https://www.salesforce.com/blog/ai/), marked
   crawled today. Opening the
   [old Einstein homepage](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
   failed as inaccessible through the tool. Thus a redirect is not evidence
   of a withdrawn grant; conversely, the cached/query-bearing retrieval is
   not represented as a verified current live-byte copy. The exact query
   parameter is retained for reproducibility of the successful retrieval.

4. **Official CC BY-SA 3.0 Unported legal code**, version 3.0:
   [HTML](https://creativecommons.org/licenses/by-sa/3.0/legalcode),
   [official plain text](https://creativecommons.org/licenses/by-sa/3.0/legalcode.txt).
   Both returned successfully, marked crawled today; the plain text has 340
   parsed lines. Section 3(a) permits reproduction; 3(b) permits making and
   reproducing adaptations and requires reasonable identification of changes.
   The grant is conditional on license compliance. Section 4(a)–(c) regulates
   distribution/public performance, license notices, attribution and adapted
   work licensing. It does not require public release merely because an
   adaptation is created. Section 4(a) also contains a credit-removal
   obligation on licensor notice for collections/adaptations. For this
   proposal, retaining notices and labeling changes locally avoids relying
   on an assumption that every condition is postponed until distribution.

5. **Official CC BY-SA 4.0 International legal code**, version 4.0:
   [English HTML](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en),
   [official plain text](https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt).
   Both returned successfully, marked crawled today; the plain text has 384
   parsed lines. Section 2(a)(1) permits reproduction and producing/reproducing
   adapted material; 2(a)(4) covers necessary technical format changes.
   Section 3(a)'s attribution, notice and modification conditions are framed
   around sharing; 3(b)'s ShareAlike conditions apply when adapted material
   is shared. Section 1(k) defines sharing as providing material to the
   public. Therefore these provisions do not require publication of private
   adaptations. Section 2(b) excludes certain other rights; this is a review
   of the recorded permission conflict, not a guarantee of title or every
   possible right. The creator/publisher grants remain conditional on rights
   they can grant and applicable license terms.

Only the creator grant's sentence is quoted from external prose.
License descriptions above are paraphrases; no corpus excerpts are included.

## Repository constraints and the proposed decision's exact extent

The original `decisions/dw_sampling_scope.md` asset paragraph requires a new
documentary/owner asset decision before any corpus body use, preserves the
existing restriction, forbids substitution, and seals the official test.
The reopening document removes engineering caps but expressly retains corpus
restrictions and says the generic instruction is not documentary resolution.
Ruling 13 distinguishes a common-permission review from authoritative
version selection, but its status is **conditional and unused**; its named
`wikitext-primary-recheck.json` evidence file was not found in this checkout.
This review relies on the newly inspected primary sources above instead of
asserting that the historical ruling already approved acquisition.

The manifest currently says `approval_status: blocked`,
`local_status: metadata_only`, and `workload_ready: false`. Its corpus
SHA256 is null and each split remains `not_downloaded`. These statements
were not modified. The available LFS identifiers are upstream claims, not
local byte verification.

If the parent adopts the proposed documentary decision, record it explicitly
before any asset use and limit it as follows:

- Permit only the specified local research workflow at the pinned revision,
  after the prior experiment/cost prerequisites pass. Keep the existing
  official validation/test sealing rules; this proposal grants no access to
  those bodies and no expansion of the data protocol.
- Preserve the complete conflicting card notices, GFDL label, source URLs,
  creator/Wikipedia attribution and citation, license links and disclaimers
  in a local provenance record. Label every preprocessing change and retain
  earlier modification notices. Respond appropriately to any licensor
  credit-removal request.
- Keep original corpus, extracted text, tokenized/otherwise transformed
  datasets, text excerpts, text-bearing logs, generated text and checkpoints
  private. The checkpoint restriction deliberately avoids needing to decide
  whether any particular trained artifact is an adaptation. Publish only
  code and research measurements that do not reproduce corpus expression;
  any later sharing proposal receives its own review.
- Verify acquired bytes against pinned source identities, preserve split
  boundaries and preprocessing provenance, and satisfy the prescribed
  workload-readiness checks. A documentary finding alone establishes none of
  these facts. Retain every historical blocked status as historical evidence.

**Actual outstanding determination:** whether the controlling review adopts
this specific local-use permission analysis as the new documentary decision
required by the original scope. The generic cap-removal instruction does
not answer that question. The unresolved historical version label need not
be selected to adopt the limited analysis; it must remain disclosed. No
corpus use is authorized by this proposal alone.

## Local evidence identities at review

Read from the shared checkout; no commits or manifest/source/config edits
were made by this review. SHA256 values were computed locally with
`shasum -a 256`:

| File | SHA256 |
| --- | --- |
| `decisions/dw_sampling_reopening.md` | `e494ab971dfdba55b8673242cc149e7b8180d86bb1b8452037c71652999c1618` |
| `decisions/dw_sampling_scope.md` | `d91d9350dc117f83e39c648e48d435de75ab9ea5a3b6bc197b216c4b73c11181` |
| `decisions/rulings.md` | `70c7e8b55f5deb93d86b5e3b40f2d2098319d10a9a8ee5efb5c02c4b6efa3c08` |
| `manifests/assets.json` | `92e7319d4a0c89810161fb718b682b7bcd3dbc665eaea2c97848ce92e58d9bdf` |
| `third_party/discovery/wikitext/README.md` | `d5aabf145d341366745ac703b5844c442c143943b6c77929481395c822b1c28c` |
