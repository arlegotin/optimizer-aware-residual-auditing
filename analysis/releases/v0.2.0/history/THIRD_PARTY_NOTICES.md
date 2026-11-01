# Third-party attribution and boundaries

The project licenses cover original material only. Archived upstream files, dependencies and standard license documents retain their own notices and terms. The release includes selected public source and metadata, not dependency wheels, datasets, pretrained weights or local tensor checkpoints.

## Archived discovery sources

The immutable [source manifest](manifests/upstream_sources.json) records exact revisions, source URLs and hashes of the originally archived bytes. The additional PyTorch license below accompanies its existing source copy; the manifest's original bytes are preserved. Discovery excerpts do not establish an implemented or calibrated named baseline port.

| Source | Recorded terms and attribution | Retained material |
| --- | --- | --- |
| [INSTANT](https://github.com/hieu-trannn/INSTANT), revision `8a04bd43c19cc52984a49fb1b7fd1dae14fb2ffc` | MIT, copyright 2024 Trung-Hieu Tran; [license](third_party/discovery/instant/LICENSE) | Selected source and README |
| [DropBP](https://github.com/WooSunghyeon/dropbp), revision `7170a637f66e4723b8e749b466061195a0cfe399` | MIT, copyright 2024 viqpldem; [license](third_party/discovery/dropbp/LICENSE) | Selected source and README |
| [VCAS](https://github.com/thu-ml/VCAS), revision `0bc7d0875de8098da53537af066a9f1921d16409` | Apache-2.0; [license](third_party/discovery/vcas/LICENSE). No root NOTICE exists at this pinned revision. | Selected source and README |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), revision `b2617bb6d3ffa2e429640350f613e3291e10b141` | MIT; [license](third_party/discovery/fashion_mnist/LICENSE) | README, tree metadata and license only; no image or label bodies |
| [WikiText](https://huggingface.co/datasets/Salesforce/wikitext), revision `b08601e04326c79dfdd32d625aee71d232d685c3` | Pinned card metadata lists CC-BY-SA-3.0 and GFDL; its prose links CC-BY-SA-4.0. The conflict remains unresolved. | Card and metadata only; no corpus bodies or official validation/test bytes |
| [PyTorch](https://github.com/pytorch/pytorch), revision `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e` | BSD-style terms with retained contributor notices; exact upstream [license](third_party/discovery/pytorch/LICENSE) | Selected `MPSFallback.mm`, not a reproducible whole build |

## Installed dependencies

The [wheel hashes](manifests/wheels.sha256), [environment manifest](manifests/environment.json) and 19 separate [wheel notice bundles](third_party/discovery/wheels) retain the measured environment's provenance and upstream notices, including bundled third-party terms. Runtime libraries are installed separately and keep their distribution licenses. Neither the project grants nor a single summary license identifier replaces those terms.

Development, experiments and reporting used AI assistance under Artem Legotin's direction. README editing used [write-like-me](https://github.com/arlegotin/write-like-me) at revision `b7a28e64fed749406a2fd8406d5ebd3eedd584c0`; the skill is not bundled here.
