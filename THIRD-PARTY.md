# Other people's work — what this leans on, and what has to be honoured

`bimm` is Apache-2.0, Copyright 2026 PLAYIDEALAB Inc. What follows is **what came
from somewhere else**, and its terms. The licences are not written from memory —
each was read out of the source it came from.

---

## timm — the architectures

| | licence | how it was confirmed |
|---|---|---|
| **timm** (`huggingface/pytorch-image-models`) | Apache-2.0 | `LICENSE` at the repository root |

**Every architecture under `timm/` was transcribed from timm**, not from the
papers. The catalogue's own listing is the current one; at the time of writing:

| here | from timm |
|---|---|
| `MobileNetV2` | `mobilenetv2_100` |
| `MobileNetV3` | `mobilenetv3_large_100` · `mobilenetv3_small_100` |
| `EfficientNet` | `efficientnet_b0` · `b1` · `b2` · `b3` |
| `VisionTransformer` | `vit_tiny_patch16_224` |

`borchvision/resnet18_cifar` is not among them — it is written here, and the
namespace says so.

Channels, strides, expansion ratios and where the activations sit were dumped
out of a live `timm.create_model(...)` rather than read off a paper — paper and
implementation genuinely diverge in places, and what has to load is the
implementation.

**The field names are timm's on purpose.** `conv_stem`, `conv_pwl`, `bn3` are not
this repository's naming. `stateDict` keys come from field names, so using
timm's names means a timm checkpoint loads **under its own keys** — there is no
mapping table, and therefore no table to be wrong. It is the one place here that
leaves camelCase, and there the name is a contract rather than a taste.

### What Apache-2.0 asks of us, and what it does not

- **A copy of the licence and the copyright notice** travel with anything derived.
  `LICENSE` is at the root of this repository and ships in the npm tarball.
- **Changed files say they changed.** These are not modified copies of timm's
  files; they are an implementation in another language, and every file says at
  the top where its structure came from.
- **A NOTICE file propagates if the original has one.** **timm has no NOTICE
  file** — checked against the repository's root listing — so §4(d) does not
  come into play. This repository does not add one either: a NOTICE would put an
  obligation on everyone downstream, and there is nothing here that needs it.

---

## timm — the weights

**This repository holds no weights.** `scripts/export.py` fetches pretrained
weights through timm and writes them to `out/cargo/`, which is ignored; from
there they go to a CDN with their digest recorded in a manifest.

**Whoever redistributes those bytes is bound by their terms**, and the terms are
not one thing:

| | |
|---|---|
| what timm declares | `apache-2.0`, read out of `model.default_cfg["license"]` and recorded in each entry's `provenance.md` |
| what the data says | ImageNet-1k's own terms are **research use** |

Those are two separate statements and the second does not disappear because the
first is permissive. Whether a licence on weights carries the terms of the data
they were trained on is **not settled**, and nothing here should be read as an
opinion on it. Each registry manifest states both fields so that the reader can
see them rather than infer one from the other.

---

## The runtime

| | licence | how it was confirmed |
|---|---|---|
| **borch-ts** | Apache-2.0 | same authors, `LICENSE` in that repository |

It is a `peerDependency` — this package does not carry a copy of it. What that
means for whoever puts it in a browser is written in the core's
[THIRD-PARTY.md](https://github.com/playidea-lab/borch/blob/main/THIRD-PARTY.md);
Pyodide is MPL-2.0 and being served in executable form comes with an obligation.
