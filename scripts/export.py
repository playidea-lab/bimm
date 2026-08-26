"""timm 의 사전학습 가중치를 **레지스트리 화물로 만든다.**

    uv run --with timm --with torch --with safetensors python scripts/export.py \
      --model mobilenetv2_100

`out/cargo/` 에 넷을 놓는다 — `model.safetensors`, `sample.in.safetensors`,
`sample.out.safetensors`, `summary.json`. 그다음은 레지스트리의 `scripts/pack.py`
(`--origin converted-from-torch`) 가 매니페스트로 만든다.

## 왜 여기서 매니페스트를 안 쓰는가

매니페스트의 주인은 레지스트리다. 형식을 아는 곳이 둘이 되면 **둘은 갈리고**, 갈린
쪽이 병합된 뒤에 받는 쪽에서 드러난다. 여기는 timm 을 아는 쪽이므로 timm 에서 꺼낼
수 있는 것만 꺼낸다.

## `num_batches_tracked` 를 float32 로 바꾼다

허브의 로더는 `loadStateDict` 를 **strict 로** 부른다. 그래서 모델이 부르는 열쇠가
전부 있어야 하고, 코어의 BatchNorm 은 이 열쇠도 내보낸다 — 빼면 화물이 안 실린다.

그런데 torch 는 이것을 int64 로 저장하고, 코어의 `decode` 는 헤더의 dtype 을 보지
않고 **바이트를 4 로 나눠** float32 로 읽는다. 8 바이트 스칼라는 "원소 2 개" 로 세어져
`shape []` 와 안 맞는다며 멈춘다(실측). 값 자체는 학습 횟수라 추론에 안 쓰이므로,
**바꾸는 쪽이 옳고 그 사실을 여기 적는다.**

## 입력 샘플이 왜 랜덤인가

`sample.in` 은 이미 전처리를 마친 텐서다 — 이미지가 아니다. 그래서 어떤 값이든
재현 검사의 목적을 채우고, 고정 시드가 있으면 언제든 다시 만들 수 있다.

**`sample.out` 은 timm 이 낸 수다.** 우리가 낸 수를 적으면 검사가 자기 답안을
채점한다. 이렇게 두면 허브가 이 화물을 실을 때마다 **timm 의 수를 재현하는지**를
보게 된다.
"""

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
def cargo_dir(model: str) -> pathlib.Path:
    """화물은 모델마다 자기 자리에 놓는다. 한 자리를 쓰면 두 번째 export 가 첫 번째를
    덮고, 덮인 줄 모른 채 매니페스트를 만들면 **다른 모델의 바이트에 이 모델의 이름이
    붙는다.**"""
    return ROOT / "out" / "cargo" / model


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenetv2_100")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import timm
    import torch
    from safetensors.torch import save_file

    model = timm.create_model(args.model, pretrained=True)
    model.eval()
    cfg = model.default_cfg

    torch.manual_seed(args.seed)
    x = torch.randn(1, 3, *cfg["input_size"][1:])
    with torch.no_grad():
        y = model(x)

    # int64 스칼라 하나 때문에 화물 전체가 안 실린다 — 위 문단을 보라.
    weights = {
        k: (v.float() if v.dtype == torch.int64 else v).contiguous()
        for k, v in model.state_dict().items()
    }

    CARGO = cargo_dir(args.model)
    CARGO.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(CARGO / "model.safetensors"),
              metadata={"source": f"timm {timm.__version__} / {args.model}"})
    save_file({"input": x.contiguous()}, str(CARGO / "sample.in.safetensors"))
    save_file({"output": y.contiguous()}, str(CARGO / "sample.out.safetensors"))

    raw = (CARGO / "model.safetensors").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    # 학습 이력이 아니라 **출처**를 적는다. 이 화물은 우리가 학습한 것이 아니다.
    summary = {
        "origin": "converted-from-torch",
        "sha256": digest,
        "bytes": len(raw),
        "source": {
            "library": "timm",
            "libraryVersion": timm.__version__,
            "model": args.model,
            "pretrainedTag": cfg.get("tag"),
            "checkpointUrl": cfg.get("hf_hub_id") or cfg.get("url"),
            "license": cfg.get("license"),
        },
        "preprocess": {
            "inputSize": list(cfg["input_size"]),
            "mean": list(cfg["mean"]),
            "std": list(cfg["std"]),
            # **크기만 맞춰서는 안 된다.** timm 은 짧은 변을 `input/crop_pct` 로 키운
            # 뒤 가운데를 자른다. 이것을 안 넘기면 받는 쪽이 이미지를 늘려 넣고,
            # 모델은 실리는데 이름이 틀리게 나온다 — 왕복 검사에서 실측으로 걸렸다.
            "cropPct": cfg.get("crop_pct"),
            "resizeShortSide": int(cfg["input_size"][1] / cfg["crop_pct"]),
            # timm 이 쓰는 보간. 코어와 레지스트리 스키마가 아는 것은 bilinear·nearest
            # 뿐이라 bicubic 은 그대로 못 옮긴다 — 무엇을 못 옮겼는지 남긴다.
            "interpolation": cfg.get("interpolation"),
        },
        "numClasses": model.num_classes,
        "publishedTop1": cfg.get("top1"),
        "sampleSeed": args.seed,
        "keys": len(weights),
    }
    (CARGO / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    # 자리마다의 이름. **첫 이름만 적는다** — timm 이 주는 것은 synset 설명 전체라
    # ("tench, Tinca tinca") 1000 자리를 그대로 실으면 매니페스트가 30KB 가 된다.
    # 받는 쪽이 argmax 를 사람 말로 바꾸는 데는 첫 이름이면 되고, 전체는 timm 에 있다.
    try:
        from timm.data import ImageNetInfo

        info = ImageNetInfo()
        names = [info.index_to_description(i, detailed=False).split(",")[0].strip()
                 for i in range(model.num_classes)]
        (CARGO / "classes.txt").write_text("\n".join(names) + "\n")
        print(f"  자리 이름 {len(names)}개 — classes.txt")
    except Exception as exc:  # noqa: BLE001
        # 이름을 못 얻는 데이터셋이면 여기서 멈추지 않는다. 매니페스트를 만드는 쪽이
        # 그 사실을 말할 것이고, 그 말은 여기서 지어내는 이름보다 정확하다.
        print(f"  자리 이름을 못 얻었다 ({exc}) — pack 이 --classes 를 요구할 것이다")

    print(f"화물을 놓았다: {CARGO.relative_to(ROOT)}")
    print(f"  {args.model} · 열쇠 {len(weights)}개 · {len(raw):,} 바이트")
    print(f"  sha256 {digest}")
    print(f"  전처리 {summary['preprocess']['inputSize']} mean={summary['preprocess']['mean']}")
    print(f"  출처 {summary['source']['checkpointUrl']} ({summary['source']['license']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
