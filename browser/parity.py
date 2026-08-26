"""timm 이 낸 수와 **여기 옮긴 것이 낸 수가 같은지** 본다.

    uv run --with playwright --with timm --with torch --with safetensors --with numpy \
      python browser/parity.py --headed

## 왜 이것이 이 저장소의 중심 검사인가

카탈로그가 하는 말은 "이 이름을 부르면 timm 의 그 아키텍처가 선다" 하나다. 층을 옳게
옮겼는지는 코드를 읽어서는 안 보인다 — depthwise 의 groups 하나, 좁히는 쪽의 ReLU6
하나가 어긋나도 **모델은 서고 수만 틀린다.** 그러면 가중치는 모양이 맞으니 실리고,
실린 다음 틀린 수를 낸다.

그래서 timm 을 실제로 세워 가중치·입력·출력을 받아 오고, 같은 가중치를 같은 입력에
통과시켜 **수를 나란히 놓는다.**

## 가중치를 왜 랜덤으로도 도는가

`--pretrained` 없이 돌면 timm 이 갓 초기화한 값을 쓴다. 학습된 값이 아니어도 **구조가
같은지는 똑같이 증명된다** — 오히려 더 가혹하다. 사전학습 가중치는 층이 조금 어긋나도
그럴듯한 수를 내는 쪽으로 학습돼 있지만, 랜덤 값은 어긋난 자리를 그대로 드러낸다.

## `num_batches_tracked` 는 안 싣는다

학습 횟수를 세는 수라 추론에 안 쓰인다. safetensors 로 0 차원 텐서를 왕복시키는 데도
값이 드는데, 그 값을 치를 이유가 없다. 대신 브라우저 쪽 `loadStateDict` 를 strict 가
아니게 부른다 — **열쇠 대조는 따로 하므로** 그 느슨함이 무엇을 가리지 않는다.
"""

import argparse
import http.server
import json
import pathlib
import socketserver
import sys
import threading

from launch import browser as browser_of, warn_if_software

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
TIMEOUT_MS = 300_000

# 허용 오차. 비트 일치는 이 프로젝트의 명시적 비목표이고, 브라우저의 fp32 는 곱셈
# 순서가 torch 와 다르다.
#
# **두 모드는 다른 것을 잰다.**
#
# 재료 모드(랜덤 초기화)가 재는 것은 **구조**다. 층 하나가 어긋나면 랜덤 값은 그것을
# 그대로 드러낸다. 실측:
#
#     mobilenetv2_100        5.2e-06
#     mobilenetv3_large_100  2.1e-08
#     mobilenetv3_small_100  1.9e-09
#
# 이 수들은 시드를 모델보다 먼저 걸고 나서야 **재현된다.** 그 전에는 같은 명령이
# 6.2e-06 과 7.6e-06 을 번갈아 냈다.
#
# 화물 모드(사전학습)가 재는 것은 **이 바이트가 원본의 수를 재현하는가**다. 구조는
# 재료 모드가 이미 봤고, 여기서 커지는 것은 학습된 값의 크기와 BatchNorm 의 분산이
# 오차를 키우기 때문이다 — 같은 코드에 가중치만 바꿔 확인했다:
#
#     mobilenetv3_small_100  랜덤 1.0e-09 · 사전학습 1.6e-05
#
# 그래서 하나의 수로 둘을 재면 한쪽이 반드시 틀린다. 1e-5 하나로 조였더니 멀쩡한
# 화물(1.6e-05)이 걸렸고, 1e-4 하나로 두면 재료 쪽이 실측보다 두 자리 위가 된다.
#
# 재료 1e-5 는 실측 최대(5.2e-06)의 1.9 배다. 좁게 잡은 까닭은 **구조가 틀리면 랜덤
# 값에서 두 자리 아래로는 절대 안 나오기 때문**이다 — 여유를 크게 두어 얻는 것이
# 없다. 화물 5e-5 는 실측 최대(1.6e-05)의 3 배다.
#
# **여기서 한 번 틀린 인과를 적었다.** 시드를 고치기 전에는 V3 의 머리를 뒤집어도
# 6.9e-05 로 통과했고, 그것이 허용치를 조이는 근거로 적혀 있었다. 시드를 모델 앞으로
# 옮기고 같은 실험을 하니 5.0e-01 로 갈린다 — 그 "거의 통과" 는 그날 뽑힌 가중치의
# 우연이었지 허용치 문제가 아니었다. 진짜로 헐거웠던 것은 **재료가 재현되지 않는
# 것**이었고, 조인 허용치는 그것과 별개로 정당한 채 남았다.
#
# 여유가 모자라는 날이 오면 **허용치를 늘리기 전에 그 모델을 재료 모드로 먼저 볼 것.**
# 구조가 맞다면 그쪽은 두 자리 아래에서 통과한다.
ATOL_MATERIAL = 1e-5
ATOL_CARGO = 5e-5


def _material(model_name: str, pretrained: bool, seed: int) -> dict[str, str]:
    """timm 을 세워 가중치·입력·기대 출력을 safetensors 하나에 담는다."""
    import timm
    import torch
    from safetensors.torch import save_file

    # **시드가 모델보다 먼저다.** 입력 직전에만 걸면 랜덤 초기화 가중치가 실행마다
    # 달라지고, 그러면 이 검사가 내는 수도 달라진다 — 같은 코드에서 6.2e-06 과
    # 7.6e-06 이 번갈아 나왔다. 재현되지 않는 수 위에는 허용치를 세울 수 없다.
    torch.manual_seed(seed)
    model = timm.create_model(model_name, pretrained=pretrained)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)

    tensors = {
        k: v.contiguous()
        for k, v in model.state_dict().items()
        if not k.endswith("num_batches_tracked")
    }
    tensors["__input"] = x.contiguous()
    tensors["__output"] = y.contiguous()

    meta = {
        "library": "timm",
        "factory": model_name,
        "numClasses": str(model.num_classes),
        "pretrained": "1" if pretrained else "0",
    }
    OUT.mkdir(exist_ok=True)
    save_file(tensors, str(OUT / "parity.safetensors"), metadata=meta)
    print(f"재료를 담았다 — {model_name}"
          f"{' (pretrained)' if pretrained else ' (랜덤 초기화)'}"
          f" · 열쇠 {len(tensors) - 2}개")
    return meta


class _Quiet(http.server.SimpleHTTPRequestHandler):
    """요청 한 줄씩 찍지 않는다 — 여기서 중요한 것은 마지막의 판정뿐이다."""

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, *a):
        pass


def _serve() -> tuple[socketserver.TCPServer, int]:
    httpd = socketserver.TCPServer(("127.0.0.1", 0), _Quiet)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def _compare(result: dict, meta: dict[str, str]) -> int:
    import numpy as np

    if "error" in result:
        print(f"브라우저에서 멈췄다: {result['error']}", file=sys.stderr)
        return 1

    print(f"어댑터 — {result.get('adapter')}")
    if result.get("mode") == "cargo":
        # strict 로 실렸다는 것이 화물 검사의 절반이다. 열쇠가 하나라도 남거나
        # 모자라면 브라우저가 그 자리에서 멈추므로, 여기까지 온 것 자체가 판정이다.
        print(f"화물 열쇠 {result.get('loaded')}개를 strict 로 실었다"
              f" — 허브의 로더와 같은 부름이다")
    warn_if_software(result.get("adapter"), "이 대조")

    # **우리가 안 실은 것을 갈렸다고 세면 안 된다.** `num_batches_tracked` 는 위에서
    # 일부러 뺐고, 그것을 여기서 빼지 않으면 검사가 자기가 만든 차이를 발견한다.
    def bare(keys: list[str]) -> list[str]:
        return [k for k in keys if not k.endswith("num_batches_tracked")]

    tracked = len(result["wanted"]) - len(bare(result["wanted"]))
    wanted, given = bare(result["wanted"]), bare(result["given"])
    if wanted != given:
        missing = sorted(set(wanted) - set(given))
        extra = sorted(set(given) - set(wanted))
        print("열쇠가 갈렸다 — 옮긴 이름이 timm 과 다르다", file=sys.stderr)
        if missing:
            print(f"  모델이 부르는데 timm 에 없다 ({len(missing)}): {missing[:6]}", file=sys.stderr)
        if extra:
            print(f"  timm 이 주는데 모델이 안 받는다 ({len(extra)}): {extra[:6]}", file=sys.stderr)
        return 1
    if tracked:
        print(f"열쇠 {len(wanted)}개 — 이름이 timm 과 같다"
              f" (num_batches_tracked {tracked}개는 대조에서 뺐다 — 추론에 안 쓰인다)")
    else:
        print(f"열쇠 {len(wanted)}개 — 이름이 timm 과 같다")

    got = np.asarray(result["got"], dtype=np.float64)
    want = np.asarray(result["want"], dtype=np.float64)
    if got.shape != want.shape:
        print(f"모양이 다르다 — {got.shape} 대 {want.shape}", file=sys.stderr)
        return 1

    gap = np.abs(got - want)
    atol = ATOL_CARGO if result.get("mode") == "cargo" else ATOL_MATERIAL
    tol = atol + atol * np.abs(want)
    worst = int(np.argmax(gap - tol))
    if np.any(gap > tol):
        print(f"수가 갈렸다 — [{worst}] {got[worst]:.9g} ≠ {want[worst]:.9g} "
              f"(최대 차 {gap.max():.3e}, 허용 {atol})", file=sys.stderr)
        return 1

    print(f"수 {got.size}개 — 최대 절대차 {gap.max():.3e} (허용 {atol})")
    print(f"가장 큰 값의 자리도 같다 — {int(np.argmax(got))} 대 {int(np.argmax(want))}")
    return 0 if int(np.argmax(got)) == int(np.argmax(want)) else 1


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenetv2_100")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cargo", action="store_true",
                    help="레지스트리에 올릴 화물 그대로 싣는다 (scripts/export.py 가 만든 것)")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args(argv)

    # 화물 모드는 재료를 새로 담지 않는다 — 이미 만들어 둔 것을 그대로 실어야
    # "올릴 파일이 실린다" 는 말이 성립한다.
    meta = {} if args.cargo else _material(args.model, args.pretrained, args.seed)
    httpd, port = _serve()
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw, browser_of(pw, headed=not args.headless) as browser:
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{port}/browser/parity.html"
                      f"{'?cargo=' + args.model if args.cargo else ''}")
            page.wait_for_function("window.__parity !== undefined", timeout=TIMEOUT_MS)
            result = json.loads(page.evaluate("JSON.stringify(window.__parity)"))
        return _compare(result, meta)
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
