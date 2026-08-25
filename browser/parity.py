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

# 골든의 허용 오차. 비트 일치는 이 프로젝트의 명시적 비목표이고, 브라우저의 fp32 는
# 곱셈 순서가 torch 와 다르다.
ATOL = 1e-4
RTOL = 1e-4


def _material(model_name: str, pretrained: bool, seed: int) -> dict[str, str]:
    """timm 을 세워 가중치·입력·기대 출력을 safetensors 하나에 담는다."""
    import timm
    import torch
    from safetensors.torch import save_file

    model = timm.create_model(model_name, pretrained=pretrained)
    model.eval()

    torch.manual_seed(seed)
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
    print(f"열쇠 {len(wanted)}개 — 이름이 timm 과 같다"
          f" (num_batches_tracked {tracked}개는 양쪽에서 뺐다 — 추론에 안 쓰인다)")

    got = np.asarray(result["got"], dtype=np.float64)
    want = np.asarray(result["want"], dtype=np.float64)
    if got.shape != want.shape:
        print(f"모양이 다르다 — {got.shape} 대 {want.shape}", file=sys.stderr)
        return 1

    gap = np.abs(got - want)
    tol = ATOL + RTOL * np.abs(want)
    worst = int(np.argmax(gap - tol))
    if np.any(gap > tol):
        print(f"수가 갈렸다 — [{worst}] {got[worst]:.9g} ≠ {want[worst]:.9g} "
              f"(최대 차 {gap.max():.3e})", file=sys.stderr)
        return 1

    print(f"수 {got.size}개 — 최대 절대차 {gap.max():.3e} (허용 {ATOL})")
    print(f"가장 큰 값의 자리도 같다 — {int(np.argmax(got))} 대 {int(np.argmax(want))}")
    return 0 if int(np.argmax(got)) == int(np.argmax(want)) else 1


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenetv2_100")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args(argv)

    meta = _material(args.model, args.pretrained, args.seed)
    httpd, port = _serve()
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw, browser_of(pw, headed=not args.headless) as browser:
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{port}/browser/parity.html")
            page.wait_for_function("window.__parity !== undefined", timeout=TIMEOUT_MS)
            result = json.loads(page.evaluate("JSON.stringify(window.__parity)"))
        return _compare(result, meta)
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
