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
import signal
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


def _material(model_name: str, pretrained: bool, seed: int, res: int = 224) -> dict[str, str]:
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

    x = torch.randn(1, 3, res, res)
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


class _Felled(Exception):
    """신호로 끊겼다. 정리 절차를 타려고 예외로 바꿔 던진다."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"signal {signum}")
        self.signum = signum


def _chromium_pids() -> set[int]:
    """지금 이 기계에 떠 있는 playwright 크로미움의 pid."""
    import subprocess

    out = subprocess.run(["pgrep", "-f", "ms-playwright/chromium"],
                         capture_output=True, text=True, check=False)
    return {int(line) for line in out.stdout.split() if line.isdigit()}


def _reap(before: set[int]) -> None:
    """**이 실행이 띄운** 브라우저만 거둔다.

    닫기가 매달려도, 예외로 빠져나가도, `timeout` 에 잘려도 한 번은 정리된다.
    남겨두면 다음 측정이 그것과 GPU 를 나눠 쓰게 되고, 그 측정은 이유 없이
    느려지거나 죽는다.

    ## 왜 `pkill -f ms-playwright/chromium` 이 아닌가

    전에는 그것이었고, **이 기계의 모든 playwright 브라우저를 죽였다** — 같은
    기계에서 도는 다른 세션의 것까지. 이 저장소는 워크트리 여럿이 한 기계를
    나눠 쓰므로 그것은 남의 측정을 중간에 끊는다.

    실측으로 걸렸다: 이 하네스를 한 번 돌렸더니 크로미움이 17 개에서 2 개로
    줄었고, 그중 대부분이 내 것이 아니었다. **고아가 쌓여 측정을 망친다는
    이야기의 절반은 그렇게 만들어진 것**일 수 있다.

    playwright 는 자기가 띄운 프로세스의 pid 를 안 알려준다. 그래서 시작 전에
    한 번 세어 두고, 끝날 때 **그때 없던 것만** 거둔다. 그 사이 다른 세션이
    새로 띄운 것을 잘못 잡을 여지는 남지만, 모두를 죽이는 것보다는 좁다.
    """
    import os
    import signal

    for pid in _chromium_pids() - before:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


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

    # 페이지가 견준 것을 받아 **판정은 여기서** 한다 — 허용치를 아는 쪽은 이쪽이다.
    atol = ATOL_CARGO if result.get("mode") == "cargo" else ATOL_MATERIAL
    worst = float(result["worst"])
    tol = atol + atol * abs(float(result["worstWant"]))
    if worst > tol:
        print(f"수가 갈렸다 — [{result['worstAt']}] {result['worstGot']:.9g} ≠ "
              f"{result['worstWant']:.9g} (최대 차 {worst:.3e}, 허용 {atol})",
              file=sys.stderr)
        for bad in result.get("offenders", [])[:5]:
            print(f"    [{bad['at']}] {bad['got']:.9g} ≠ {bad['want']:.9g}",
                  file=sys.stderr)
        return 1

    print(f"수 {result['count']}개 — 최대 절대차 {worst:.3e} (허용 {atol}) "
          f"· 평균 {float(result['meanGap']):.3e}")
    same = result["argmaxGot"] == result["argmaxWant"]
    print(f"가장 큰 값의 자리{'도 같다' if same else '가 다르다'} — "
          f"{result['argmaxGot']} 대 {result['argmaxWant']}")
    return 0 if same else 1


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenetv2_100")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # 입력 크기는 중간 활성화의 크기를 정한다 — 브라우저가 어디서 버티지 못하는지
    # 가를 때 쓴다. 대조 자체는 어떤 크기에서도 성립한다(양쪽에 같은 것을 넣는다).
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--cargo", action="store_true",
                    help="레지스트리에 올릴 화물 그대로 싣는다 (scripts/export.py 가 만든 것)")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args(argv)

    # 화물 모드는 재료를 새로 담지 않는다 — 이미 만들어 둔 것을 그대로 실어야
    # "올릴 파일이 실린다" 는 말이 성립한다.
    meta = {} if args.cargo else _material(args.model, args.pretrained, args.seed, args.res)
    httpd, port = _serve()
    # **브라우저를 띄우기 전에 한 번 센다.** 끝날 때 거둘 것을 이 차집합으로 고른다 —
    # playwright 가 자기 프로세스의 pid 를 안 알려주기 때문이다.
    before = _chromium_pids()
    # **신호로 죽을 때도 거둔다.**
    #
    # `finally` 만으로는 부족하다 — 파이썬은 SIGTERM 을 받으면 정리 절차를 안 돌고
    # 그 자리에서 끝난다(실측: `timeout` 에 잘린 프로세스에서 `finally` 가 안 돌았다).
    # 그런데 이 하네스가 고아를 남기는 경우가 **정확히 그 경우**다. `timeout` 으로
    # 자르거나 사람이 끊을 때다.
    #
    # 그래서 두 신호를 받아 예외로 바꾼다. 그러면 `finally` 가 돌고, 종료 코드는
    # 신호로 죽은 것과 같게 남긴다 — 부르는 쪽이 "잘렸다" 를 구별할 수 있어야 한다.
    def _felled(signum: int, _frame: object) -> None:
        raise _Felled(signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _felled)
    # **어디까지 갔는지 말한다.** 이 하네스가 조용히 매달린 적이 있고, 그때 로그는
    # "재료를 담았다" 에서 끝나 있었다 — 브라우저를 여는 중인지, 페이지를 기다리는
    # 중인지 알 수가 없었다. 한 줄씩 흘려보내면 멈춘 자리가 마지막 줄이 된다.
    say = lambda what: print(f"  … {what}", flush=True)
    say(f"서버 {port}")
    try:
        from playwright.sync_api import sync_playwright

        say("playwright 진입")
        with sync_playwright() as pw:
            say("브라우저 여는 중")
            with browser_of(pw, headed=not args.headless) as browser:
                say("페이지 만드는 중")
                page = browser.new_page()
                # **콘솔을 읽는다.** 코어는 디바이스를 잃으면 거기에 남기는데, 그것을
                # 안 읽으면 원인이 화면 밖에 남는다.
                page.on("console", lambda m: (
                    print(f"  [browser] {m.text}", flush=True)
                    if m.type in ("error", "warning") else None))
                url = (f"http://127.0.0.1:{port}/browser/parity.html"
                       f"{'?cargo=' + args.model if args.cargo else ''}")
                say(f"이동 {url}")
                page.goto(url)
                say("페이지가 답하기를 기다린다")
                page.wait_for_function("window.__parity !== undefined", timeout=TIMEOUT_MS)
                say("받았다 — 조각내어 가져온다")
                # **한 번에 가져오면 매달린다.** 큰 배열이 든 객체를 통째로
                # 직렬화해 CDP 로 넘기다 멈추는 것을 실측했다(efficientnet_b1,
                # 네 번 연속). 조각으로 나누면 지나가고, 멈추더라도 어느 열쇠에서
                # 멈췄는지가 남는다.
                result = {}
                for key in page.evaluate("Object.keys(window.__parity)"):
                    say(f"조각 {key}")
                    raw = page.evaluate(
                        f"JSON.stringify(window.__parity[{json.dumps(key)}])")
                    # `JSON.stringify(undefined)` 는 undefined 를 돌려주고 여기서
                    # None 이 된다. 그것을 그대로 json.loads 에 넣으면 진단하려던
                    # 실행이 **진단 코드 때문에** 죽는다(실측).
                    result[key] = None if raw is None else json.loads(raw)

                # **판정을 여기서 끝낸다.** 브라우저를 닫는 데서 매달리는 것을
                # 실측했다 — `with` 를 빠져나가야 판정이 돌게 두면, 결과를 다 받아
                # 놓고도 아무 말 없이 멈춘다. 닫기는 그 뒤에 하든 말든 이미 늦었다.
                verdict = _compare(result, meta)

        # **닫히기를 기다리지 않는다.** 판정은 위에서 이미 났고, `browser.close()` 가
        # 매달리는 것을 실측했다(결과를 다 받아 놓고 timeout 으로 죽는다). 거두는
        # 것은 아래 `finally` 가 한다 — 여기서 하면 정상 종료에만 닿는다.
        return verdict
    except _Felled as felled:
        # 신호로 잘렸다. 거두는 것은 아래 `finally` 가 하고, 여기서는 종료 코드만
        # 신호에 맞춰 남긴다.
        print(f"  … 신호 {felled.signum} 로 끊겼다 — 브라우저를 거둔다", flush=True)
        return 128 + felled.signum
    finally:
        # **어떻게 끝나든 거둔다.** 전에는 정상 종료 경로에만 있었고, 그래서
        # 이 하네스가 고아를 남기는 경우가 정확히 **거두지 않는 경우**였다 —
        # `timeout` 에 잘리거나, 페이지가 던지거나, 사람이 끊을 때다.
        #
        # 남은 브라우저는 조용히 새지 않는다. GPU 를 물고 있어서 **다음 측정을
        # 망친다** — 한 번은 아홉 개가 쌓여 있어서 혼자 0.5 초에 끝나는 모델이
        # 아무 줄도 못 찍고 죽었고, 그것을 모델 탓으로 읽을 뻔했다.
        _reap(before)
        httpd.shutdown()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
