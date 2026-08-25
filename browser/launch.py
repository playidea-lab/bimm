"""브라우저를 여는 문. **여기서 다시 쓰지 않고 코어의 것을 들여온다.**

코어가 같은 판단을 이미 한 번 했다(`borch-ts/test/launch.py`): 러너마다 갈린 인자로
브라우저를 띄우면 나란히 놓은 두 수가 같은 잣대가 아니게 된다. 특히 헤드리스는
**조용히 소프트웨어 래스터라이저**로 떨어지는데, 그 자리에서 잰 수를 timm 과 나란히
놓고 "같다"고 말하면 그것이 거짓이 되는 종류다.

플래그를 여기 베껴 두면 언젠가 갈린다. 그래서 옆에 나란히 받아둔 코어에서 가져오고,
없으면 멈춘다 — 몰래 우리 플래그로 띄우느니 안 도는 편이 낫다.

이 파일이 npm 설치본으로 안 되는 이유: `launch.py` 는 코어의 배포에 실리지 않는다.
대조 검사(`test/samemodel.test.ts`)와 달리 이 하네스는 **개발자 도구**라 CI 의 필수
경로가 아니고, 그래서 옆 저장소를 요구하는 것이 값을 치를 만하다.
"""

import importlib.util
import pathlib
import sys

CORE = pathlib.Path(__file__).resolve().parents[2] / "borch"
_PATH = CORE / "tests" / "browser" / "launch.py"

if not _PATH.exists():
    print(
        f"코어를 못 찾았다: {_PATH}\n"
        "  이 저장소 옆에 나란히 받아야 한다:\n"
        "    git clone git@github.com:playidea-lab/borch.git ../borch\n"
        "  브라우저 플래그를 여기서 새로 쓰지 않는 이유는 launch.py 첫 문단에 있다.",
        file=sys.stderr,
    )
    raise SystemExit(2)

_spec = importlib.util.spec_from_file_location("core_browser_launch", _PATH)
if _spec is None or _spec.loader is None:
    raise SystemExit(f"코어의 launch.py 를 못 읽었다: {_PATH}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

browser = _mod.browser
is_software = _mod.is_software
warn_if_software = _mod.warn_if_software
