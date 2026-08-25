/**
 * 이름 하나로 모델 구조를 되살린다 — **timm 의 자리다.**
 *
 * `borch` 가 torch 이고 `borchvision` 이 torchvision 인 것과 같은 뜻으로, 이 패키지는
 * timm 이다: 아키텍처 카탈로그와, 이름으로 그것을 만들어 주는 함수 하나.
 *
 * ## 이름이 timm 과 갈리는 한 자리, 그리고 그것이 일부러인 까닭
 *
 * timm 은 `create_model("resnet18")` 처럼 **이름 하나**를 받는다. 여기는 둘을 받는다
 * — `createModel("borchvision", "resnet18_cifar", …)`.
 *
 * 흉내를 깨는 자리이므로 적어 둔다. 실제 생태계에서 torchvision 의 `resnet18` 과
 * timm 의 `resnet18` 은 **다른 모델이고 가중치가 안 호환된다.** timm 은 이름공간이
 * 없어서 그 충돌을 문서로만 다루고, 우리는 카탈로그가 두 라이브러리를 동시에 들도록
 * 만들어졌다. 이름공간 없이 시작하면 **이미 배포된 매니페스트**가 그 이름을 박은
 * 뒤에는 못 고친다.
 *
 * 즉 이것은 timm 을 덜 흉내 낸 것이 아니라, timm 이 나중에 아쉬워한 자리를 먼저
 * 잡은 것이다. 그렇게 읽히도록 여기 적는다.
 *
 * ## 패키지가 어디 있는지와 `library` 이름은 별개다
 *
 * 이 파일이 `bimm` 안에 있다고 해서 여기 실린 것이 전부 `bimm` 의 모델은 아니다.
 * `library` 는 **아키텍처의 출신**을 가리키는 규약이고, 코드가 사는 곳은 그것과
 * 상관없다. 그래서 `bimm` 이 `borchvision/resnet18_cifar` 를 들고 있는 것은 모순이
 * 아니라 설계다.
 *
 * **이름을 지우거나 뜻을 바꾸면 그 이름을 적어둔 매니페스트가 전부 죽는다** —
 * 늘리는 것은 되지만 줄이는 것은 안 된다.
 */

import { nn } from "borch-ts";

import { BimmError } from "./errors.js";
import { checkArgs, type FactoryArgs } from "./args.js";
import { MobileNetV2 } from "./mobilenet.js";
import { ResNet18Cifar } from "./resnet.js";

interface Factory {
  readonly spec: FactoryArgs;
  readonly build: (args: Readonly<Record<string, number>>) => nn.Module;
}

/** 열쇠는 `library/factory` 다 — 표를 찾는 데만 쓰고 밖으로는 안 내보낸다. */
const FACTORIES: Readonly<Record<string, Factory>> = {
  "borchvision/resnet18_cifar": {
    spec: { numClasses: { kind: "int", min: 1 } },
    // `?? 1` 은 도달하지 않는다 — checkArgs 가 없는 인자를 이미 거절했다.
    // noUncheckedIndexedAccess 를 켠 값이 이런 자리를 눈에 보이게 하는 것이다.
    build: (args) => new ResNet18Cifar(args["numClasses"] ?? 1),
  },
  // timm 에서 옮겨 온 첫 아키텍처. 이름이 `timm/` 아래인 것은 **출신을 가리키는
  // 규약**이고, 같은 표에 `borchvision/` 이 나란히 서는 지금이 이름공간을 둘로 받은
  // 까닭이 처음으로 눈에 보이는 자리다 — timm 의 resnet18 과 torchvision 의
  // resnet18 은 실제로 다른 모델이고 가중치가 안 호환된다.
  "timm/mobilenetv2_100": {
    spec: { numClasses: { kind: "int", min: 1 } },
    build: (args) => new MobileNetV2(args["numClasses"] ?? 1),
  },
};

export interface FactoryName {
  readonly library: string;
  readonly factory: string;
}

/** 카탈로그에 있는 것들. 발견 레이어와 오류 문구가 같은 표를 본다. */
export function listModels(): readonly FactoryName[] {
  return Object.keys(FACTORIES).sort().map((key) => {
    const cut = key.indexOf("/");
    return { library: key.slice(0, cut), factory: key.slice(cut + 1) };
  });
}

function shown(): string {
  return listModels().map((f) => `${f.library}/${f.factory}`).join(", ");
}

function find(library: string, factory: string): Factory {
  const found = FACTORIES[`${library}/${factory}`];
  if (found === undefined) {
    throw new BimmError(
      `unknown factory: ${library}/${factory}\n  catalogue: ${shown()}`,
    );
  }
  return found;
}

/** 이 팩토리가 받는 인자 규격. 매니페스트를 쓰는 쪽이 물어볼 수 있어야 한다. */
export function factorySpec(library: string, factory: string): FactoryArgs {
  return find(library, factory).spec;
}

/**
 * 이름과 인자로 실제 모델을.
 *
 * **`await init()` 이 먼저다.** 층이 곧 텐서이고 텐서는 WebGPU 어댑터 위에 선다.
 * 안 부르고 여기 오면 코어가 그 자리에서 멈춘다 — 그 진단을 가로채 우리 말로
 * 바꾸지 않는다. 원인은 코어 쪽이고, 코어의 문구가 더 정확하다.
 */
export function createModel(
  library: string,
  factory: string,
  args: Readonly<Record<string, unknown>> = {},
): nn.Module {
  const found = find(library, factory);
  return found.build(checkArgs(`${library}/${factory}`, found.spec, args));
}
