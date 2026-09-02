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
import {
  efficientnetB0, efficientnetB1, efficientnetB2, efficientnetB3,
  efficientnetB4, efficientnetB5, efficientnetB6,
} from "./efficientnet.js";
import { MobileNetV2 } from "./mobilenet.js";
import { mobilenetv3Large, mobilenetv3Small } from "./mobilenetv3.js";
import { vitBasePatch16, vitSmallPatch16, vitTinyPatch16 } from "./vit.js";
import { ResNet18Cifar } from "./resnet.js";
import {
  resnet18, resnet34, resnet50, resnet101, resnet152,
} from "./resnet50.js";

/**
 * `numClasses` 의 위쪽 끝.
 *
 * 실제 상한이 아니라 **오타를 거르는 자리다.** ImageNet-21k 가 21841 이고 그보다 큰
 * 공개 분류 과제는 흔하지 않으므로, 이 수를 넘는 값은 자리를 잘못 누른 쪽일 가능성이
 * 훨씬 크다. 진짜로 더 필요해지는 날 이 수를 올리면 된다 — 올리는 것은 되고 내리는
 * 것은 안 되는 방향이다.
 */
const MAX_CLASSES = 100_000;

interface Factory {
  readonly spec: FactoryArgs;
  readonly build: (args: Readonly<Record<string, number>>) => nn.Module;
}

/** 열쇠는 `library/factory` 다 — 표를 찾는 데만 쓰고 밖으로는 안 내보낸다. */
const FACTORIES: Readonly<Record<string, Factory>> = {
  "borchvision/resnet18_cifar": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    // `?? 1` 은 도달하지 않는다 — checkArgs 가 없는 인자를 이미 거절했다.
    // noUncheckedIndexedAccess 를 켠 값이 이런 자리를 눈에 보이게 하는 것이다.
    build: (args) => new ResNet18Cifar(args["numClasses"] ?? 1),
  },
  // timm 에서 옮겨 온 첫 아키텍처. 이름이 `timm/` 아래인 것은 **출신을 가리키는
  // 규약**이고, 같은 표에 `borchvision/` 이 나란히 서는 지금이 이름공간을 둘로 받은
  // 까닭이 처음으로 눈에 보이는 자리다 — timm 의 resnet18 과 torchvision 의
  // resnet18 은 실제로 다른 모델이고 가중치가 안 호환된다.
  "timm/mobilenetv2_100": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => new MobileNetV2(args["numClasses"] ?? 1),
  },
  // 두 판이 같은 빌더에서 나온다 — 갈리는 것은 블록 표와 머리 채널뿐이다.
  "timm/mobilenetv3_large_100": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => mobilenetv3Large(args["numClasses"] ?? 1),
  },
  "timm/mobilenetv3_small_100": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => mobilenetv3Small(args["numClasses"] ?? 1),
  },
  // 합성곱이 아닌 첫 아키텍처. 표에 이름 하나가 늘어난 것으로 보이지만, 뼈대가
  // 다르므로 `vit.ts` 첫 문단을 읽고 손대는 편이 낫다.
  "timm/vit_tiny_patch16_224": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => vitTinyPatch16(args["numClasses"] ?? 1),
  },
  // small·base 는 tiny 와 `dim`·`heads` 둘만 다르다 — 깊이도 eps 도 같다.
  "timm/vit_small_patch16_224": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => vitSmallPatch16(args["numClasses"] ?? 1),
  },
  "timm/vit_base_patch16_224": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => vitBasePatch16(args["numClasses"] ?? 1),
  },
  // **이 카탈로그의 첫 ImageNet ResNet.** 위의 `borchvision/resnet18_cifar` 와
  // 이름이 비슷하지만 다른 물건이다 — 스템부터 다르고 가중치가 안 호환된다.
  // 이름공간을 둘로 받은 설계가 여기서 처음으로 값을 한다.
  "timm/resnet18": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => resnet18(args["numClasses"] ?? 1),
  },
  "timm/resnet34": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => resnet34(args["numClasses"] ?? 1),
  },
  "timm/resnet50": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => resnet50(args["numClasses"] ?? 1),
  },
  "timm/resnet101": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => resnet101(args["numClasses"] ?? 1),
  },
  "timm/resnet152": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => resnet152(args["numClasses"] ?? 1),
  },
  // 네 판이 같은 표에서 나온다 — 갈리는 것은 width·depth 두 수뿐이다.
  "timm/efficientnet_b0": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => efficientnetB0(args["numClasses"] ?? 1),
  },
  "timm/efficientnet_b1": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => efficientnetB1(args["numClasses"] ?? 1),
  },
  "timm/efficientnet_b2": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => efficientnetB2(args["numClasses"] ?? 1),
  },
  "timm/efficientnet_b3": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => efficientnetB3(args["numClasses"] ?? 1),
  },
  "timm/efficientnet_b4": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => efficientnetB4(args["numClasses"] ?? 1),
  },
  "timm/efficientnet_b5": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => efficientnetB5(args["numClasses"] ?? 1),
  },
  "timm/efficientnet_b6": {
    spec: { numClasses: { kind: "int", min: 1, max: MAX_CLASSES } },
    build: (args) => efficientnetB6(args["numClasses"] ?? 1),
  },
  // **b7 은 아직 여기 없다.** 계획은 timm 과 맞고 죽지도 않지만 **218 초**가 걸린다.
  //
  // b6 가 돌아온 것은 코어가 셰이더 진단을 끄면서다(borch#132, `borch-ts@0.2.6`).
  // 파이프라인마다 `getCompilationInfo()` 를 기다리지도 않고 걸어서, 프로미스 2 만
  // 개가 각자 WGSL 소스를 붙든 채 떠 있다가 디바이스를 떨어뜨렸다.
  //
  // 남은 수는 그대로다 — **모델 하나가 파이프라인 2 만 개**를 만든다:
  //
  //     resnet18          66      efficientnet_b4  19,531
  //     resnet152         72      efficientnet_b6  24,798
  //     vit_base         121
  //
  // 열쇠에 공간 크기와 채널이 구워지고 depthwise 는 블록마다 둘 다 바꾸기 때문이다.
  // 그것이 borch#121 의 본체이고, 고쳐지면 b7 도 들어온다.
  //
  // 안 도는 이름을 표에 두지 않는 것이 이 저장소의 규칙이라(README 참고) 코어가
  // 고쳐질 때까지 뺀다. `efficientnetB6`·`efficientnetB7` 자체는 남아 있다 —
  // 지우면 그 판을 다시 만들 때 배율을 또 찾아야 한다.


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
