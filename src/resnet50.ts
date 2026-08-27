/**
 * ResNet-50 — timm 의 ImageNet 판.
 *
 * ## 이 카탈로그의 첫 ImageNet ResNet
 *
 * `resnet.ts` 에 이미 ResNet 이 있지만 그것은 **CIFAR 판**이다. 스템부터 다르고
 * (3×3 stride 1 대 7×7 stride 2 + maxpool) 가중치가 안 호환된다. 그래서 이름도
 * 계보도 갈라 둔다 — `borchvision/resnet18_cifar` 와 `timm/resnet50` 이다.
 *
 * 이름공간을 둘로 받은 설계가 여기서 처음으로 값을 한다. 한 이름공간이었다면
 * `resnet18` 과 `resnet50` 이 나란히 서면서 **한쪽만 CIFAR 인 표**가 됐을 것이다.
 *
 * ## 블록이 다르다 — Bottleneck
 *
 * BasicBlock 은 3×3 을 둘 쌓는다. Bottleneck 은 셋을 쌓되 가운데만 3×3 이다:
 *
 *     1×1 로 좁히고 → 3×3 으로 보고 → 1×1 로 넓힌다
 *
 * 넓히는 배수는 4 다. `layer1` 은 64 로 좁혀 보고 256 으로 내놓는다. 그래서 같은
 * 깊이에 파라미터가 덜 들고, 그 덕에 50 층이 실용적인 크기(25.6M)에 머문다.
 *
 * ## `downsample` 은 각 layer 의 **첫 블록에만** 있다
 *
 * 잔차를 더하려면 모양이 맞아야 하는데, 첫 블록은 채널을 바꾸고(때로 stride 도)
 * 나머지는 안 바꾼다. 그래서 첫 블록만 1×1 컨볼루션으로 지름길을 맞춘다.
 *
 * **`layer1` 의 첫 블록도 downsample 을 든다.** stride 는 1 인데 채널이 64 에서
 * 256 으로 바뀌기 때문이다 — stride 만 보고 "여긴 필요 없다" 고 판단하면 그
 * 블록만 열쇠가 빠진다.
 *
 * ## 열쇠 이름이 EfficientNet 계열과 다르다
 *
 * 이쪽은 `layer1.0.conv1` 이고 `blocks.0.0...` 이 아니다. 분류기도 `classifier` 가
 * 아니라 `fc` 다. timm 안에서도 계열마다 다르므로, **옮겨오는 계열의 이름을 그대로**
 * 쓴다 — 우리 쪽에서 하나로 통일하면 그 순간 체크포인트가 안 실린다.
 */

import { nn, type Tensor } from "borch-ts";

/** 스템이 내는 채널. 다섯 판이 같다. */
const STEM_CHANNELS = 64;
/** 각 layer 가 좁혀서 보는 폭. 다섯 판이 같다. */
const WIDTHS = [64, 128, 256, 512] as const;
/** 첫 블록의 stride. `layer1` 만 크기를 안 줄인다. */
const STRIDES = [1, 2, 2, 2] as const;

/**
 * 한 판을 정하는 것 둘 — **블록 종류와 블록 수.**
 *
 * `basic` 은 3×3 을 둘 쌓고 넓히지 않는다(배수 1). `bottleneck` 은 1×1 로 좁혀
 * 3×3 으로 보고 1×1 로 **네 배** 넓힌다. 같은 깊이에 파라미터가 덜 들어서 50 층
 * 이상이 실용적인 크기에 머문다.
 *
 * 그래서 `resnet18` 과 `resnet34` 는 블록 수만 다르고, `resnet50` 과 `resnet18` 은
 * 블록 자체가 다르다. 이 표는 그 둘을 한 축에 놓는다.
 */
interface Variant {
  readonly block: "basic" | "bottleneck";
  readonly blocks: readonly [number, number, number, number];
}

/** timm 이 세우는 수다. 층에서 읽었다. */
const VARIANT: Readonly<Record<string, Variant>> = {
  resnet18: { block: "basic", blocks: [2, 2, 2, 2] },
  resnet34: { block: "basic", blocks: [3, 4, 6, 3] },
  resnet50: { block: "bottleneck", blocks: [3, 4, 6, 3] },
  resnet101: { block: "bottleneck", blocks: [3, 4, 23, 3] },
  resnet152: { block: "bottleneck", blocks: [3, 8, 36, 3] },
};

/** 지름길을 맞추는 1×1. 첫 블록에만 있다. */
export interface DownsamplePlan {
  readonly cin: number;
  readonly cout: number;
  readonly stride: number;
}

/** 한 블록이 쓰게 되는 수들. */
export interface BlockPlan {
  readonly cin: number;
  /** 1×1 이 좁히는 폭. 3×3 이 보는 것도 이 폭이다. */
  readonly width: number;
  /** 마지막 1×1 이 내놓는 폭 — `width * 4`. */
  readonly cout: number;
  /** 3×3 이 지는 stride. 나머지 둘은 언제나 1 이다. */
  readonly stride: number;
  /** 없으면 `null` — 그 블록은 입력을 그대로 더한다. */
  readonly downsample: DownsamplePlan | null;
}

/** 모델 하나가 쓰는 수 전부. */
export interface Plan {
  readonly stem: number;
  /** `layer1`..`layer4`. 묶음이 곧 열쇠 이름이다. */
  readonly layers: readonly (readonly BlockPlan[])[];
  readonly fcIn: number;
  /** 어느 블록을 쌓는가. 열쇠 이름이 여기서 갈린다 — basic 은 `conv3` 가 없다. */
  readonly block: "basic" | "bottleneck";
}

/**
 * 표에서 **층을 만들기 전에** 수를 전부 뽑는다.
 *
 * 다른 계열과 같은 이유다 — 층은 GPU 를 요구하고 수는 요구하지 않으므로, 이 산수는
 * `npm test` 에서 timm 과 대 볼 수 있다.
 */
export function resnetPlan(name: string): Plan {
  const v = VARIANT[name];
  if (v === undefined) {
    throw new Error(`알 수 없는 판: ${name} — ${Object.keys(VARIANT).join(", ")}`);
  }
  const expansion = v.block === "basic" ? 1 : 4;
  const layers: BlockPlan[][] = [];
  let cin = STEM_CHANNELS;
  for (const [s, width] of WIDTHS.entries()) {
    const blocks: BlockPlan[] = [];
    const cout = width * expansion;
    const count = v.blocks[s] ?? 0;
    for (let i = 0; i < count; i += 1) {
      const stride = i === 0 ? (STRIDES[s] ?? 1) : 1;
      // **첫 블록이어도 할 일이 없으면 안 둔다.**
      //
      // Bottleneck 의 `layer1` 은 stride 가 1 인데도 downsample 을 든다 — 채널이
      // 64 에서 256 으로 바뀌기 때문이다. BasicBlock 의 `layer1` 은 넓히지 않으므로
      // (배수 1) 채널도 그대로고 stride 도 1 이라 **아무것도 맞출 것이 없다.**
      //
      // 그래서 조건은 "첫 블록" 이 아니라 **"모양이 바뀌는가"** 다. 앞의 것으로 쓰면
      // resnet18 의 `layer1` 에 없는 열쇠 둘이 생긴다.
      const changes = stride !== 1 || cin !== cout;
      blocks.push({
        cin,
        width,
        cout,
        stride,
        downsample: changes ? { cin, cout, stride } : null,
      });
      cin = cout;
    }
    layers.push(blocks);
  }
  return { stem: STEM_CHANNELS, layers, fcIn: cin, block: v.block };
}

/** 이름을 남겨 둔 것은 이미 쓰는 곳이 있어서다. */
export function resnet50Plan(): Plan {
  return resnetPlan("resnet50");
}

/**
 * 1×1 → 3×3 → 1×1.
 *
 * 필드 이름이 timm 의 것이다 — `conv1`·`bn1`·`conv2`·`bn2`·`conv3`·`bn3` 과
 * `downsample`. `downsample` 이 `Sequential` 인 것도 timm 을 따른 것이라 열쇠가
 * `downsample.0.weight`·`downsample.1.weight` 로 나온다.
 */
export class Bottleneck extends nn.Module {
  private readonly conv1: nn.Conv2d;
  private readonly bn1: nn.BatchNormND;
  private readonly conv2: nn.Conv2d;
  private readonly bn2: nn.BatchNormND;
  private readonly conv3: nn.Conv2d;
  private readonly bn3: nn.BatchNormND;
  private readonly downsample: nn.Sequential | null;

  constructor(plan: BlockPlan) {
    super();
    this.conv1 = new nn.Conv2d(plan.cin, plan.width, 1, 1, 0, 1, 1, false);
    this.bn1 = new nn.BatchNormND(plan.width);
    // **stride 는 가운데 3×3 이 진다.** timm 이 그렇게 두고(torchvision 도 v1.5
    // 이후 같다), 첫 1×1 에 두면 같은 모양에 다른 수가 나온다.
    this.conv2 = new nn.Conv2d(plan.width, plan.width, 3, plan.stride, 1, 1, 1, false);
    this.bn2 = new nn.BatchNormND(plan.width);
    this.conv3 = new nn.Conv2d(plan.width, plan.cout, 1, 1, 0, 1, 1, false);
    this.bn3 = new nn.BatchNormND(plan.cout);
    this.downsample = plan.downsample === null ? null : new nn.Sequential([
      new nn.Conv2d(plan.downsample.cin, plan.downsample.cout, 1,
        plan.downsample.stride, 0, 1, 1, false),
      new nn.BatchNormND(plan.downsample.cout),
    ]);
  }

  override forward(x: Tensor): Tensor {
    let h = this.bn1.forward(this.conv1.forward(x)).unary("relu");
    h = this.bn2.forward(this.conv2.forward(h)).unary("relu");
    h = this.bn3.forward(this.conv3.forward(h));
    // **더하기가 먼저, relu 가 나중.** 뒤집으면 잔차가 아니라 그냥 깊은 망이 된다.
    const shortcut = this.downsample === null ? x : this.downsample.forward(x);
    return h.add(shortcut).unary("relu");
  }
}

/**
 * 3×3 을 둘. **넓히지 않는다**(배수 1).
 *
 * `conv3`·`bn3` 가 없다는 것이 Bottleneck 과 갈리는 자리이고, 그것이 곧 열쇠
 * 이름의 차이다. 두 블록을 한 클래스로 합치면 안 쓰는 층이 열쇠를 만들거나 조건이
 * 붙은 필드가 생기고, 둘 다 체크포인트를 깬다.
 */
export class BasicBlock extends nn.Module {
  private readonly conv1: nn.Conv2d;
  private readonly bn1: nn.BatchNormND;
  private readonly conv2: nn.Conv2d;
  private readonly bn2: nn.BatchNormND;
  private readonly downsample: nn.Sequential | null;

  constructor(plan: BlockPlan) {
    super();
    // **stride 는 첫 3×3 이 진다.** Bottleneck 은 가운데가 지는데, 이쪽은 셋이
    // 아니라 둘이라 "가운데" 가 없다.
    this.conv1 = new nn.Conv2d(plan.cin, plan.width, 3, plan.stride, 1, 1, 1, false);
    this.bn1 = new nn.BatchNormND(plan.width);
    this.conv2 = new nn.Conv2d(plan.width, plan.cout, 3, 1, 1, 1, 1, false);
    this.bn2 = new nn.BatchNormND(plan.cout);
    this.downsample = plan.downsample === null ? null : new nn.Sequential([
      new nn.Conv2d(plan.downsample.cin, plan.downsample.cout, 1,
        plan.downsample.stride, 0, 1, 1, false),
      new nn.BatchNormND(plan.downsample.cout),
    ]);
  }

  override forward(x: Tensor): Tensor {
    let h = this.bn1.forward(this.conv1.forward(x)).unary("relu");
    h = this.bn2.forward(this.conv2.forward(h));
    const shortcut = this.downsample === null ? x : this.downsample.forward(x);
    return h.add(shortcut).unary("relu");
  }
}

/**
 * ResNet-50.
 *
 * `global_pool` 과 `act1` 은 여기 층으로 두지 않는다 — 파라미터가 없어 열쇠를
 * 만들지 않으므로 `forward` 에서 부른다. 다른 계열에서와 같은 규칙이다.
 */
export class ResNet extends nn.Module {
  private readonly conv1: nn.Conv2d;
  private readonly bn1: nn.BatchNormND;
  private readonly layer1: nn.Sequential;
  private readonly layer2: nn.Sequential;
  private readonly layer3: nn.Sequential;
  private readonly layer4: nn.Sequential;
  private readonly fc: nn.Linear;
  private readonly fcIn: number;

  constructor(numClasses: number, name = "resnet50") {
    super();
    const plan = resnetPlan(name);
    // 7×7 stride 2 — CIFAR 판의 3×3 stride 1 과 갈리는 첫 자리다.
    this.conv1 = new nn.Conv2d(3, plan.stem, 7, 2, 3, 1, 1, false);
    this.bn1 = new nn.BatchNormND(plan.stem);

    // **넷을 각자 필드로 둔다.** `Sequential` 하나에 담으면 열쇠가 `layers.0...`
    // 이 되어 timm 의 `layer1...` 과 갈린다.
    const build = (i: number): nn.Sequential =>
      new nn.Sequential((plan.layers[i] ?? []).map((b) => (plan.block === "basic"
        ? new BasicBlock(b)
        : new Bottleneck(b))));
    this.layer1 = build(0);
    this.layer2 = build(1);
    this.layer3 = build(2);
    this.layer4 = build(3);

    // **`classifier` 가 아니라 `fc` 다.** timm 안에서도 계열마다 다르다.
    this.fc = new nn.Linear(plan.fcIn, numClasses, true);
    this.fcIn = plan.fcIn;
  }

  override forward(x: Tensor): Tensor {
    let h = this.bn1.forward(this.conv1.forward(x)).unary("relu");
    // **padding 1 이 있어야 한다.** `maxPool2d` 는 padding 을 안 받아서 크기가
    // 하나 어긋나고, 그 어긋남은 마지막 평균까지 살아남는다. `poolND` 가 받는다.
    h = h.poolND("max", 3, 2, 1);
    h = this.layer1.forward(h);
    h = this.layer2.forward(h);
    h = this.layer3.forward(h);
    h = this.layer4.forward(h);
    h = h.adaptiveAvgPool(1);
    return this.fc.forward(h.reshape([h.shape[0] ?? 1, this.fcIn]));
  }
}

/** timm 의 `resnet18`. BasicBlock 을 [2,2,2,2] 로 쌓는다. */
export function resnet18(numClasses: number): ResNet {
  return new ResNet(numClasses, "resnet18");
}

/** timm 의 `resnet34`. resnet18 과 블록 수만 다르다. */
export function resnet34(numClasses: number): ResNet {
  return new ResNet(numClasses, "resnet34");
}

/** timm 의 `resnet50`. */
export function resnet50(numClasses: number): ResNet {
  return new ResNet(numClasses, "resnet50");
}

/** timm 의 `resnet101`. resnet50 과 `layer3` 의 깊이만 다르다(6 → 23). */
export function resnet101(numClasses: number): ResNet {
  return new ResNet(numClasses, "resnet101");
}

/** timm 의 `resnet152`. */
export function resnet152(numClasses: number): ResNet {
  return new ResNet(numClasses, "resnet152");
}

/** 판 이름들. 검사가 다섯을 같은 자리에서 묻는다. */
export const RESNETS = Object.keys(VARIANT);
