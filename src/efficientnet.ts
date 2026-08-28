/**
 * EfficientNet — timm 의 `efficientnet_b0` ~ `b3`.
 *
 * 뼈대는 MobileNet 과 같다 — 넓히고, depthwise 로 걸르고, 저울질하고, 좁힌다.
 * 갈리는 자리는 셋이고 **셋 다 조용히 틀릴 수 있는 종류**다.
 *
 * ## 1. SE 가 **블록 입력**을 기준으로 좁힌다
 *
 * MobileNetV3 는 넓힌 뒤의 채널(`mid`)에 0.25 를 곱한다. 여기는 **블록에 들어온
 * 채널**에 곱한다 — 같은 0.25 인데 기준이 다르다. `se=96→4` 가 그 증거다: 96 의
 * 0.25 는 24 지만 실린 것은 4 이고, 그것은 입력 16 의 0.25 다.
 *
 * 기준을 잘못 잡으면 채널이 여섯 배 어긋나고 가중치가 아예 안 실린다. 실리지 않는
 * 것은 그나마 다행인 종류다.
 *
 * ## 2. 문이 sigmoid 다
 *
 * V3 는 hardsigmoid 를 쓴다. 둘은 모양이 비슷해서 바꿔 써도 모델은 서고 수만 조금씩
 * 다른데, 그 조금이 층을 타고 커진다.
 *
 * ## 3. 머리가 V2 의 순서다
 *
 * 넓힌 뒤 평균이다(`conv_head` → `bn2` → pool). V3 만 뒤집혀 있고 여기는 안 뒤집혔다.
 *
 * ## 확장비는 규칙적이다
 *
 * V3 는 배수로 안 떨어져 절대 채널을 적어야 했지만(80 → 200, 80 → 184), 여기는 1 과
 * 6 뿐이라 V2 처럼 배수로 적는다.
 *
 * ## b1~b3 는 같은 표에 배율을 씌운 것이다
 *
 * 채널에 width, 반복에 depth 를 곱한다. 그래서 표는 하나뿐이고 판마다 두 수만 다르다
 * — **계열이란 것이 이런 모양이다.**
 *
 * 배율을 곱한 뒤의 채널을 8 의 배수로 맞추는 규칙은 MobileNetV3 의 것과 같은 식인데
 * (`round_channels`), 여기서는 곱셈이 먼저다. b1~b3 의 stem·stage·head 채널과 반복
 * 수를 timm 에서 뽑아 이 규칙과 대조했고 **전부 일치했다.**
 *
 * **입력 해상도는 판마다 다르고, 논문의 수와도 다르다** — timm 의 b2 는 260 이 아니라
 * 256, b3 는 300 이 아니라 288 이다. 그 수는 이 파일이 아니라 매니페스트의 전처리에
 * 적히고, 만드는 쪽이 `default_cfg` 에서 받아 적는다.
 */

import { nn, type Tensor } from "borch-ts";

import { roundChannels } from "./channels.js";

/** 스템이 내는 채널. */
const STEM_CHANNELS = 32;
/** 분류기 앞 채널. */
const HEAD_CHANNELS = 1280;

interface Stage {
  readonly kernel: number;
  readonly expansion: number;
  readonly cout: number;
  readonly repeats: number;
  readonly stride: number;
}

/** timm 의 `efficientnet_b0`. 배율이 걸리기 전의 표다. */
const B0: readonly Stage[] = [
  { kernel: 3, expansion: 1, cout: 16, repeats: 1, stride: 1 },
  { kernel: 3, expansion: 6, cout: 24, repeats: 2, stride: 2 },
  { kernel: 5, expansion: 6, cout: 40, repeats: 2, stride: 2 },
  { kernel: 3, expansion: 6, cout: 80, repeats: 3, stride: 2 },
  { kernel: 5, expansion: 6, cout: 112, repeats: 3, stride: 1 },
  { kernel: 5, expansion: 6, cout: 192, repeats: 4, stride: 2 },
  { kernel: 3, expansion: 6, cout: 320, repeats: 1, stride: 1 },
];

/** 한 블록이 쓰게 되는 수들. 텐서를 만들기 전에 이미 정해져 있다. */
export interface BlockPlan {
  /** 첫 단계만 확장 없이 depthwise 로 받는다. */
  readonly kind: "dw" | "ir";
  readonly cin: number;
  readonly cout: number;
  readonly kernel: number;
  readonly stride: number;
  /** 넓힌 채널. `dw` 는 넓히지 않으므로 `cin` 과 같다. */
  readonly mid: number;
  /** SE 가 좁히는 폭. **블록 입력에서 나온다** — 넓힌 채널이 아니다. */
  readonly se: number;
}

/**
 * 모델 하나가 쓰는 수 전부.
 *
 * **단계별로 묶여 있다.** 평평하게 두면 안 되는 이유는 열쇠 이름이다 — timm 은
 * `blocks.2.1.conv_pw` 처럼 단계와 그 안의 자리로 부르고, 우리도 `Sequential` 을
 * 단계마다 두어 같은 이름이 나온다. 묶음이 풀리면 이름이 갈리고, 이름이 갈리면
 * 가중치가 안 실린다.
 */
export interface Plan {
  readonly stem: number;
  readonly stages: readonly (readonly BlockPlan[])[];
  readonly head: number;
}

/**
 * 배율 둘에서 **층을 만들기 전에** 수를 전부 뽑는다.
 *
 * ## 왜 갈라 두는가
 *
 * 층을 하나라도 만들려면 WebGPU 어댑터가 필요하고, 그건 브라우저에서만 잡힌다.
 * 그래서 이 계열의 산수 — 배율을 먹인 채널, 올림한 반복수, SE 가 좁히는 폭 — 는
 * **GPU 가 있는 자리에서만 확인할 수 있었다.** 실제로는 브라우저를 띄우는 parity
 * 하네스가 유일한 검사였고, 그것은 CI 에서 안 돈다.
 *
 * 수를 먼저 뽑아 두면 그 산수는 어디서나 검사된다. `channels.ts` 와 같은 이유이고,
 * 여기는 그것을 모델 한 채 전체로 넓힌 것이다.
 */
export function efficientnetPlan(width: number, depth: number): Plan {
  const stages: BlockPlan[][] = [];
  let cin = roundChannels(STEM_CHANNELS, width);
  const stem = cin;
  for (const [index, stage] of B0.entries()) {
    const blocks: BlockPlan[] = [];
    const cout = roundChannels(stage.cout, width);
    // 반복은 올림이다 — timm 이 `ceil(n * depth)` 로 센다.
    const repeats = Math.ceil(stage.repeats * depth);
    for (let i = 0; i < repeats; i += 1) {
      const kind = index === 0 ? "dw" : "ir";
      blocks.push({
        kind,
        cin,
        cout,
        kernel: stage.kernel,
        stride: i === 0 ? stage.stride : 1,
        mid: kind === "dw" ? cin : cin * stage.expansion,
        se: seChannels(cin),
      });
      cin = cout;
    }
    stages.push(blocks);
  }
  return { stem, stages, head: roundChannels(HEAD_CHANNELS, width) };
}

/**
 * SE 가 좁히는 폭 — **블록 입력의 0.25.**
 *
 * MobileNetV3 와 갈리는 자리다. 저쪽은 넓힌 채널에서 시작해 `make_divisible` 을
 * 통과하는데, 여기는 **블록 입력**에서 시작해 그냥 반올림한다. b0 부터 b3 까지
 * 블록 88 개를 timm 과 대 봐서 확인했다 — 한 곳도 안 갈린다.
 */
function seChannels(fromInput: number): number {
  return Math.round(fromInput * 0.25);
}

/**
 * 채널마다 하나의 수를 뽑아 채널을 저울질한다.
 *
 * **좁히는 폭이 블록 입력에서 나온다** — 넓힌 채널이 아니다. 위 문단을 보라.
 * B0 에서는 그 0.25 가 언제나 정수로 떨어져 반올림 규칙이 필요 없다.
 */
class SqueezeExcite extends nn.Module {
  private readonly conv_reduce: nn.Conv2d;
  private readonly conv_expand: nn.Conv2d;

  constructor(channels: number, fromInput: number) {
    super();
    const rd = Math.round(fromInput * 0.25);
    this.conv_reduce = new nn.Conv2d(channels, rd, 1, 1, 0, 1, 1, true);
    this.conv_expand = new nn.Conv2d(rd, channels, 1, 1, 0, 1, 1, true);
  }

  override forward(x: Tensor): Tensor {
    const pooled = x.adaptiveAvgPool(1);
    const narrowed = this.conv_reduce.forward(pooled).unary("silu");
    return x.mul(this.conv_expand.forward(narrowed).unary("sigmoid"));
  }
}

/** 첫 stage 하나뿐인 블록 — 확장 없이 depthwise 로 받고 pointwise 로 좁힌다. */
class DepthwiseSeparableConv extends nn.Module {
  private readonly conv_dw: nn.Conv2d;
  private readonly bn1: nn.BatchNormND;
  private readonly se: SqueezeExcite;
  private readonly conv_pw: nn.Conv2d;
  private readonly bn2: nn.BatchNormND;
  private readonly skip: boolean;

  constructor(cin: number, cout: number, kernel: number, stride: number) {
    super();
    const pad = (kernel - 1) / 2;
    this.conv_dw = new nn.Conv2d(cin, cin, kernel, stride, pad, 1, cin, false);
    this.bn1 = new nn.BatchNormND(cin);
    this.se = new SqueezeExcite(cin, cin);
    this.conv_pw = new nn.Conv2d(cin, cout, 1, 1, 0, 1, 1, false);
    this.bn2 = new nn.BatchNormND(cout);
    this.skip = stride === 1 && cin === cout;
  }

  override forward(x: Tensor): Tensor {
    let h = this.bn1.forward(this.conv_dw.forward(x)).unary("silu");
    h = this.se.forward(h);
    const out = this.bn2.forward(this.conv_pw.forward(h));
    return this.skip ? out.add(x) : out;
  }
}

/** 나머지 전부 — 넓히고, 걸르고, 저울질하고, 좁힌다. */
class InvertedResidual extends nn.Module {
  private readonly conv_pw: nn.Conv2d;
  private readonly bn1: nn.BatchNormND;
  private readonly conv_dw: nn.Conv2d;
  private readonly bn2: nn.BatchNormND;
  private readonly se: SqueezeExcite;
  private readonly conv_pwl: nn.Conv2d;
  private readonly bn3: nn.BatchNormND;
  private readonly skip: boolean;

  constructor(cin: number, cout: number, kernel: number, stride: number, expansion: number) {
    super();
    const mid = cin * expansion;
    const pad = (kernel - 1) / 2;
    this.conv_pw = new nn.Conv2d(cin, mid, 1, 1, 0, 1, 1, false);
    this.bn1 = new nn.BatchNormND(mid);
    this.conv_dw = new nn.Conv2d(mid, mid, kernel, stride, pad, 1, mid, false);
    this.bn2 = new nn.BatchNormND(mid);
    // **좁히는 폭은 `cin` 에서 나온다.** `mid` 를 넘기면 여섯 배 어긋난다.
    this.se = new SqueezeExcite(mid, cin);
    this.conv_pwl = new nn.Conv2d(mid, cout, 1, 1, 0, 1, 1, false);
    this.bn3 = new nn.BatchNormND(cout);
    this.skip = stride === 1 && cin === cout;
  }

  override forward(x: Tensor): Tensor {
    let h = this.bn1.forward(this.conv_pw.forward(x)).unary("silu");
    h = this.bn2.forward(this.conv_dw.forward(h)).unary("silu");
    h = this.se.forward(h);
    const out = this.bn3.forward(this.conv_pwl.forward(h));
    return this.skip ? out.add(x) : out;
  }
}

/**
 * EfficientNet-B0.
 *
 * 필드 이름이 timm 의 것이다 — 그 까닭은 `mobilenet.ts` 첫 문단에 있다.
 */
export class EfficientNet extends nn.Module {
  private readonly conv_stem: nn.Conv2d;
  private readonly bn1: nn.BatchNormND;
  private readonly blocks: nn.Sequential;
  private readonly conv_head: nn.Conv2d;
  private readonly bn2: nn.BatchNormND;
  private readonly classifier: nn.Linear;
  private readonly headChannels: number;

  constructor(numClasses: number, width = 1, depth = 1) {
    super();
    // **수는 이미 정해져 있다.** 여기서는 그것을 층으로 세우기만 한다 — 산수는
    // `efficientnetPlan` 이 하고, 그쪽은 GPU 없이 검사된다.
    const plan = efficientnetPlan(width, depth);
    this.conv_stem = new nn.Conv2d(3, plan.stem, 3, 2, 1, 1, 1, false);
    this.bn1 = new nn.BatchNormND(plan.stem);

    this.blocks = new nn.Sequential(plan.stages.map((blocks) => new nn.Sequential(
      blocks.map((b) => (b.kind === "dw"
        ? new DepthwiseSeparableConv(b.cin, b.cout, b.kernel, b.stride)
        : new InvertedResidual(b.cin, b.cout, b.kernel, b.stride, b.mid / b.cin))),
    )));

    const last = plan.stages[plan.stages.length - 1];
    const cin = last?.[last.length - 1]?.cout ?? plan.stem;
    this.conv_head = new nn.Conv2d(cin, plan.head, 1, 1, 0, 1, 1, false);
    this.bn2 = new nn.BatchNormND(plan.head);
    this.classifier = new nn.Linear(plan.head, numClasses);
    this.headChannels = plan.head;
  }

  override forward(x: Tensor): Tensor {
    let h = this.bn1.forward(this.conv_stem.forward(x)).unary("silu");
    h = this.blocks.forward(h);
    // **넓힌 뒤 평균이다.** V3 만 뒤집혀 있고 여기는 V2 와 같다.
    h = this.bn2.forward(this.conv_head.forward(h)).unary("silu");
    h = h.adaptiveAvgPool(1);
    return this.classifier.forward(h.reshape([h.shape[0] ?? 1, this.headChannels]));
  }
}

/** timm 의 `efficientnet_b0` — 배율이 없는 판. */
export function efficientnetB0(numClasses: number): EfficientNet {
  return new EfficientNet(numClasses, 1.0, 1.0);
}

/** timm 의 `efficientnet_b1`. */
export function efficientnetB1(numClasses: number): EfficientNet {
  return new EfficientNet(numClasses, 1.0, 1.1);
}

/** timm 의 `efficientnet_b2`. */
export function efficientnetB2(numClasses: number): EfficientNet {
  return new EfficientNet(numClasses, 1.1, 1.2);
}

/** timm 의 `efficientnet_b3`. */
export function efficientnetB3(numClasses: number): EfficientNet {
  return new EfficientNet(numClasses, 1.2, 1.4);
}

/**
 * b4 부터 b7 까지. **배율 둘 말고는 b0 과 같은 표다.**
 *
 * 수는 timm 의 소스에서 읽었다 — `_gen_efficientnet` 을 부르는 자리의
 * `channel_multiplier` 와 `depth_multiplier` 다. 논문 표를 옮기지 않은 것은, 이
 * 저장소에서 표를 보고 옮긴 산수가 원본과 갈린 적이 있어서다.
 *
 * **입력 크기는 판마다 다르다**(b4 는 320, b7 은 600). 그것은 구조가 아니라
 * 전처리이므로 여기 없고 매니페스트에 적힌다 — 모델은 어떤 크기든 받는다.
 */
export function efficientnetB4(numClasses: number): EfficientNet {
  return new EfficientNet(numClasses, 1.4, 1.8);
}

/** timm 의 `efficientnet_b5`. */
export function efficientnetB5(numClasses: number): EfficientNet {
  return new EfficientNet(numClasses, 1.6, 2.2);
}

/** timm 의 `efficientnet_b6`. */
export function efficientnetB6(numClasses: number): EfficientNet {
  return new EfficientNet(numClasses, 1.8, 2.6);
}

/** timm 의 `efficientnet_b7`. 블록 55 개로 이 계열에서 가장 깊다. */
export function efficientnetB7(numClasses: number): EfficientNet {
  return new EfficientNet(numClasses, 2.0, 3.1);
}

/** 판마다의 배율. 검사가 여덟을 같은 자리에서 묻는다. */
export const SCALES: Readonly<Record<string, readonly [number, number]>> = {
  efficientnet_b0: [1.0, 1.0], efficientnet_b1: [1.0, 1.1],
  efficientnet_b2: [1.1, 1.2], efficientnet_b3: [1.2, 1.4],
  efficientnet_b4: [1.4, 1.8], efficientnet_b5: [1.6, 2.2],
  efficientnet_b6: [1.8, 2.6], efficientnet_b7: [2.0, 3.1],
};
