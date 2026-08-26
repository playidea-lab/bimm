/**
 * EfficientNet — timm 의 `efficientnet_b0`.
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
 * 6 뿐이라 V2 처럼 배수로 적는다. width·depth 배율을 받는 변종(b1~b7)은 이 표에
 * 배율을 씌워 나오는데, 그 규칙은 여기 없다 — **묻는 사람이 생기는 날 온다.**
 */

import { nn, type Tensor } from "borch-ts";

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
export class EfficientNetB0 extends nn.Module {
  private readonly conv_stem: nn.Conv2d;
  private readonly bn1: nn.BatchNormND;
  private readonly blocks: nn.Sequential;
  private readonly conv_head: nn.Conv2d;
  private readonly bn2: nn.BatchNormND;
  private readonly classifier: nn.Linear;

  constructor(numClasses: number) {
    super();
    this.conv_stem = new nn.Conv2d(3, STEM_CHANNELS, 3, 2, 1, 1, 1, false);
    this.bn1 = new nn.BatchNormND(STEM_CHANNELS);

    const stages: nn.Module[] = [];
    let cin = STEM_CHANNELS;
    for (const [index, stage] of B0.entries()) {
      const built: nn.Module[] = [];
      for (let i = 0; i < stage.repeats; i += 1) {
        const stride = i === 0 ? stage.stride : 1;
        built.push(index === 0
          ? new DepthwiseSeparableConv(cin, stage.cout, stage.kernel, stride)
          : new InvertedResidual(cin, stage.cout, stage.kernel, stride, stage.expansion));
        cin = stage.cout;
      }
      stages.push(new nn.Sequential(built));
    }
    this.blocks = new nn.Sequential(stages);

    this.conv_head = new nn.Conv2d(cin, HEAD_CHANNELS, 1, 1, 0, 1, 1, false);
    this.bn2 = new nn.BatchNormND(HEAD_CHANNELS);
    this.classifier = new nn.Linear(HEAD_CHANNELS, numClasses);
  }

  override forward(x: Tensor): Tensor {
    let h = this.bn1.forward(this.conv_stem.forward(x)).unary("silu");
    h = this.blocks.forward(h);
    // **넓힌 뒤 평균이다.** V3 만 뒤집혀 있고 여기는 V2 와 같다.
    h = this.bn2.forward(this.conv_head.forward(h)).unary("silu");
    h = h.adaptiveAvgPool(1);
    return this.classifier.forward(h.reshape([h.shape[0] ?? 1, HEAD_CHANNELS]));
  }
}
