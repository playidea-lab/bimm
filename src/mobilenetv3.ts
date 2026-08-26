/**
 * MobileNetV3 — timm 의 `mobilenetv3_large_100` 과 `mobilenetv3_small_100`.
 *
 * ## V2 와 갈리는 자리 넷
 *
 * 1. **머리가 뒤집혔다.** V2 는 넓힌 뒤 평균을 내고(`conv_head` → pool), V3 는 평균을
 *    먼저 낸 뒤 1×1 로 넓힌다(pool → `conv_head`). 1×1 이 1×1 격자 위에서 도니 계산이
 *    훨씬 싸고, 그것이 이 배치의 요점이다. `conv_head` 에 **bias 가 있다** — V2 의
 *    것에는 없다.
 * 2. **SE 블록.** 채널마다 하나의 수를 뽑아 다시 채널을 저울질한다.
 * 3. **활성화가 섞인다.** 앞쪽 stage 는 ReLU, 뒤쪽은 Hardswish 다 — 싼 자리에만
 *    비싼 활성화를 쓴다.
 * 4. **5×5 depthwise** 가 섞인다.
 *
 * ## 확장 채널을 배수로 안 적는다
 *
 * V2 는 `expansion` 이 6 으로 고르지만 V3 는 블록마다 다르고 배수로 떨어지지도 않는다
 * (80 → 200, 80 → 184). 논문이 표로 준 절대 수이므로 표에도 절대 수로 적는다.
 *
 * ## 구조는 모델에서 받아 적었다
 *
 * `timm.create_model(...)` 을 세워 놓고 블록마다 conv 설정·활성화·SE 유무를 뽑았다.
 * V2 때와 같은 이유다 — 논문과 구현은 갈리는 자리가 있고, 실려야 하는 것은 구현이다.
 */

import { nn, type Tensor } from "borch-ts";

/** stem 이 내는 채널. 두 판이 같다. */
const STEM_CHANNELS = 16;

/** 활성화는 이름으로 고른다 — 코어의 `unary` 가 받는 이름 그대로다. */
type Act = "relu" | "hardswish";

interface Block {
  /** depthwise 커널. 3 이나 5 다. */
  readonly kernel: number;
  /** 넓힌 뒤의 채널. **배수가 아니라 절대 수다.** */
  readonly mid: number;
  readonly cout: number;
  readonly stride: number;
  readonly se: boolean;
  readonly act: Act;
}

/**
 * SE 가 좁히는 채널 수 — torch 의 `make_divisible(mid * 0.25, 8)`.
 *
 * **0.25 를 곱하고 끝나지 않는다.** 8 의 배수로 맞추되, 반올림이 원래 값의 90% 아래로
 * 떨어지면 한 칸 올린다. 72 → 18 → 16 은 16.2 보다 작으므로 24 가 된다. 이 한 줄이
 * 없으면 그 블록만 채널이 어긋나고, 가중치는 모양이 안 맞아 실리지 않는다.
 */
function reduced(mid: number): number {
  const target = mid * 0.25;
  let out = Math.max(8, Math.floor(target + 4) - (Math.floor(target + 4) % 8));
  if (out < 0.9 * target) out += 8;
  return out;
}

/**
 * 채널마다 하나의 수를 뽑아 채널을 저울질한다.
 *
 * 평균이 `[B, C, 1, 1]` 로 나오고 곱셈이 그것을 `[B, C, H, W]` 에 펼친다 — torch 의
 * 브로드캐스트 규칙이 오른쪽부터 맞추므로 따로 늘릴 것이 없다.
 */
class SqueezeExcite extends nn.Module {
  private readonly conv_reduce: nn.Conv2d;
  private readonly conv_expand: nn.Conv2d;

  constructor(channels: number) {
    super();
    const rd = reduced(channels);
    // **bias 가 있다.** 이 두 conv 는 1×1 격자 위에서 도는 작은 완전연결이고,
    // timm 이 bias 를 켜 둔다.
    this.conv_reduce = new nn.Conv2d(channels, rd, 1, 1, 0, 1, 1, true);
    this.conv_expand = new nn.Conv2d(rd, channels, 1, 1, 0, 1, 1, true);
  }

  override forward(x: Tensor): Tensor {
    const pooled = x.adaptiveAvgPool(1);
    const narrowed = this.conv_reduce.forward(pooled).unary("relu");
    // **문은 hardsigmoid 다.** sigmoid 를 쓰면 값이 조금씩 다르고, 그 조금이 곱해져
    // 층을 타고 커진다.
    const gate = this.conv_expand.forward(narrowed).unary("hardsigmoid");
    return x.mul(gate);
  }
}

/**
 * 첫 stage 하나뿐인 블록 — 확장 없이 depthwise 로 받고 pointwise 로 좁힌다.
 *
 * 좁히는 쪽(`bn2`)에 활성화가 없는 것은 V2 와 같다.
 */
class DepthwiseSeparableConv extends nn.Module {
  private readonly conv_dw: nn.Conv2d;
  private readonly bn1: nn.BatchNormND;
  private readonly se: SqueezeExcite | null;
  private readonly conv_pw: nn.Conv2d;
  private readonly bn2: nn.BatchNormND;
  private readonly skip: boolean;
  private readonly act: Act;

  constructor(cin: number, spec: Block) {
    super();
    const pad = (spec.kernel - 1) / 2;
    this.conv_dw = new nn.Conv2d(cin, cin, spec.kernel, spec.stride, pad, 1, cin, false);
    this.bn1 = new nn.BatchNormND(cin);
    this.se = spec.se ? new SqueezeExcite(cin) : null;
    this.conv_pw = new nn.Conv2d(cin, spec.cout, 1, 1, 0, 1, 1, false);
    this.bn2 = new nn.BatchNormND(spec.cout);
    this.skip = spec.stride === 1 && cin === spec.cout;
    this.act = spec.act;
  }

  override forward(x: Tensor): Tensor {
    let h = this.bn1.forward(this.conv_dw.forward(x)).unary(this.act);
    if (this.se !== null) h = this.se.forward(h);
    const out = this.bn2.forward(this.conv_pw.forward(h));
    return this.skip ? out.add(x) : out;
  }
}

/**
 * 나머지 전부 — 넓히고, 걸르고, 저울질하고, 좁힌다.
 *
 * **SE 는 `bn2` 와 `conv_pwl` 사이다.** 넓은 자리에서 저울질해야 채널마다의 차이가
 * 남는다. 좁힌 뒤에 두면 이미 섞인 것을 저울질하게 된다.
 */
class InvertedResidual extends nn.Module {
  private readonly conv_pw: nn.Conv2d;
  private readonly bn1: nn.BatchNormND;
  private readonly conv_dw: nn.Conv2d;
  private readonly bn2: nn.BatchNormND;
  private readonly se: SqueezeExcite | null;
  private readonly conv_pwl: nn.Conv2d;
  private readonly bn3: nn.BatchNormND;
  private readonly skip: boolean;
  private readonly act: Act;

  constructor(cin: number, spec: Block) {
    super();
    const pad = (spec.kernel - 1) / 2;
    this.conv_pw = new nn.Conv2d(cin, spec.mid, 1, 1, 0, 1, 1, false);
    this.bn1 = new nn.BatchNormND(spec.mid);
    this.conv_dw = new nn.Conv2d(spec.mid, spec.mid, spec.kernel, spec.stride, pad, 1,
                                 spec.mid, false);
    this.bn2 = new nn.BatchNormND(spec.mid);
    this.se = spec.se ? new SqueezeExcite(spec.mid) : null;
    this.conv_pwl = new nn.Conv2d(spec.mid, spec.cout, 1, 1, 0, 1, 1, false);
    this.bn3 = new nn.BatchNormND(spec.cout);
    this.skip = spec.stride === 1 && cin === spec.cout;
    this.act = spec.act;
  }

  override forward(x: Tensor): Tensor {
    let h = this.bn1.forward(this.conv_pw.forward(x)).unary(this.act);
    h = this.bn2.forward(this.conv_dw.forward(h)).unary(this.act);
    if (this.se !== null) h = this.se.forward(h);
    const out = this.bn3.forward(this.conv_pwl.forward(h));
    return this.skip ? out.add(x) : out;
  }
}

/** 마지막 stage 하나뿐인 1×1 — 머리로 넘기기 전에 채널을 넓힌다. */
class ConvBnAct extends nn.Module {
  private readonly conv: nn.Conv2d;
  private readonly bn1: nn.BatchNormND;

  constructor(cin: number, cout: number) {
    super();
    this.conv = new nn.Conv2d(cin, cout, 1, 1, 0, 1, 1, false);
    this.bn1 = new nn.BatchNormND(cout);
  }

  override forward(x: Tensor): Tensor {
    return this.bn1.forward(this.conv.forward(x)).unary("hardswish");
  }
}

/** `blocks.0` 은 언제나 확장 없는 블록 하나, 마지막은 1×1 하나다. */
interface Recipe {
  /** stage 마다 블록 목록. 첫 stage 는 확장 없이 도는 것으로 읽는다. */
  readonly stages: readonly (readonly Block[])[];
  /** 마지막 1×1 이 넓히는 채널. */
  readonly wide: number;
  /** 머리에서 분류기 앞까지 넓히는 채널. */
  readonly head: number;
}

const LARGE: Recipe = {
  stages: [
    [{ kernel: 3, mid: 16, cout: 16, stride: 1, se: false, act: "relu" }],
    [{ kernel: 3, mid: 64, cout: 24, stride: 2, se: false, act: "relu" },
     { kernel: 3, mid: 72, cout: 24, stride: 1, se: false, act: "relu" }],
    [{ kernel: 5, mid: 72, cout: 40, stride: 2, se: true, act: "relu" },
     { kernel: 5, mid: 120, cout: 40, stride: 1, se: true, act: "relu" },
     { kernel: 5, mid: 120, cout: 40, stride: 1, se: true, act: "relu" }],
    [{ kernel: 3, mid: 240, cout: 80, stride: 2, se: false, act: "hardswish" },
     { kernel: 3, mid: 200, cout: 80, stride: 1, se: false, act: "hardswish" },
     { kernel: 3, mid: 184, cout: 80, stride: 1, se: false, act: "hardswish" },
     { kernel: 3, mid: 184, cout: 80, stride: 1, se: false, act: "hardswish" }],
    [{ kernel: 3, mid: 480, cout: 112, stride: 1, se: true, act: "hardswish" },
     { kernel: 3, mid: 672, cout: 112, stride: 1, se: true, act: "hardswish" }],
    [{ kernel: 5, mid: 672, cout: 160, stride: 2, se: true, act: "hardswish" },
     { kernel: 5, mid: 960, cout: 160, stride: 1, se: true, act: "hardswish" },
     { kernel: 5, mid: 960, cout: 160, stride: 1, se: true, act: "hardswish" }],
  ],
  wide: 960,
  head: 1280,
};

const SMALL: Recipe = {
  stages: [
    // **small 의 첫 블록은 stride 2 이고 SE 를 단다.** large 의 것과 이름만 같다.
    [{ kernel: 3, mid: 16, cout: 16, stride: 2, se: true, act: "relu" }],
    [{ kernel: 3, mid: 72, cout: 24, stride: 2, se: false, act: "relu" },
     { kernel: 3, mid: 88, cout: 24, stride: 1, se: false, act: "relu" }],
    [{ kernel: 5, mid: 96, cout: 40, stride: 2, se: true, act: "hardswish" },
     { kernel: 5, mid: 240, cout: 40, stride: 1, se: true, act: "hardswish" },
     { kernel: 5, mid: 240, cout: 40, stride: 1, se: true, act: "hardswish" }],
    [{ kernel: 5, mid: 120, cout: 48, stride: 1, se: true, act: "hardswish" },
     { kernel: 5, mid: 144, cout: 48, stride: 1, se: true, act: "hardswish" }],
    [{ kernel: 5, mid: 288, cout: 96, stride: 2, se: true, act: "hardswish" },
     { kernel: 5, mid: 576, cout: 96, stride: 1, se: true, act: "hardswish" },
     { kernel: 5, mid: 576, cout: 96, stride: 1, se: true, act: "hardswish" }],
  ],
  wide: 576,
  head: 1024,
};

/**
 * MobileNetV3.
 *
 * 필드 이름이 timm 의 것이다 — `stateDict` 열쇠가 거기서 나오고, 그래야 timm
 * 체크포인트가 **이름 그대로** 실린다. 그 까닭은 `mobilenet.ts` 첫 문단에 적혀 있다.
 *
 * timm 의 `global_pool`·`norm_head`·`act2`·`flatten` 은 여기 없다. 파라미터가 없어
 * 열쇠를 만들지 않으므로 층으로 두는 대신 `forward` 에서 부른다 — `norm_head` 는
 * 애초에 `Identity` 다.
 */
export class MobileNetV3 extends nn.Module {
  private readonly conv_stem: nn.Conv2d;
  private readonly bn1: nn.BatchNormND;
  private readonly blocks: nn.Sequential;
  private readonly conv_head: nn.Conv2d;
  private readonly classifier: nn.Linear;
  private readonly headChannels: number;

  constructor(recipe: Recipe, numClasses: number) {
    super();
    this.conv_stem = new nn.Conv2d(3, STEM_CHANNELS, 3, 2, 1, 1, 1, false);
    this.bn1 = new nn.BatchNormND(STEM_CHANNELS);

    const stages: nn.Module[] = [];
    let cin = STEM_CHANNELS;
    for (const [index, blocks] of recipe.stages.entries()) {
      const built: nn.Module[] = [];
      for (const spec of blocks) {
        // 첫 stage 만 확장 없이 돈다 — timm 이 그 자리에 두는 블록이 다르고,
        // 열쇠 이름도 그래서 다르다(`conv_pw` 가 없다).
        built.push(index === 0
          ? new DepthwiseSeparableConv(cin, spec)
          : new InvertedResidual(cin, spec));
        cin = spec.cout;
      }
      stages.push(new nn.Sequential(built));
    }
    // 마지막 stage 는 1×1 하나. timm 도 이것을 `blocks` 안에 둔다.
    stages.push(new nn.Sequential([new ConvBnAct(cin, recipe.wide)]));
    this.blocks = new nn.Sequential(stages);

    // **bias 가 있다.** 평균을 낸 뒤에 도는 1×1 이라 정규화가 뒤따르지 않는다
    // (`norm_head` 는 Identity), 그래서 치우침을 이 층이 직접 든다.
    this.conv_head = new nn.Conv2d(recipe.wide, recipe.head, 1, 1, 0, 1, 1, true);
    this.classifier = new nn.Linear(recipe.head, numClasses);
    this.headChannels = recipe.head;
  }

  override forward(x: Tensor): Tensor {
    let h = this.bn1.forward(this.conv_stem.forward(x)).unary("hardswish");
    h = this.blocks.forward(h);
    // **평균이 먼저다.** 넓힌 뒤에 평균을 내면 V2 가 되고, 수도 달라진다.
    h = h.adaptiveAvgPool(1);
    h = this.conv_head.forward(h).unary("hardswish");
    return this.classifier.forward(h.reshape([h.shape[0] ?? 1, this.headChannels]));
  }
}

/** timm 의 `mobilenetv3_large_100`. */
export function mobilenetv3Large(numClasses: number): MobileNetV3 {
  return new MobileNetV3(LARGE, numClasses);
}

/** timm 의 `mobilenetv3_small_100`. */
export function mobilenetv3Small(numClasses: number): MobileNetV3 {
  return new MobileNetV3(SMALL, numClasses);
}
