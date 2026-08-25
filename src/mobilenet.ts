/**
 * MobileNetV2 — timm 의 `mobilenetv2_100` 을 옮긴 것.
 *
 * ## 필드 이름이 timm 의 것이고, 그것이 요점이다
 *
 * `conv_stem`·`conv_pwl`·`bn3` 은 이 저장소의 이름 짓기가 아니다. **`stateDict` 의
 * 열쇠가 필드 이름에서 나오므로**, timm 의 이름을 그대로 쓰면 timm 체크포인트가
 * **이름 그대로** 실린다 — 옮기는 표가 아예 없고, 따라서 그 표가 틀릴 자리도 없다.
 *
 * 코어가 여기까지 맞춰 두었기에 성립한다: `BatchNormND` 는 `running_mean`·
 * `num_batches_tracked` 까지 torch 와 같은 열쇠로 내보내고, `Sequential` 은 자식을
 * `0.weight` 처럼 자리 번호로 부른다. 둘 중 하나만 달랐어도 표가 필요했다.
 *
 * camelCase 를 쓰지 않는 유일한 자리다. 여기서 이름은 **취향이 아니라 계약**이다.
 *
 * ## 구조는 모델에서 받아 적었다
 *
 * 채널 수·stride·확장비·활성화 위치는 `timm.create_model("mobilenetv2_100")` 를
 * 세워 놓고 뽑았지 논문에서 옮기지 않았다. 논문과 구현이 갈리는 자리가 실제로 있고
 * (마지막 두 stage 의 stride 배치), 우리가 실어야 하는 것은 **구현 쪽**이다.
 *
 * ## 활성화가 BN 안에 없다
 *
 * timm 은 `BatchNormAct2d` 로 정규화와 활성화를 한 모듈에 묶는데, 그 묶음은
 * `stateDict` 열쇠를 하나도 만들지 않는다(활성화에 파라미터가 없다). 그래서 여기서는
 * 부르는 자리에서 활성화를 얹는다 — 열쇠는 같고 층은 하나 줄어든다.
 */

import { nn, type Tensor } from "borch-ts";

/** 스템이 내는 채널. timm 의 width 1.0 판이라 수를 그대로 쓴다. */
const STEM_CHANNELS = 32;
/** 분류기 앞 채널. `conv_head` 가 여기까지 넓힌다. */
const HEAD_CHANNELS = 1280;

/**
 * timm 이 블록을 세우는 표 — 확장비·출력 채널·반복·첫 블록의 stride.
 *
 * `width_mult` 가 1.0 이 아닌 변종(`mobilenetv2_050` 등)은 이 수에 배율을 곱하고
 * 8 의 배수로 맞춘다. 그 규칙은 여기 없다 — **팩토리가 하나뿐이므로 아직 아무도
 * 묻지 않았고**, 묻는 날 이 표 옆에 온다.
 */
const STAGES = [
  { expansion: 1, cout: 16, repeats: 1, stride: 1 },
  { expansion: 6, cout: 24, repeats: 2, stride: 2 },
  { expansion: 6, cout: 32, repeats: 3, stride: 2 },
  { expansion: 6, cout: 64, repeats: 4, stride: 2 },
  { expansion: 6, cout: 96, repeats: 3, stride: 1 },
  { expansion: 6, cout: 160, repeats: 3, stride: 2 },
  { expansion: 6, cout: 320, repeats: 1, stride: 1 },
] as const;

/**
 * 첫 stage 하나뿐인 블록 — 확장 없이 depthwise 로 받고 pointwise 로 좁힌다.
 *
 * **좁히는 쪽(`bn2`)에는 활성화가 없다.** timm 이 `has_pw_act=False` 로 두는 자리이고,
 * ReLU6 를 하나 더 얹으면 음수가 잘려 다른 모델이 된다.
 */
class DepthwiseSeparableConv extends nn.Module {
  private readonly conv_dw: nn.Conv2d;
  private readonly bn1: nn.BatchNormND;
  private readonly conv_pw: nn.Conv2d;
  private readonly bn2: nn.BatchNormND;
  private readonly skip: boolean;

  constructor(cin: number, cout: number, stride: number) {
    super();
    this.conv_dw = new nn.Conv2d(cin, cin, 3, stride, 1, 1, cin, false);
    this.bn1 = new nn.BatchNormND(cin);
    this.conv_pw = new nn.Conv2d(cin, cout, 1, 1, 0, 1, 1, false);
    this.bn2 = new nn.BatchNormND(cout);
    this.skip = stride === 1 && cin === cout;
  }

  override forward(x: Tensor): Tensor {
    const h = this.bn1.forward(this.conv_dw.forward(x)).unary("relu6");
    const out = this.bn2.forward(this.conv_pw.forward(h));
    return this.skip ? out.add(x) : out;
  }
}

/**
 * 나머지 전부 — 1×1 로 넓히고, depthwise 로 걸르고, 1×1 로 다시 좁힌다.
 *
 * 좁히는 쪽 이름이 `conv_pwl` 인 것은 timm 이 **linear** 를 뜻으로 붙였기 때문이다.
 * `bn3` 뒤에 활성화가 없다는 뜻이고, 그것이 이 블록이 "inverted" 인 이유다 —
 * 좁은 자리에서 비선형을 걸면 정보가 거기서 죽는다.
 */
class InvertedResidual extends nn.Module {
  private readonly conv_pw: nn.Conv2d;
  private readonly bn1: nn.BatchNormND;
  private readonly conv_dw: nn.Conv2d;
  private readonly bn2: nn.BatchNormND;
  private readonly conv_pwl: nn.Conv2d;
  private readonly bn3: nn.BatchNormND;
  private readonly skip: boolean;

  constructor(cin: number, cout: number, stride: number, expansion: number) {
    super();
    const mid = cin * expansion;
    this.conv_pw = new nn.Conv2d(cin, mid, 1, 1, 0, 1, 1, false);
    this.bn1 = new nn.BatchNormND(mid);
    this.conv_dw = new nn.Conv2d(mid, mid, 3, stride, 1, 1, mid, false);
    this.bn2 = new nn.BatchNormND(mid);
    this.conv_pwl = new nn.Conv2d(mid, cout, 1, 1, 0, 1, 1, false);
    this.bn3 = new nn.BatchNormND(cout);
    this.skip = stride === 1 && cin === cout;
  }

  override forward(x: Tensor): Tensor {
    let h = this.bn1.forward(this.conv_pw.forward(x)).unary("relu6");
    h = this.bn2.forward(this.conv_dw.forward(h)).unary("relu6");
    const out = this.bn3.forward(this.conv_pwl.forward(h));
    return this.skip ? out.add(x) : out;
  }
}

/**
 * MobileNetV2 (width 1.0, ImageNet).
 *
 * 필드 여섯이 그대로 자식이고 이름이 그대로 `stateDict` 의 앞자리다 — timm 의
 * `conv_stem`·`bn1`·`blocks`·`conv_head`·`bn2`·`classifier` 와 같다.
 *
 * timm 의 `global_pool` 은 여기 없다. 파라미터가 없어 열쇠를 만들지 않으므로,
 * 층으로 두는 대신 `forward` 에서 부른다.
 */
export class MobileNetV2 extends nn.Module {
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

    // stage 마다 Sequential 하나. 그래야 열쇠가 `blocks.2.1...` 로 timm 과 같아진다 —
    // 평평하게 펴면 `blocks.7...` 이 되고 그 체크포인트는 안 실린다.
    const stages: nn.Module[] = [];
    let cin = STEM_CHANNELS;
    for (const stage of STAGES) {
      const blocks: nn.Module[] = [];
      for (let i = 0; i < stage.repeats; i += 1) {
        // stride 는 stage 의 첫 블록만 진다. 나머지는 크기를 유지하며 쌓인다.
        const stride = i === 0 ? stage.stride : 1;
        blocks.push(
          stage.expansion === 1
            ? new DepthwiseSeparableConv(cin, stage.cout, stride)
            : new InvertedResidual(cin, stage.cout, stride, stage.expansion),
        );
        cin = stage.cout;
      }
      stages.push(new nn.Sequential(blocks));
    }
    this.blocks = new nn.Sequential(stages);

    this.conv_head = new nn.Conv2d(cin, HEAD_CHANNELS, 1, 1, 0, 1, 1, false);
    this.bn2 = new nn.BatchNormND(HEAD_CHANNELS);
    this.classifier = new nn.Linear(HEAD_CHANNELS, numClasses);
  }

  override forward(x: Tensor): Tensor {
    let h = this.bn1.forward(this.conv_stem.forward(x)).unary("relu6");
    h = this.blocks.forward(h);
    h = this.bn2.forward(this.conv_head.forward(h)).unary("relu6");
    h = h.adaptiveAvgPool(1);
    return this.classifier.forward(h.reshape([h.shape[0] ?? 1, HEAD_CHANNELS]));
  }
}
