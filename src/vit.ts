/**
 * Vision Transformer — timm 의 `vit_tiny_patch16_224`.
 *
 * 지금까지 옮긴 것과 **뼈대가 다르다.** 합성곱이 격자를 훑는 대신, 그림을 16×16
 * 조각으로 잘라 토큰으로 세우고 토큰끼리 서로를 본다. 갈리는 자리가 넷이고 넷 다
 * 조용히 틀릴 종류다.
 *
 * ## 1. 층이 아닌 파라미터가 둘 있다
 *
 * `cls_token` 과 `pos_embed` 는 어느 층에도 속하지 않는 텐서다. 코어는 필드를 훑어
 * 층을 찾으므로 이 둘은 `ownParameters()` 로 직접 등록해야 하고, **그 이름이 그대로
 * `stateDict` 열쇠가 된다** — timm 의 이름을 쓰지 않으면 체크포인트가 안 실린다.
 *
 * ## 2. cls 토큰은 `Tensor.cat` 으로 붙인다
 *
 * 처음엔 코어에 `cat` 이 없는 줄 알고 양쪽을 0 으로 늘려 더했다. 있었다 —
 * **인스턴스 메서드가 아니라 static 이라** 메서드만 훑은 검색에 안 걸렸을 뿐이다.
 * torch 도 `torch.cat` 이 자유 함수이므로 부르는 모양까지 같다(borch#87).
 *
 * ## 3. q·k·v 가 한 층에 들어 있다
 *
 * timm 은 `Linear 192→576` 하나로 셋을 함께 낸다. 따로 만들면 열쇠가 안 맞으므로
 * 한 번에 계산하고 `narrow` 로 갈라 쓴다.
 *
 * ## 4. 머리가 평균이 아니라 토큰 하나다
 *
 * MobileNet 계열은 공간 평균을 냈지만 여기는 **0 번 토큰(cls)만** 뽑는다. 평균을
 * 내면 값이 그럴듯하게 나오면서 다른 모델이 된다.
 *
 * ## 전처리도 다르다
 *
 * mean·std 가 0.5 이고 crop_pct 가 0.9 다 — 앞의 일곱과 전부 다르다. 그 수는 이
 * 파일이 아니라 매니페스트에 적히고, 만드는 쪽이 `default_cfg` 에서 받아 적는다.
 */

import { nn, Tensor } from "borch-ts";

/** 패치 한 변. `/16` 판들이 전부 이 수다. */
const PATCH = 16;
/** MLP 가 넓히는 배수. timm 의 `mlp_ratio` 는 이 계열에서 4 로 고정이다. */
const MLP_RATIO = 4;
/** LayerNorm 의 eps. torch 기본값(1e-5)이 아니라 **1e-6** 이다. */
const NORM_EPS = 1e-6;

/**
 * 한 판을 정하는 수들.
 *
 * **tiny·small·base 가 갈리는 것은 `dim` 과 `heads` 둘뿐이다.** 깊이도 MLP 배수도
 * eps 도 셋이 같다 — timm 에 물어 확인했다. 그래서 판을 늘리는 일이 계열을 늘리는
 * 일과 값이 다르다: 이쪽은 표에 줄 하나이고, 저쪽은 코어에 없는 연산을 만난다.
 */
interface Variant {
  readonly dim: number;
  readonly heads: number;
  readonly depth: number;
}

/** timm 이 실제로 세우는 수다. 표를 보고 옮긴 것이 아니라 층에서 읽었다. */
const TINY: Variant = { dim: 192, heads: 3, depth: 12 };
const SMALL: Variant = { dim: 384, heads: 6, depth: 12 };
const BASE: Variant = { dim: 768, heads: 12, depth: 12 };

/**
 * 이 판이 쓰는 수 전부 — 상수 여섯에서 나오는 것까지.
 *
 * ViT 는 앞의 합성곱 계열과 성격이 다르다. 표가 없고 상수 몇 개뿐이지만, **그
 * 상수에서 파생되는 수들**이 어긋날 자리다 — 패치 수, head 하나가 보는 차원,
 * qkv 가 내는 폭, MLP 가 넓히는 폭, `pos_embed` 의 길이.
 *
 * 이 다섯은 전부 곱셈이나 나눗셈 하나이고, 그래서 **틀려도 그럴듯하다.** 예를 들어
 * `pos_embed` 길이를 196 으로 두면(cls 토큰을 빼먹으면) 모양이 하나 어긋난 채
 * 나머지는 다 맞아 보인다.
 */
export interface VitPlan {
  readonly dim: number;
  readonly depth: number;
  readonly heads: number;
  /** head 하나가 보는 차원. `dim / heads` 다. */
  readonly headDim: number;
  readonly patch: number;
  /** 한 장이 내는 패치 수. `(224 / patch)^2` 다. */
  readonly patches: number;
  /** `pos_embed` 의 길이 — **패치 수 + cls 토큰 하나.** */
  readonly posLen: number;
  /** qkv 한 층이 내는 폭. 셋을 함께 내므로 `dim * 3` 이다. */
  readonly qkvOut: number;
  /** MLP 가 넓히는 폭. */
  readonly mlpHidden: number;
  readonly normEps: number;
}

/** 위 수들을 상수에서 뽑는다. 층을 만들지 않으므로 GPU 없이 검사된다. */
export function vitPlan(v: Variant, imageSize = 224): VitPlan {
  const patches = (imageSize / PATCH) ** 2;
  return {
    dim: v.dim,
    depth: v.depth,
    heads: v.heads,
    headDim: v.dim / v.heads,
    patch: PATCH,
    patches,
    posLen: patches + 1,
    qkvOut: v.dim * 3,
    mlpHidden: v.dim * MLP_RATIO,
    normEps: NORM_EPS,
  };
}

/** tiny 판. 이름을 남겨 둔 것은 이미 쓰는 곳이 있어서다. */
export function vitTinyPlan(imageSize = 224): VitPlan {
  return vitPlan(TINY, imageSize);
}

/**
 * 토큰끼리 서로를 본다.
 *
 * `qkv` 가 셋을 함께 내므로 `narrow` 로 갈라 쓴다 — 열쇠를 timm 과 맞추려면 층을
 * 쪼갤 수 없다.
 */
class Attention extends nn.Module {
  private readonly qkv: nn.Linear;
  private readonly proj: nn.Linear;
  private readonly headDim: number;
  private readonly heads: number;
  private readonly scale: number;

  constructor(dim: number, heads: number) {
    super();
    this.qkv = new nn.Linear(dim, dim * 3, true);
    this.proj = new nn.Linear(dim, dim, true);
    this.headDim = dim / heads;
    // **머리 수를 들고 있어야 한다.** 전에는 모듈 상수를 봤는데, 그것은 판이
    // 하나일 때만 맞는다 — small 은 6 이고 base 는 12 다.
    this.heads = heads;
    // **곱하기 전에 나눈다.** torch 도 `q * scale` 을 먼저 하고, 순서를 바꾸면 큰
    // 내적에서 float32 가 먼저 넘친다.
    this.scale = 1 / Math.sqrt(this.headDim);
  }

  override forward(x: Tensor): Tensor {
    const [batch = 1, tokens = 1] = x.shape;
    const dim = this.headDim * this.heads;

    const fused = this.qkv.forward(x);                       // [B, N, 3D]
    // [B, N, 3, H, d] → [3, B, H, N, d] 로 세우면 셋을 같은 모양으로 꺼낼 수 있다.
    const parts = fused.reshape([batch, tokens, 3, this.heads, this.headDim])
      .permute([2, 0, 3, 1, 4]);
    const q = parts.select(0, 0);
    const k = parts.select(0, 1);
    const v = parts.select(0, 2);

    // 헤드를 배치로 접어 `bmm` 에 넣는다 — 코어의 행렬곱은 3 차원까지 본다.
    const folded = [batch * this.heads, tokens, this.headDim];
    // 스칼라 곱도 텐서를 거친다 — 코어의 `mul` 은 텐서만 받고, 0 차원은
    // 브로드캐스트로 펼쳐진다.
    const qf = q.reshape(folded).mul(Tensor.owned([], this.scale));
    const kf = k.reshape(folded);
    const vf = v.reshape(folded);

    const scores = qf.bmm(kf.transpose(-2, -1));             // [B*H, N, N]
    const weights = scores.softmax(-1);
    const mixed = weights.bmm(vf);                           // [B*H, N, d]

    // 접었던 헤드를 다시 펴서 [B, N, D] 로.
    const merged = mixed.reshape([batch, this.heads, tokens, this.headDim])
      .permute([0, 2, 1, 3])
      .reshape([batch, tokens, dim]);
    return this.proj.forward(merged);
  }
}

/** 토큰마다 따로 도는 두 층. */
class Mlp extends nn.Module {
  private readonly fc1: nn.Linear;
  private readonly fc2: nn.Linear;

  constructor(dim: number, hidden: number) {
    super();
    this.fc1 = new nn.Linear(dim, hidden, true);
    this.fc2 = new nn.Linear(hidden, dim, true);
  }

  override forward(x: Tensor): Tensor {
    const hidden = this.fc1.forward(x).unary("gelu");
    return this.fc2.forward(hidden);
  }
}

/**
 * 블록 하나 — 정규화가 **앞에** 있고 잔차가 둘이다.
 *
 * `x + attn(norm(x))` 이지 `norm(x + attn(x))` 가 아니다(pre-norm). 순서를 바꾸면
 * 깊은 판에서 학습이 갈리고, 추론에서도 다른 수가 나온다.
 */
class Block extends nn.Module {
  private readonly norm1: nn.LayerNorm;
  private readonly attn: Attention;
  private readonly norm2: nn.LayerNorm;
  private readonly mlp: Mlp;

  constructor(dim: number, heads: number) {
    super();
    this.norm1 = new nn.LayerNorm([dim], NORM_EPS);
    this.attn = new Attention(dim, heads);
    this.norm2 = new nn.LayerNorm([dim], NORM_EPS);
    this.mlp = new Mlp(dim, dim * MLP_RATIO);
  }

  override forward(x: Tensor): Tensor {
    const attended = x.add(this.attn.forward(this.norm1.forward(x)));
    return attended.add(this.mlp.forward(this.norm2.forward(attended)));
  }
}

/** 그림을 16×16 조각으로 자른다 — stride 가 커널과 같은 합성곱이 곧 자르기다. */
class PatchEmbed extends nn.Module {
  private readonly proj: nn.Conv2d;

  constructor(dim: number, patch: number) {
    super();
    this.proj = new nn.Conv2d(3, dim, patch, patch, 0, 1, 1, true);
  }

  override forward(x: Tensor): Tensor {
    const grid = this.proj.forward(x);                        // [B, D, H/p, W/p]
    const [batch = 1, dim = 1, gh = 1, gw = 1] = grid.shape;
    // [B, D, H', W'] → [B, D, N] → [B, N, D]
    return grid.reshape([batch, dim, gh * gw]).permute([0, 2, 1]);
  }
}

/**
 * ViT-Tiny/16.
 *
 * `cls_token`·`pos_embed` 가 층이 아니므로 `ownParameters()` 로 등록한다 — 이름이
 * 그대로 열쇠이고, timm 의 것과 같아야 체크포인트가 실린다.
 */
export class VisionTransformer extends nn.Module {
  private readonly cls_token: Tensor;
  private readonly pos_embed: Tensor;
  private readonly patch_embed: PatchEmbed;
  private readonly blocks: nn.Sequential;
  private readonly norm: nn.LayerNorm;
  private readonly head: nn.Linear;
  private readonly patches: number;
  private readonly dim: number;

  constructor(numClasses: number, v: Variant = TINY, image = 224) {
    super();
    const grid = image / PATCH;
    this.patches = grid * grid;
    this.dim = v.dim;

    // 코어가 파라미터를 만드는 방식 그대로다(`BatchNormND` 를 보라) — 텐서를
    // 만들고 `claim` 으로 붙든다. 안 붙들면 스코프가 닫힐 때 값이 풀린다.
    this.cls_token = Tensor.owned([1, 1, v.dim], 0);
    this.pos_embed = Tensor.owned([1, this.patches + 1, v.dim], 0);
    this.claim(this.cls_token, this.pos_embed);

    this.patch_embed = new PatchEmbed(v.dim, PATCH);
    this.blocks = new nn.Sequential(
      Array.from({ length: v.depth }, () => new Block(v.dim, v.heads)),
    );
    this.norm = new nn.LayerNorm([v.dim], NORM_EPS);
    this.head = new nn.Linear(v.dim, numClasses, true);
  }

  /**
   * 층에 안 든 파라미터 둘. **이름이 곧 열쇠다.**
   */
  override ownParameters(): Record<string, Tensor> {
    return { cls_token: this.cls_token, pos_embed: this.pos_embed };
  }

  /**
   * timm 의 `forward_features` — 정규화까지 마친 **토큰 열** `[N, 1 + 패치 수, D]`.
   * cls 토큰이 0 번이다. 동결 백본으로 쓸 때 `forwardHead(h, true)` 와 짝이다.
   */
  forwardFeatures(x: Tensor): Tensor {
    const tokens = this.patch_embed.forward(x);               // [B, N, D]
    const [batch = 1] = tokens.shape;

    // cls 토큰을 앞에 붙인다. torch 의 `torch.cat([cls, x], dim=1)` 과 같은
    // 자리에 같은 이름으로 있다 — **메서드가 아니라 static 이라 못 찾았었다.**
    const cls = this.cls_token.expand(batch, 1, this.dim);
    let h = Tensor.cat([cls, tokens], 1).add(this.pos_embed);

    h = this.blocks.forward(h);
    return this.norm.forward(h);
  }

  /**
   * timm 의 `forward_head` — **0 번 토큰만** 뽑아 분류기까지(평균을 내면 그럴듯한
   * 다른 모델이 된다). `preLogits` 면 분류기 앞의 cls 토큰 `[N, numFeatures]` 다.
   */
  forwardHead(h: Tensor, preLogits = false): Tensor {
    const cls = h.select(1, 0);
    return preLogits ? cls : this.head.forward(cls);
  }

  /** 분류기 앞 벡터의 길이. timm 의 `num_features`. */
  get numFeatures(): number {
    return this.dim;
  }

  override forward(x: Tensor): Tensor {
    return this.forwardHead(this.forwardFeatures(x));
  }
}

/** timm 의 `vit_tiny_patch16_224`. */
export function vitTinyPatch16(numClasses: number): VisionTransformer {
  return new VisionTransformer(numClasses, TINY, 224);
}

/** timm 의 `vit_small_patch16_224`. tiny 와 갈리는 것은 `dim` 과 `heads` 뿐이다. */
export function vitSmallPatch16(numClasses: number): VisionTransformer {
  return new VisionTransformer(numClasses, SMALL, 224);
}

/** timm 의 `vit_base_patch16_224`. */
export function vitBasePatch16(numClasses: number): VisionTransformer {
  return new VisionTransformer(numClasses, BASE, 224);
}

/** 판마다의 계획. 검사가 셋을 같은 자리에서 묻는다. */
export const VARIANTS = { tiny: TINY, small: SMALL, base: BASE } as const;
