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

/** ViT-Tiny/16 의 수. 판을 늘릴 때 갈리는 것은 이 네 개다. */
const DIM = 192;
const HEADS = 3;
const DEPTH = 12;
const PATCH = 16;
/** MLP 가 넓히는 배수. timm 의 `mlp_ratio` 는 4 로 고정이다. */
const MLP_RATIO = 4;
/** LayerNorm 의 eps. torch 기본값(1e-5)이 아니라 **1e-6** 이다. */
const NORM_EPS = 1e-6;

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
  private readonly scale: number;

  constructor(dim: number, heads: number) {
    super();
    this.qkv = new nn.Linear(dim, dim * 3, true);
    this.proj = new nn.Linear(dim, dim, true);
    this.headDim = dim / heads;
    // **곱하기 전에 나눈다.** torch 도 `q * scale` 을 먼저 하고, 순서를 바꾸면 큰
    // 내적에서 float32 가 먼저 넘친다.
    this.scale = 1 / Math.sqrt(this.headDim);
  }

  override forward(x: Tensor): Tensor {
    const [batch = 1, tokens = 1] = x.shape;
    const dim = this.headDim * HEADS;

    const fused = this.qkv.forward(x);                       // [B, N, 3D]
    // [B, N, 3, H, d] → [3, B, H, N, d] 로 세우면 셋을 같은 모양으로 꺼낼 수 있다.
    const parts = fused.reshape([batch, tokens, 3, HEADS, this.headDim])
      .permute([2, 0, 3, 1, 4]);
    const q = parts.select(0, 0);
    const k = parts.select(0, 1);
    const v = parts.select(0, 2);

    // 헤드를 배치로 접어 `bmm` 에 넣는다 — 코어의 행렬곱은 3 차원까지 본다.
    const folded = [batch * HEADS, tokens, this.headDim];
    // 스칼라 곱도 텐서를 거친다 — 코어의 `mul` 은 텐서만 받고, 0 차원은
    // 브로드캐스트로 펼쳐진다.
    const qf = q.reshape(folded).mul(Tensor.owned([], this.scale));
    const kf = k.reshape(folded);
    const vf = v.reshape(folded);

    const scores = qf.bmm(kf.transpose(-2, -1));             // [B*H, N, N]
    const weights = scores.softmax(-1);
    const mixed = weights.bmm(vf);                           // [B*H, N, d]

    // 접었던 헤드를 다시 펴서 [B, N, D] 로.
    const merged = mixed.reshape([batch, HEADS, tokens, this.headDim])
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
    const [batch = 1, dim = DIM, gh = 1, gw = 1] = grid.shape;
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

  constructor(numClasses: number, image = 224) {
    super();
    const grid = image / PATCH;
    this.patches = grid * grid;

    // 코어가 파라미터를 만드는 방식 그대로다(`BatchNormND` 를 보라) — 텐서를
    // 만들고 `claim` 으로 붙든다. 안 붙들면 스코프가 닫힐 때 값이 풀린다.
    this.cls_token = Tensor.owned([1, 1, DIM], 0);
    this.pos_embed = Tensor.owned([1, this.patches + 1, DIM], 0);
    this.claim(this.cls_token, this.pos_embed);

    this.patch_embed = new PatchEmbed(DIM, PATCH);
    this.blocks = new nn.Sequential(
      Array.from({ length: DEPTH }, () => new Block(DIM, HEADS)),
    );
    this.norm = new nn.LayerNorm([DIM], NORM_EPS);
    this.head = new nn.Linear(DIM, numClasses, true);
  }

  /**
   * 층에 안 든 파라미터 둘. **이름이 곧 열쇠다.**
   */
  override ownParameters(): Record<string, Tensor> {
    return { cls_token: this.cls_token, pos_embed: this.pos_embed };
  }

  override forward(x: Tensor): Tensor {
    const tokens = this.patch_embed.forward(x);               // [B, N, D]
    const [batch = 1] = tokens.shape;

    // cls 토큰을 앞에 붙인다. torch 의 `torch.cat([cls, x], dim=1)` 과 같은
    // 자리에 같은 이름으로 있다 — **메서드가 아니라 static 이라 못 찾았었다.**
    const cls = this.cls_token.expand(batch, 1, DIM);
    let h = Tensor.cat([cls, tokens], 1).add(this.pos_embed);

    h = this.blocks.forward(h);
    h = this.norm.forward(h);
    // **0 번 토큰만.** 평균을 내면 그럴듯한 다른 모델이 된다.
    return this.head.forward(h.select(1, 0));
  }
}

/** timm 의 `vit_tiny_patch16_224`. */
export function vitTinyPatch16(numClasses: number): VisionTransformer {
  return new VisionTransformer(numClasses, 224);
}
