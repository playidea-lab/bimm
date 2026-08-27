/**
 * 남은 두 계열의 수를 **timm 과 대 본다** — MobileNetV2 와 ViT-Tiny.
 *
 * 앞의 둘(`efficientnet-plan`·`mobilenetv3-plan`)과 같은 뜻이고, 여기서 끝나면
 * 카탈로그의 아홉 이름이 전부 GPU 없이 검사되는 수를 갖는다.
 *
 * ## 두 계열이 서로 다른 이유로 위험하다
 *
 * **MobileNetV2** 는 MobileNetV3 와 같다 — 배율이 없고 표를 손으로 옮겼다. 다만
 * 갈리는 자리가 하나 더 있다: 확장이 1 인 단계에는 **확장 층이 없다.** `mid` 가
 * `cin` 과 같은 그 자리에 `conv_pw` 를 두면 열쇠가 하나 늘어 체크포인트가 안 실린다.
 *
 * **ViT** 는 표가 없다. 상수 여섯 개뿐인데, 그 상수에서 나오는 파생값들이 전부
 * 곱셈이나 나눗셈 하나라 **틀려도 그럴듯하다.** `pos_embed` 길이에서 cls 토큰을
 * 빼먹으면 196 이 되고, 그 수는 패치 수와 같아서 어디를 봐도 자연스럽다.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { type BlockPlan, mobilenetv2Plan } from "../src/mobilenet.js";
import { vitTinyPlan } from "../src/vit.js";

interface V2Row {
  readonly kind: string;
  readonly cin: number;
  readonly mid: number;
  readonly cout: number;
  readonly stride: number;
}

interface Fixtures {
  readonly mobilenetv2_100: {
    readonly stem: number;
    readonly stages: readonly (readonly V2Row[])[];
    readonly head: number;
  };
  readonly vit_tiny_patch16_224: Readonly<Record<string, number>>;
}

const here = dirname(fileURLToPath(import.meta.url));
const TIMM = JSON.parse(
  readFileSync(join(here, "..", "..", "test", "plans.json"), "utf8"),
) as Fixtures;

test("mobilenetv2_100 의 블록 표가 timm 과 같다", () => {
  const want = TIMM.mobilenetv2_100;
  const got = mobilenetv2Plan();

  assert.equal(got.stem, want.stem, "stem");
  assert.equal(got.head, want.head, "head");
  assert.equal(got.stages.length, want.stages.length, "단계 수");

  for (const [s, wantStage] of want.stages.entries()) {
    const gotStage = got.stages[s];
    assert.ok(gotStage, `단계 ${s} 가 없다`);
    assert.equal(gotStage.length, wantStage.length, `단계 ${s} 의 블록 수`);
    for (const [b, row] of wantStage.entries()) {
      const block: BlockPlan | undefined = gotStage[b];
      assert.ok(block, `blocks.${s}.${b} 가 없다`);
      const at = `blocks.${s}.${b}`;
      assert.equal(block.kind, row.kind, `${at} 의 종류`);
      assert.equal(block.cin, row.cin, `${at} 의 입력 채널`);
      assert.equal(block.mid, row.mid, `${at} 의 넓힌 채널`);
      assert.equal(block.cout, row.cout, `${at} 의 출력 채널`);
      assert.equal(block.stride, row.stride, `${at} 의 stride`);
    }
  }
});

test("확장이 1 인 단계에만 확장 층이 없다", () => {
  // `mid === cin` 인 블록이 곧 확장 층이 없는 블록이다. 첫 단계에만 있어야 하고,
  // 다른 단계에 생기면 그 블록의 열쇠 하나가 사라진다.
  const stages = mobilenetv2Plan().stages;
  for (const [s, blocks] of stages.entries()) {
    for (const b of blocks) {
      const flat = b.mid === b.cin;
      assert.equal(flat, s === 0,
        `단계 ${s}: 확장 없는 블록은 첫 단계에만 있어야 한다 (mid=${b.mid} cin=${b.cin})`);
    }
  }
});

test("vit_tiny_patch16_224 의 수가 timm 과 같다", () => {
  const want = TIMM.vit_tiny_patch16_224;
  const got = vitTinyPlan() as unknown as Readonly<Record<string, number>>;
  for (const key of Object.keys(want)) {
    assert.equal(got[key], want[key], `${key}`);
  }
});

test("ViT 의 파생값이 상수에서 실제로 나온다", () => {
  const p = vitTinyPlan();
  // 픽스처와 대 보는 것만으로는 **상수를 무시하고 수를 박아 넣은 경우**를 못 잡는다.
  // 관계식이 성립하는지를 따로 본다.
  assert.equal(p.headDim * p.heads, p.dim, "head 차원 × head 수 = 임베딩 차원");
  assert.equal(p.posLen, p.patches + 1, "pos_embed 는 패치 수 + cls 토큰 하나");
  assert.equal(p.qkvOut, p.dim * 3, "qkv 는 셋을 함께 낸다");

  // 다른 크기에서도 관계가 유지되는지 — 224 에만 맞춰 박아 둔 수는 여기서 걸린다.
  const bigger = vitTinyPlan(384);
  assert.equal(bigger.patches, (384 / 16) ** 2, "384 의 패치 수");
  assert.equal(bigger.posLen, bigger.patches + 1, "384 의 pos_embed 길이");
});
