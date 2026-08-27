/**
 * MobileNetV3 의 블록 표를 **timm 이 실제로 세운 모델과 대 본다.**
 *
 * ## 여기서 위험한 것은 산수가 아니라 표다
 *
 * EfficientNet 쪽은 배율을 먹인 채널이 틀릴 수 있는 계열이었다. 이쪽에는 배율이
 * 없다 — `_100` 두 판이 전부이고, `mid` 와 `cout` 은 계산되는 것이 아니라 표에
 * **절대 수로 손으로 적혀 있다.**
 *
 * 그래서 이 검사가 보는 것은 옮겨 적은 수 26 줄이다. 한 칸이 틀리면 그 블록만
 * 채널이 어긋나고 체크포인트가 안 실린다. 손으로 옮긴 표는 눈으로 두 번 봐도
 * 틀리며, 그것을 잡는 방법은 원본과 기계로 대 보는 것뿐이다.
 *
 * ## SE 가 붙는 자리와 활성화도 본다
 *
 * 이 계열은 블록마다 **SE 가 있기도 없기도** 하고 활성화가 `relu` 와 `hardswish`
 * 로 갈린다. 둘 다 채널 수를 안 바꾸므로 **모양만 보는 검사는 통과시킨다** —
 * SE 가 빠진 자리는 열쇠가 통째로 없어 strict 로 실을 때 걸리지만, 활성화가 갈린
 * 것은 아무 데서도 안 걸리고 조용히 다른 수를 낸다.
 *
 * 기대값은 timm 을 세워 층에서 읽었다. `bn1` 이 `BatchNormAct2d` 라 활성화가 그
 * 안에 들어 있어서, 처음에 `blk.act1` 을 찾다가 없다는 말을 들었다.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { type BlockPlan, mobilenetv3Plan } from "../src/mobilenetv3.js";

interface Row {
  readonly kind: string;
  readonly cin: number;
  readonly mid: number;
  readonly cout: number;
  readonly kernel: number;
  readonly stride: number;
  readonly se: number;
  readonly act: string;
}

interface Expected {
  readonly which: "large" | "small";
  readonly stem: number;
  readonly stages: readonly (readonly Row[])[];
  readonly wide: number;
  readonly head: number;
}

const here = dirname(fileURLToPath(import.meta.url));
const TIMM = JSON.parse(
  readFileSync(join(here, "..", "..", "test", "mobilenetv3-plan.json"), "utf8"),
) as Readonly<Record<string, Expected>>;

for (const [name, want] of Object.entries(TIMM)) {
  test(`${name} 의 블록 표가 timm 과 같다`, () => {
    const got = mobilenetv3Plan(want.which);

    assert.equal(got.stem, want.stem, "stem");
    assert.equal(got.wide, want.wide, "blocks 끝의 1×1 이 내는 채널");
    assert.equal(got.head, want.head, "head");
    // **끝의 1×1 은 단계로 세지 않는다.** timm 은 그것을 `blocks` 의 마지막
    // 단계로 두고, 우리 계획은 `wide` 로 따로 든다 — 블록이 아니라 채널 하나이기
    // 때문이다. 그래서 픽스처를 만들 때도 그 단계를 빼고 수를 `wide` 로 옮겼다.
    // 층으로 세울 때는 생성자가 그 자리에 `Sequential` 하나를 더한다.
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
        assert.equal(block.kernel, row.kernel, `${at} 의 커널`);
        assert.equal(block.stride, row.stride, `${at} 의 stride`);
        assert.equal(block.se, row.se, `${at} 의 SE 폭 (0 이면 SE 없음)`);
        assert.equal(block.act, row.act, `${at} 의 활성화`);
      }
    }
  });
}

test("SE 가 붙는 자리와 안 붙는 자리가 둘 다 있다", () => {
  // SE 를 전부 켜거나 전부 끄는 실수는 위 검사가 잡는다. 이것은 **픽스처 쪽이**
  // 한쪽으로 쏠렸을 때를 막는다 — 기대값이 전부 0 이면 위 검사는 우리 것도 전부
  // 0 일 때 통과하고, 그러면 둘이 사이좋게 틀린 채 초록이 된다.
  for (const which of ["large", "small"] as const) {
    const blocks = mobilenetv3Plan(which).stages.flat();
    assert.ok(blocks.some((b) => b.se > 0), `${which} 에 SE 가 하나도 없다`);
    assert.ok(blocks.some((b) => b.se === 0), `${which} 가 전부 SE 다`);
  }
});

test("활성화가 두 가지 다 쓰인다", () => {
  // 같은 이유다. 이 계열은 앞쪽이 relu 이고 뒤쪽이 hardswish 인데, 한 가지로
  // 뭉개도 채널 수는 그대로라 모양만 보는 검사는 통과한다.
  for (const which of ["large", "small"] as const) {
    const acts = new Set(mobilenetv3Plan(which).stages.flat().map((b) => b.act));
    assert.deepEqual([...acts].sort(), ["hardswish", "relu"], `${which} 의 활성화`);
  }
});
