/**
 * EfficientNet 이 뽑는 수를 **timm 이 실제로 세운 모델과 대 본다.**
 *
 * ## 무엇을 보는가
 *
 * 블록 88 개(b0 16 · b1 23 · b2 23 · b3 26)의 여섯 자리 — 들어오는 채널, 나가는
 * 채널, 커널, stride, 넓힌 채널, SE 가 좁히는 폭 — 과 stem·head 다. 여기가 어긋나면
 * **가중치가 안 실린다.** 모양이 우연히 맞으면 실린 다음 틀린 수를 낸다.
 *
 * ## 왜 GPU 없이 도는가
 *
 * `efficientnetPlan` 은 층을 만들지 않는다. 배율을 먹인 채널, 올림한 반복수, SE 폭
 * — 산수만 한다. 그래서 이 검사는 브라우저도 어댑터도 timm 도 없이 `npm test` 에서
 * 돈다.
 *
 * 그전까지 이 계열을 지키던 것은 브라우저를 띄우는 parity 하네스 하나였고, 그것은
 * CI 에서 못 돈다 — **사람이 기억해야만 도는 검사**였다. 채널 수를 하나 바꿔도 검사는
 * 전부 초록이었다.
 *
 * ## 기대값은 어디서 왔는가
 *
 * `efficientnet-plan.json` 은 timm 을 세워 **실제 층에서 읽어 적은 것**이다 — 표를
 * 보고 옮긴 것이 아니다. 이 저장소에서 표를 보고 옮긴 산수가 원본과 갈린 적이 있고
 * (`round_channels` 의 배율 1 자리), 그때도 화면은 초록이었다.
 *
 * 다시 뽑으려면 `timm.create_model` 로 세워 `blocks` 를 훑으면 된다. 값이 바뀌는
 * 경우는 timm 이 정의를 바꿨을 때뿐이고, 그때는 **우리가 따라가야 하는 것**이므로
 * 이 파일이 먼저 빨개지는 편이 맞다.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { type BlockPlan, efficientnetPlan } from "../src/efficientnet.js";

interface Row {
  readonly kind: string;
  readonly cin: number;
  readonly cout: number;
  readonly kernel: number;
  readonly stride: number;
  readonly mid: number;
  readonly se: number;
}

interface Expected {
  readonly width: number;
  readonly depth: number;
  readonly stem: number;
  readonly stages: readonly (readonly Row[])[];
  readonly head: number;
}

const here = dirname(fileURLToPath(import.meta.url));
const TIMM = JSON.parse(
  readFileSync(join(here, "..", "..", "test", "efficientnet-plan.json"), "utf8"),
) as Readonly<Record<string, Expected>>;

for (const [name, want] of Object.entries(TIMM)) {
  test(`${name} 의 채널 계획이 timm 과 같다`, () => {
    const got = efficientnetPlan(want.width, want.depth);

    assert.equal(got.stem, want.stem, "stem");
    assert.equal(got.head, want.head, "head");
    assert.equal(got.stages.length, want.stages.length, "단계 수");

    for (const [s, wantStage] of want.stages.entries()) {
      const gotStage = got.stages[s];
      assert.ok(gotStage, `단계 ${s} 가 없다`);
      // **반복수가 먼저다.** 여기가 틀리면 아래 자리 비교가 엉뚱한 짝을 본다.
      assert.equal(gotStage.length, wantStage.length, `단계 ${s} 의 블록 수`);

      for (const [b, row] of wantStage.entries()) {
        const block: BlockPlan | undefined = gotStage[b];
        assert.ok(block, `blocks.${s}.${b} 가 없다`);
        const at = `blocks.${s}.${b}`;
        assert.equal(block.kind, row.kind, `${at} 의 종류`);
        assert.equal(block.cin, row.cin, `${at} 의 입력 채널`);
        assert.equal(block.cout, row.cout, `${at} 의 출력 채널`);
        assert.equal(block.kernel, row.kernel, `${at} 의 커널`);
        assert.equal(block.stride, row.stride, `${at} 의 stride`);
        assert.equal(block.mid, row.mid, `${at} 의 넓힌 채널`);
        assert.equal(block.se, row.se, `${at} 의 SE 폭`);
      }
    }
  });
}

test("네 판이 실제로 다른 수를 낸다", () => {
  // 같은 표에서 나오므로, 배율을 안 먹이는 실수를 하면 넷이 같은 계획이 된다.
  // 그러면 위 검사 넷 중 하나만 맞고 셋이 틀릴 것 같지만, 픽스처를 잘못 만들면
  // 넷 다 같은 것을 보고 넷 다 통과한다. 그 경우를 여기서 막는다.
  const counts = [[1.0, 1.0], [1.0, 1.1], [1.1, 1.2], [1.2, 1.4]]
    .map(([w = 1, d = 1]) => {
      const p = efficientnetPlan(w, d);
      return `${p.stem}:${p.head}:${p.stages.reduce((n, s) => n + s.length, 0)}`;
    });
  assert.equal(new Set(counts).size, 4, `네 판이 갈려야 한다 — ${counts.join(" · ")}`);
});
