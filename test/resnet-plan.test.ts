/**
 * ResNet-50 의 수를 **timm 이 실제로 세운 모델과 대 본다.**
 *
 * ## 이 계열에서 틀리기 쉬운 자리
 *
 * **`downsample` 이 어디 붙는가.** 각 `layer` 의 첫 블록에만 붙는데, 붙는 조건을
 * `stride > 1` 로 읽으면 `layer1` 이 빠진다 — 거기는 stride 가 1 이면서 채널이
 * 64 에서 256 으로 바뀌기 때문이다. 그 하나가 빠지면 그 블록의 열쇠 여섯이 통째로
 * 사라지고, strict 로 실을 때 걸린다. 걸리는 편이 낫지만 **걸리기 전에 여기서**
 * 잡는 편이 더 낫다.
 *
 * **stride 를 어느 컨볼루션이 지는가.** timm 은 가운데 3×3 에 둔다(torchvision 도
 * v1.5 이후 같다). 첫 1×1 에 두면 **모양은 그대로이고 수만 달라진다** — 열쇠도 다
 * 맞고 체크포인트도 실리고, 그저 답이 틀린다. 이 검사가 `stride` 를 블록마다 보는
 * 까닭이다.
 *
 * ## 열쇠 수도 센다
 *
 * 320 개다. 계획이 맞아도 층을 잘못 엮으면(예: `layer1..4` 를 `Sequential` 하나에
 * 담으면) 이름이 `layers.0...` 이 되어 전부 갈린다. 그 수까지는 이 검사가 못 보지만,
 * 블록 수와 자리는 여기서 못박고 이름은 parity 가 본다.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { type BlockPlan, RESNETS, resnetPlan } from "../src/resnet50.js";

interface Down {
  readonly cin: number;
  readonly cout: number;
  readonly kernel: number;
  readonly stride: number;
}

interface Row {
  readonly cin: number;
  readonly width: number;
  readonly cout: number;
  readonly stride: number;
  readonly downsample: Down | null;
}

const here = dirname(fileURLToPath(import.meta.url));
const TIMM = JSON.parse(
  readFileSync(join(here, "..", "..", "test", "resnet-plan.json"), "utf8"),
) as Readonly<Record<string, {
  readonly block: string;
  readonly stem: number;
  readonly layers: readonly (readonly Row[])[];
  readonly fcIn: number;
  readonly keys: number;
}>>;

for (const [name, want] of Object.entries(TIMM)) {
  test(`${name} 의 계획이 timm 과 같다`, () => {
  const got = resnetPlan(name);
  assert.equal(got.block, want.block, "블록 종류");

  assert.equal(got.stem, want.stem, "stem");
  assert.equal(got.fcIn, want.fcIn, "fc 가 받는 폭");
  assert.equal(got.layers.length, want.layers.length, "layer 수");

  for (const [l, wantLayer] of want.layers.entries()) {
    const gotLayer = got.layers[l];
    assert.ok(gotLayer, `layer${l + 1} 이 없다`);
    assert.equal(gotLayer.length, wantLayer.length, `layer${l + 1} 의 블록 수`);

    for (const [b, row] of wantLayer.entries()) {
      const block: BlockPlan | undefined = gotLayer[b];
      assert.ok(block, `layer${l + 1}.${b} 가 없다`);
      const at = `layer${l + 1}.${b}`;
      assert.equal(block.cin, row.cin, `${at} 의 입력 채널`);
      assert.equal(block.width, row.width, `${at} 의 좁힌 폭`);
      assert.equal(block.cout, row.cout, `${at} 의 출력 채널`);
      assert.equal(block.stride, row.stride, `${at} 의 stride`);

      if (row.downsample === null) {
        assert.equal(block.downsample, null, `${at} 에 downsample 이 없어야 한다`);
      } else {
        assert.ok(block.downsample, `${at} 에 downsample 이 있어야 한다`);
        assert.equal(block.downsample.cin, row.downsample.cin, `${at} downsample 입력`);
        assert.equal(block.downsample.cout, row.downsample.cout, `${at} downsample 출력`);
        assert.equal(block.downsample.stride, row.downsample.stride,
          `${at} downsample stride`);
      }
    }
  }
  });
}

test("Bottleneck 은 layer1 의 첫 블록도 downsample 을 든다", () => {
  // **stride 가 1 인데 필요한 자리다.** 채널이 64 에서 256 으로 바뀌기 때문이고,
  // `stride > 1` 을 조건으로 읽으면 여기만 빠진다. 조건을 그렇게 쓰는 것은 그럴듯해서
  // 따로 못박는다.
  const first = resnetPlan("resnet50").layers[0]?.[0];
  assert.ok(first, "layer1 의 첫 블록이 없다");
  assert.equal(first.stride, 1, "layer1 은 크기를 안 줄인다");
  assert.ok(first.downsample, "그래도 downsample 은 있어야 한다 — 채널이 바뀐다");
  assert.equal(first.downsample.cin, 64);
  assert.equal(first.downsample.cout, 256);
});

test("downsample 은 모양이 바뀌는 자리에만 있다", () => {
  // **"첫 블록" 이 아니라 "모양이 바뀌는가" 다.** Bottleneck 은 넓히므로 네 layer 의
  // 첫 블록이 전부 해당하지만, BasicBlock 은 안 넓혀서 `layer1` 의 첫 블록이
  // 해당하지 않는다 — stride 도 1 이고 채널도 그대로다. 조건을 "첫 블록" 으로 쓰면
  // resnet18 의 `layer1` 에 timm 에 없는 열쇠 둘이 생긴다.
  for (const name of RESNETS) {
    for (const [l, blocks] of resnetPlan(name).layers.entries()) {
      for (const [b, block] of blocks.entries()) {
        const changes = block.stride !== 1 || block.cin !== block.cout;
        assert.equal(block.downsample !== null, changes,
          `${name} layer${l + 1}.${b}: 모양이 바뀌는 자리에만 있어야 한다`);
      }
    }
  }
});

test("BasicBlock 의 layer1 에는 downsample 이 없다", () => {
  // Bottleneck 쪽과 정반대다. 두 검사를 나란히 두는 것은 **한쪽 규칙을 다른 쪽에
  // 적용하는 것**이 이 계열에서 가장 그럴듯한 실수이기 때문이다.
  for (const name of ["resnet18", "resnet34"]) {
    const first = resnetPlan(name).layers[0]?.[0];
    assert.ok(first, `${name} 의 layer1 첫 블록이 없다`);
    assert.equal(first.cin, first.cout, `${name}: 넓히지 않으므로 채널이 같다`);
    assert.equal(first.downsample, null, `${name}: 맞출 것이 없으므로 없어야 한다`);
  }
});

test("넓히는 배수가 블록 종류를 따른다", () => {
  // Bottleneck 을 Bottleneck 이게 하는 수다. basic 은 1, bottleneck 은 4 이고,
  // 섞이면 채널이 전부 어긋난다.
  for (const name of RESNETS) {
    const plan = resnetPlan(name);
    const want = plan.block === "basic" ? 1 : 4;
    for (const block of plan.layers.flat()) {
      assert.equal(block.cout, block.width * want,
        `${name}: 좁힌 폭 ${block.width} 는 ${block.width * want} 여야 한다`);
    }
  }
});

test("픽스처가 다섯 판을 다 든다", () => {
  // 판을 늘리고 픽스처를 안 늘리면 위 검사들이 **조용히 줄어든다** — 실패가 아니라
  // 그냥 안 물어본 것이 되고, 화면은 초록이다.
  assert.deepEqual(Object.keys(TIMM).sort(), [...RESNETS].sort());
});
