/**
 * 채널 반올림 규칙을 **timm 이 낸 수와 대 본다.**
 *
 * ## 이 검사가 이 저장소에서 처음인 자리
 *
 * 여기까지 CI 가 보던 것은 인자 검사·카탈로그 표·ResNet 뿐이었다. timm 에서 옮겨 온
 * 여덟 아키텍처(900 줄 남짓)는 **어떤 검사도 통과하지 않고 있었다** — 이름이 카탈로그
 * 목록에 있는지만 확인됐다. 채널 수를 하나 바꿔도 `npm test` 는 전부 초록이었다.
 *
 * 그 여덟을 지키던 것은 parity 하네스 하나인데, 그것은 브라우저·GPU·timm 이 다 있어야
 * 돌아서 CI 에서는 못 돈다. **사람이 기억해야만 도는 검사**였다는 뜻이다.
 *
 * 이 규칙은 순수한 산수라 그 셋이 없어도 돈다. 그래서 여기서 시작한다.
 *
 * ## 값은 어디서 왔는가
 *
 * 전부 timm 을 실제로 돌려 받아 적은 것이다 — `timm.layers.make_divisible` 과
 * `timm.models._efficientnet_builder.round_channels`. 손으로 계산한 값을 적으면
 * 검사가 내 산수를 채점하게 되고, 그 산수가 원본과 갈린 것이 애초에 이 규칙을 한
 * 곳으로 모은 까닭이다.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { makeDivisible, roundChannels } from "../src/channels.js";

/** MobileNetV3 의 SE 가 좁히는 자리 — `mid * 0.25`. timm 이 낸 수. */
const SE: readonly (readonly [number, number])[] = [
  [16, 8], [72, 24], [88, 24], [96, 24], [120, 32], [144, 40],
  [240, 64], [288, 72], [480, 120], [576, 144], [672, 168], [960, 240],
];

/** EfficientNet 의 채널 표를 배율마다. b0=1.0 · b2=1.1 · b3=1.2. */
const BY_WIDTH: readonly (readonly [number, readonly (readonly [number, number])[]])[] = [
  [1.0, [[32, 32], [16, 16], [24, 24], [40, 40], [80, 80],
    [112, 112], [192, 192], [320, 320], [1280, 1280]]],
  [1.1, [[32, 32], [16, 16], [24, 24], [40, 48], [80, 88],
    [112, 120], [192, 208], [320, 352], [1280, 1408]]],
  [1.2, [[32, 40], [16, 24], [24, 32], [40, 48], [80, 96],
    [112, 136], [192, 232], [320, 384], [1280, 1536]]],
];

test("SE 가 좁히는 채널이 timm 과 같다", () => {
  for (const [mid, want] of SE) {
    assert.equal(makeDivisible(mid * 0.25), want, `mid=${mid}`);
  }
});

test("배율을 먹인 채널이 timm 과 같다", () => {
  for (const [width, rows] of BY_WIDTH) {
    for (const [channels, want] of rows) {
      assert.equal(roundChannels(channels, width), want,
        `width=${width} channels=${channels}`);
    }
  }
});

test("내림이 10% 를 넘으면 한 칸 올린다", () => {
  // 이 한 줄이 없으면 MobileNetV3 의 여러 블록이 어긋난다. 18 → 16 은 16.2 보다
  // 작으므로 24 가 되어야 한다 — 그것이 `72 * 0.25` 자리다.
  for (const [v, want] of [[18, 24], [20, 24], [4, 8], [12, 16], [13, 16]] as const) {
    assert.equal(makeDivisible(v), want, `v=${v}`);
  }
});

test("배율 1 도 그냥 지나가지 않는다", () => {
  // **timm 이 원래 값을 돌려주는 것은 배율이 0 일 때뿐이다.** 전에 이 자리에
  // `width === 1` 이면 그대로 돌려주는 지름길이 있었고, 지금 표의 채널이 전부 8 의
  // 배수라서 결과가 같았을 뿐이다.
  for (const [c, want] of [[17, 16], [30, 32], [100, 104], [3, 8],
    [1, 8], [7, 8], [9, 16]] as const) {
    assert.equal(roundChannels(c, 1.0), want, `channels=${c}`);
  }
});

test("배율이 0 이면 원래 값을 그대로 돌려준다", () => {
  assert.equal(roundChannels(123, 0), 123);
});

test("두 모델 계열이 같은 규칙을 부른다", () => {
  // 전에는 같은 세 줄이 `efficientnet.ts` 와 `mobilenetv3.ts` 에 따로 있었다. 이름도
  // 설명도 달랐지만 계산은 글자까지 같았고, 갈리면 그 블록만 채널이 어긋나 가중치가
  // 안 실린다. 한 곳으로 모은 것을 여기서 못박는다 — 다시 갈라지면 이 검사가 아니라
  // 이 문장이 먼저 눈에 띄어야 한다.
  const asSe = makeDivisible(72 * 0.25);
  const asWidth = roundChannels(72 * 0.25, 1.0);
  assert.equal(asSe, asWidth,
    "SE 쪽과 배율 쪽이 같은 규칙을 통과해야 한다 — 갈리면 규칙이 두 벌이 된 것이다");
  assert.equal(asSe, 24);
});
