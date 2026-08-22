/**
 * 인자 검사가 **오타와 누락을 병합 전에 잡는지** 본다.
 *
 * 이 검사들은 GPU 없이 돈다. 그것이 `args.ts` 를 `registry.ts` 에서 떼어 놓은
 * 이유다 — 값이 틀렸다는 것을 확인하는 데 WebGPU 어댑터가 필요하면, 레지스트리
 * CI 는 매니페스트를 못 본다.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { BimmError } from "../src/errors.js";
import { checkArgs, type FactoryArgs } from "../src/args.js";

const RESNET: FactoryArgs = { numClasses: { kind: "int", min: 1 } };

function rejects(given: Record<string, unknown>, mentions: string): void {
  assert.throws(
    () => checkArgs("resnet18", RESNET, given),
    (err: unknown) => {
      assert.ok(err instanceof BimmError, `BimmError 여야 합니다 — ${String(err)}`);
      assert.ok(err.message.includes(mentions), `'${mentions}' 를 짚어야 합니다 — ${err.message}`);
      return true;
    },
  );
}

test("규격에 맞는 인자는 수로 좁혀져 나온다", () => {
  assert.deepEqual(checkArgs("resnet18", RESNET, { numClasses: 10 }), { numClasses: 10 });
});

test("오타 난 인자는 조용히 무시되지 않고 거절된다", () => {
  // 무시하면 기본값 모델이 만들어지고, 올린 사람은 자기가 무엇을 올렸는지 모른다.
  rejects({ numClases: 10 }, "numClases");
});

test("빠진 인자는 기본값으로 메우지 않고 거절한다", () => {
  rejects({}, "numClasses");
});

test("정수가 아닌 값은 거절한다", () => {
  rejects({ numClasses: 10.5 }, "정수");
  rejects({ numClasses: "10" }, "정수");
});

test("범위를 벗어난 값은 거절한다", () => {
  rejects({ numClasses: 0 }, "1 이상");
});

test("거절 메시지가 받는 인자 목록을 같이 말한다", () => {
  // 무엇이 틀렸는지만 말하고 무엇이 맞는지를 안 말하면 올리는 쪽이 또 물어본다.
  assert.throws(
    () => checkArgs("resnet18", RESNET, { classes: 10 }),
    (err: unknown) => err instanceof BimmError && err.message.includes("numClasses"),
  );
});
