/**
 * 카탈로그가 **이름을 어떻게 받고 어떻게 거절하는지** 본다.
 *
 * ## 여기서 모델은 하나도 안 만들어진다
 *
 * 만들려면 WebGPU 어댑터가 들고, 그것은 브라우저에서만 잡힌다. 그래서 이 파일이
 * 보는 것은 만들기 **전까지** — 이름을 찾고, 인자를 좁히고, 아니면 거절하는 데까지다.
 * 실제로 층이 서는지는 코어의 브라우저 하네스가 본다.
 *
 * 그 경계가 이 패키지 설계의 주장이기도 하다: 매니페스트가 틀렸다는 것을 확인하는
 * 데 GPU 가 있는 기계가 필요하면, 레지스트리 CI 는 매니페스트를 못 본다. 아래
 * 마지막 검사가 그 주장을 그대로 시험한다.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { BimmError } from "../src/errors.js";
import { createModel, factorySpec, listModels } from "../src/registry.js";

const LIBRARY = "borchvision";
const FACTORY = "resnet18_cifar";

test("카탈로그가 실린 이름을 든다", () => {
  assert.deepEqual(listModels(), [
    { library: LIBRARY, factory: FACTORY },
    // 표는 이름순으로 나온다 — 아래 "정해진 순서로 나온다" 가 지키는 규칙이고,
    // 표에 적은 차례가 아니라 이름이 순서를 정한다.
    { library: "timm", factory: "efficientnet_b0" },
    { library: "timm", factory: "mobilenetv2_100" },
    { library: "timm", factory: "mobilenetv3_large_100" },
    { library: "timm", factory: "mobilenetv3_small_100" },
  ]);
});

test("같은 표에 라이브러리가 둘 선다", () => {
  // 이름을 둘로 받은 까닭이 여기서 처음 눈에 보인다 — 표 하나가 두 출신을 든다.
  const libraries = new Set(listModels().map((f) => f.library));
  assert.deepEqual([...libraries].sort(), ["borchvision", "timm"]);
});

test("카탈로그는 정해진 순서로 나온다", () => {
  // 발견 레이어가 이 순서를 그대로 보여 준다. 표에 적은 순서가 새면 같은 카탈로그가
  // 볼 때마다 다르게 보인다.
  const names = listModels().map((f) => `${f.library}/${f.factory}`);
  assert.deepEqual(names, [...names].sort());
});

test("인자 규격을 물어볼 수 있다", () => {
  // 매니페스트를 쓰는 쪽이 무엇을 적어야 하는지 알아야 한다 — 그래서 표가 밖으로
  // 열린다.
  assert.deepEqual(factorySpec(LIBRARY, FACTORY), {
    numClasses: { kind: "int", min: 1 },
  });
});

test("모르는 팩토리는 거절하고 무엇이 있는지 같이 말한다", () => {
  assert.throws(
    () => createModel(LIBRARY, "resnet50_cifar", { numClasses: 10 }),
    (err: unknown) => {
      assert.ok(err instanceof BimmError, `BimmError 여야 합니다 — ${String(err)}`);
      assert.ok(err.message.includes("resnet50_cifar"), err.message);
      // 무엇이 없는지만 말하고 무엇이 있는지를 안 말하면 부르는 쪽이 또 물어본다.
      assert.ok(err.message.includes(`${LIBRARY}/${FACTORY}`), err.message);
      return true;
    },
  );
});

test("모르는 라이브러리도 같은 자리에서 거절된다", () => {
  // 이름공간이 둘인 것이 이 패키지가 timm 과 갈리는 자리다 — 앞자리도 표를 탄다.
  assert.throws(
    () => factorySpec("timm", FACTORY),
    (err: unknown) => err instanceof BimmError && err.message.includes("timm"),
  );
});

test("인자 검사가 모델을 만들기 전에 온다", () => {
  // GPU 가 없는 이 자리에서 오타를 BimmError 로 거절한다는 것이, 검사가 build 앞에
  // 있다는 뜻이다. 순서가 뒤집히면 여기서 나는 것은 코어의 "no device" 가 되고,
  // 레지스트리 CI 는 매니페스트의 오타를 영영 못 본다.
  assert.throws(
    () => createModel(LIBRARY, FACTORY, { numClases: 10 }),
    (err: unknown) => {
      assert.ok(err instanceof BimmError, `BimmError 여야 합니다 — ${String(err)}`);
      assert.ok(err.message.includes("numClases"), err.message);
      return true;
    },
  );
});
