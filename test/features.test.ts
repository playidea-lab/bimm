/**
 * 모든 모델 클래스가 **같은 세 자리**를 가진다 — `forwardFeatures`, `forwardHead`,
 * `numFeatures` — 그리고 `forward` 는 그 둘의 합성이다.
 *
 * 왜 소스를 보는가: `samemodel.test.ts` 와 같은 이유다. 층이 곧 텐서라 모델을 만들려면
 * WebGPU 어댑터가 들고, 그것은 브라우저 하네스에만 있다. 실제 값의 대조(`forward` ==
 * `forwardHead(forwardFeatures(x))`, `preLogits` 의 길이 == `numFeatures`)는 borch 쪽
 * `hub:py` 프로브가 GPU 위에서 한다. 여기서 잡는 것은 **자리를 빼먹은 클래스**다 —
 * 새 계열이 들어올 때 `forward` 만 쓰고 지나가면 동결 백본으로 쓸 수 없고, 그 사실은
 * 현장 학습기가 그 모델을 고르는 날에야 드러난다.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = (name: string): string => readFileSync(join(here, "..", "..", "src", name), "utf8");

const MODELS: readonly (readonly [string, string])[] = [
  ["efficientnet.ts", "EfficientNet"],
  ["mobilenet.ts", "MobileNetV2"],
  ["mobilenetv3.ts", "MobileNetV3"],
  ["resnet.ts", "ResNet18Cifar"],
  ["resnet50.ts", "ResNet"],
  ["vit.ts", "VisionTransformer"],
];

for (const [file, cls] of MODELS) {
  test(`${cls} carries forwardFeatures, forwardHead and numFeatures, and forward composes them`, () => {
    const text = src(file);
    const start = text.indexOf(`export class ${cls} extends nn.Module`);
    assert.ok(start >= 0, `${cls} is not in ${file}`);
    const body = text.slice(start, text.indexOf("\n}\n", start));
    for (const member of ["forwardFeatures(x: Tensor): Tensor", "forwardHead(h: Tensor, preLogits = false): Tensor", "get numFeatures(): number"]) {
      assert.ok(body.includes(member), `${cls} lacks \`${member}\``);
    }
    assert.ok(body.includes("return this.forwardHead(this.forwardFeatures(x));"),
      `${cls}.forward is not forwardHead(forwardFeatures(x))`);
  });
}
