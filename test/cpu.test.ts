/**
 * plan 에서 지은 **cpu 그래프**가 체크포인트를 남김없이, 빠짐없이 읽고, 실제로 도는지 본다.
 *
 * ## GPU 없이 본다
 *
 * 여기서 모델(`nn.Module`)은 하나도 안 만들어진다 — 그것은 어댑터가 든다. 대신 timm 이
 * 내는 열쇠 이름과 모양을 plan 에서 **따로** 적고(아래 `keysOf`), 그 모양의 난수 텐서로
 * 체크포인트를 흉내 내서 변환에 넣는다. 변환이 읽은 열쇠와 흉내 낸 열쇠가 집합으로
 * 같아야 한다: 없는 것을 찾으면 변환이 던지고, 읽지 않은 것이 남으면 변환이 무엇인가를
 * 빼먹은 것이다. 그 뒤 borch-ts 의 `cpu.CpuRunner` 로 작은 입력을 실제로 통과시킨다.
 *
 * 값이 torch 와 같은지는 여기서 못 본다. 그것은 코어의 `npm run cpu:ts` 가 허브의 진짜
 * 체크포인트로 WebGPU 장치와 대조한다 — 이 파일은 그 앞의, GPU 없이 볼 수 있는 절반이다.
 *
 * `keysOf` 는 이름 규칙의 두 번째 사본이다. 사본은 갈리는 모양이지만, 검사가 소스와
 * 같은 규칙을 다시 쓰지 않으면 소스의 오타를 소스로 확인하는 꼴이 된다 — 두 벌이 서로를
 * 붙잡는 것이 여기서는 맞다.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { cpu } from "borch-ts";

import { CPU_FACTORIES, cpuGraphFor, efficientnetCpuGraph, resnetCpuGraph } from "../src/cpu.js";
import { efficientnetPlan, type Plan as EfficientNetPlan } from "../src/efficientnet.js";
import { BimmError } from "../src/errors.js";
import { listModels } from "../src/registry.js";
import { resnetPlan, type Plan as ResNetPlan } from "../src/resnet50.js";

type Shapes = Map<string, readonly number[]>;

function bnKeys(shapes: Shapes, prefix: string, c: number): void {
  for (const k of ["weight", "bias", "running_mean", "running_var"]) shapes.set(`${prefix}.${k}`, [c]);
}

/** timm `efficientnet_b*` 의 state_dict 열쇠와 모양 — `num_batches_tracked` 는 뺀다(추론에 안 쓰인다). */
function efficientnetKeys(plan: EfficientNetPlan, numClasses: number): Shapes {
  const s: Shapes = new Map();
  s.set("conv_stem.weight", [plan.stem, 3, 3, 3]); bnKeys(s, "bn1", plan.stem);
  plan.stages.forEach((blocks, si) => blocks.forEach((b, bi) => {
    const p = `blocks.${si}.${bi}`;
    if (b.kind === "dw") {
      s.set(`${p}.conv_dw.weight`, [b.cin, 1, b.kernel, b.kernel]); bnKeys(s, `${p}.bn1`, b.cin);
      s.set(`${p}.se.conv_reduce.weight`, [b.se, b.cin, 1, 1]); s.set(`${p}.se.conv_reduce.bias`, [b.se]);
      s.set(`${p}.se.conv_expand.weight`, [b.cin, b.se, 1, 1]); s.set(`${p}.se.conv_expand.bias`, [b.cin]);
      s.set(`${p}.conv_pw.weight`, [b.cout, b.cin, 1, 1]); bnKeys(s, `${p}.bn2`, b.cout);
    } else {
      s.set(`${p}.conv_pw.weight`, [b.mid, b.cin, 1, 1]); bnKeys(s, `${p}.bn1`, b.mid);
      s.set(`${p}.conv_dw.weight`, [b.mid, 1, b.kernel, b.kernel]); bnKeys(s, `${p}.bn2`, b.mid);
      s.set(`${p}.se.conv_reduce.weight`, [b.se, b.mid, 1, 1]); s.set(`${p}.se.conv_reduce.bias`, [b.se]);
      s.set(`${p}.se.conv_expand.weight`, [b.mid, b.se, 1, 1]); s.set(`${p}.se.conv_expand.bias`, [b.mid]);
      s.set(`${p}.conv_pwl.weight`, [b.cout, b.mid, 1, 1]); bnKeys(s, `${p}.bn3`, b.cout);
    }
  }));
  const last = plan.stages[plan.stages.length - 1];
  const cin = last?.[last.length - 1]?.cout ?? plan.stem;
  s.set("conv_head.weight", [plan.head, cin, 1, 1]); bnKeys(s, "bn2", plan.head);
  s.set("classifier.weight", [numClasses, plan.head]); s.set("classifier.bias", [numClasses]);
  return s;
}

/** timm `resnet*` 의 열쇠와 모양. */
function resnetKeys(plan: ResNetPlan, numClasses: number): Shapes {
  const s: Shapes = new Map();
  s.set("conv1.weight", [plan.stem, 3, 7, 7]); bnKeys(s, "bn1", plan.stem);
  plan.layers.forEach((blocks, li) => blocks.forEach((b, bi) => {
    const p = `layer${li + 1}.${bi}`;
    if (plan.block === "basic") {
      s.set(`${p}.conv1.weight`, [b.width, b.cin, 3, 3]); bnKeys(s, `${p}.bn1`, b.width);
      s.set(`${p}.conv2.weight`, [b.cout, b.width, 3, 3]); bnKeys(s, `${p}.bn2`, b.cout);
    } else {
      s.set(`${p}.conv1.weight`, [b.width, b.cin, 1, 1]); bnKeys(s, `${p}.bn1`, b.width);
      s.set(`${p}.conv2.weight`, [b.width, b.width, 3, 3]); bnKeys(s, `${p}.bn2`, b.width);
      s.set(`${p}.conv3.weight`, [b.cout, b.width, 1, 1]); bnKeys(s, `${p}.bn3`, b.cout);
    }
    if (b.downsample) { s.set(`${p}.downsample.0.weight`, [b.downsample.cout, b.downsample.cin, 1, 1]); bnKeys(s, `${p}.downsample.1`, b.downsample.cout); }
  }));
  s.set("fc.weight", [numClasses, plan.fcIn]); s.set("fc.bias", [numClasses]);
  return s;
}

let seed = 12345;
function rnd(): number { seed ^= seed << 13; seed >>>= 0; seed ^= seed >>> 17; seed ^= seed << 5; seed >>>= 0; return seed / 4294967296 - 0.5; }

/** 모양대로 난수를 채운 체크포인트 흉내. `running_var` 는 양수로. 읽힌 열쇠를 기록한다. */
function fakeCheckpoint(shapes: Shapes): { st: cpu.HostStateDict; read: Set<string> } {
  const read = new Set<string>();
  const tensors = new Map<string, cpu.HostTensor>();
  for (const [key, shape] of shapes) {
    const n = shape.reduce((a, d) => a * d, 1);
    const scale = key.endsWith("running_var") ? 0 : 0.2;
    const data = Float32Array.from({ length: n }, () => (key.endsWith("running_var") ? 1 + Math.abs(rnd()) : rnd() * scale));
    tensors.set(key, { shape, data });
  }
  const recording = {
    get: (key: string) => { read.add(key); return tensors.get(key); },
    keys: () => tensors.keys(),
  };
  // HostStateDict 가 요구하는 것은 ReadonlyMap 의 모양 — 읽기 기록기를 그 위에 얹는다.
  const st: cpu.HostStateDict = { tensors: Object.assign(new Map(tensors), recording), metadata: {} };
  return { st, read };
}

const CASES: readonly { name: string; shapes: Shapes; build: (st: cpu.HostStateDict, features: boolean) => cpu.CpuGraph; features: number }[] = [
  { name: "efficientnet_b0", shapes: efficientnetKeys(efficientnetPlan(1, 1), 10), build: (st, f) => efficientnetCpuGraph(efficientnetPlan(1, 1), st, { numClasses: 10, features: f }), features: 1280 },
  { name: "efficientnet_b2", shapes: efficientnetKeys(efficientnetPlan(1.1, 1.2), 10), build: (st, f) => efficientnetCpuGraph(efficientnetPlan(1.1, 1.2), st, { numClasses: 10, features: f }), features: 1408 },
  { name: "resnet18", shapes: resnetKeys(resnetPlan("resnet18"), 10), build: (st, f) => resnetCpuGraph(resnetPlan("resnet18"), st, { numClasses: 10, features: f }), features: 512 },
  { name: "resnet50", shapes: resnetKeys(resnetPlan("resnet50"), 10), build: (st, f) => resnetCpuGraph(resnetPlan("resnet50"), st, { numClasses: 10, features: f }), features: 2048 },
];

for (const c of CASES) {
  test(`${c.name}: 변환이 체크포인트의 열쇠를 남김없이, 빠짐없이 읽는다`, () => {
    const { st, read } = fakeCheckpoint(c.shapes);
    c.build(st, false);
    const unread = [...c.shapes.keys()].filter((k) => !read.has(k));
    assert.deepEqual(unread, [], `변환이 읽지 않은 열쇠 — 무엇인가를 빼먹었다: ${unread.slice(0, 6).join(", ")}`);
    const unknown = [...read].filter((k) => !c.shapes.has(k));
    assert.deepEqual(unknown, [], `변환이 체크포인트에 없는 열쇠를 찾았다: ${unknown.slice(0, 6).join(", ")}`);
  });

  test(`${c.name}: 그래프가 cpu 장치에서 실제로 돈다 — 로짓과 특징 둘 다`, async () => {
    const K = await cpu.loadKernels();
    const { st } = fakeCheckpoint(c.shapes);
    const B = 2, S = 64;
    const input = Float32Array.from({ length: B * 3 * S * S }, () => rnd() * 2);
    const logits = new cpu.CpuRunner(K, c.build(st, false)).forward(input, B, S, S);
    assert.equal(logits.length, B * 10);
    assert.ok(logits.every(Number.isFinite), "로짓에 NaN 또는 Inf");
    const feats = new cpu.CpuRunner(K, c.build(st, true)).forward(input, B, S, S);
    assert.equal(feats.length, B * c.features, `특징 폭은 plan 의 head/fcIn 인 ${c.features}`);
  });
}

test("없는 열쇠는 그 자리의 이웃 열쇠를 들고 실패한다 — 조용히 넘어가지 않는다", () => {
  const shapes = efficientnetKeys(efficientnetPlan(1, 1), 10);
  shapes.delete("blocks.1.0.conv_pwl.weight");
  const { st } = fakeCheckpoint(shapes);
  assert.throws(() => efficientnetCpuGraph(efficientnetPlan(1, 1), st, { numClasses: 10 }), (e: unknown) =>
    e instanceof BimmError && e.message.includes("blocks.1.0.conv_pwl.weight") && e.message.includes("blocks.1.0"));
});

test("cpuGraphFor 가 받는 이름은 전부 카탈로그에 있고, 카탈로그 밖 이름은 이름으로 거절한다", () => {
  const catalogue = new Set(listModels().map((f) => `${f.library}/${f.factory}`));
  // b7 은 카탈로그에 아직 없다(README 가 그 까닭을 적는다) — SCALES 에는 있어 그래프는 지을 수 있다.
  const outside = CPU_FACTORIES.filter((f) => !catalogue.has(`${f.library}/${f.factory}`)).map((f) => f.factory);
  assert.deepEqual(outside, ["efficientnet_b7"]);
  const { st } = fakeCheckpoint(resnetKeys(resnetPlan("resnet18"), 5));
  const g = cpuGraphFor({ library: "timm", factory: "resnet18" }, st, { numClasses: 5 });
  assert.equal(g.outputChannels, 5);
  assert.throws(() => cpuGraphFor({ library: "timm", factory: "mobilenetv2_100" }, st, { numClasses: 5 }), (e: unknown) =>
    e instanceof BimmError && e.message.includes("mobilenetv2_100") && e.message.includes("timm/resnet18"));
  assert.throws(() => cpuGraphFor({ library: "borchvision", factory: "resnet18_cifar" }, st, { numClasses: 5 }), BimmError);
});
