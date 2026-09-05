/**
 * 카탈로그의 모델을 **borch-ts 의 `cpu` 장치가 도는 그래프**로 — 어댑터가 없는 기계를 위해.
 *
 * ## 왜 여기인가
 *
 * `cpu` 장치는 `nn.Module` 을 걷지 않는다. 그 장치가 존재하는 기계에는 WebGPU 어댑터가
 * 없고, 그러니 걸을 모듈 인스턴스도 없다. 있는 것은 체크포인트의 바이트와 **네트워크가
 * 어떤 모양인지에 대한 앎**이고, 그 앎은 이 저장소의 plan 표다 — `efficientnetPlan` 과
 * `resnetPlan` 이 `nn.Module` 을 짓는 바로 그 표. 그래서 표에서 그래프를 짓는 함수도
 * 표 옆에 선다. 코어의 검사 페이지가 한동안 이 표의 사본을 들고 있었고, 사본은 갈리는
 * 모양이라 이리로 옮겼다.
 *
 * 두 길이 같은 것에서 출발한다:
 *
 *     어댑터 있음 → hub.load()          → createModel()  → WebGPU
 *     어댑터 없음 → hub.fetchWeights()  → cpuGraphFor()  → cpu.CpuRunner
 *
 * ## 열쇠 이름은 timm 의 것이다
 *
 * `blocks.2.1.conv_pw.weight`, `layer3.0.downsample.1.running_var` — `nn.Module` 쪽이
 * `Sequential` 을 단계마다 두어 내는 이름과 같은 이름을 여기서 손으로 적는다. 하나라도
 * 틀리면 체크포인트에 없는 열쇠를 찾다가 **크게** 실패하고, 그 자리에 있는 열쇠들을
 * 같이 말한다 — 조용히 넘어가는 실패가 아니다. 검사(`test/cpu.test.ts`)는 반대 방향도
 * 본다: 체크포인트의 열쇠 중 이 변환이 읽지 않은 것이 있으면, 무엇인가를 빼먹은 것이다.
 *
 * ## 여기 없는 것
 *
 * MobileNetV2·V3(hardsigmoid, V3 의 뒤집힌 머리 순서)와 ViT(attention)는 `cpu` 장치에
 * 그 연산이 없어 여기서도 없다. `cpuGraphFor` 는 이름으로 거절한다.
 */

import { cpu } from "borch-ts";

import { efficientnetPlan, SCALES, type Plan as EfficientNetPlan } from "./efficientnet.js";
import { BimmError } from "./errors.js";
import type { FactoryName } from "./registry.js";
import { resnetPlan, RESNETS, type Plan as ResNetPlan } from "./resnet50.js";

export interface CpuGraphOptions {
  /** 분류기의 출력 수. 매니페스트의 `arch.args.numClasses` 가 말한다. */
  readonly numClasses: number;
  /** 참이면 분류기 앞, 전역 평균 풀까지만 — `forwardHead(h, preLogits=true)` 와 같은 자리. */
  readonly features?: boolean;
}

/** 체크포인트에서 열쇠 하나. 없으면 그 근처의 열쇠를 들고 실패한다. */
function need(st: cpu.HostStateDict, key: string): Float32Array {
  const t = st.tensors.get(key);
  if (t) return t.data;
  const stem = key.split(".").slice(0, 2).join(".");
  const near = [...st.tensors.keys()].filter((k) => k.startsWith(stem)).slice(0, 8);
  throw new BimmError(
    `체크포인트에 '${key}' 가 없습니다.\n`
    + (near.length ? `  같은 자리의 열쇠: ${near.join(", ")}` : "  같은 자리에 열쇠가 하나도 없습니다 — 다른 모델의 체크포인트입니다."),
  );
}

function bn(st: cpu.HostStateDict, prefix: string): cpu.BatchNorm {
  return {
    weight: need(st, `${prefix}.weight`),
    bias: need(st, `${prefix}.bias`),
    runningMean: need(st, `${prefix}.running_mean`),
    runningVar: need(st, `${prefix}.running_var`),
  };
}

/**
 * timm 의 `efficientnet_b*` 를 그래프로. `plan` 은 `efficientnetPlan(width, depth)`.
 *
 * 블록의 모양은 `DepthwiseSeparableConv`·`InvertedResidual` 의 `forward` 와 한 줄씩 같다:
 * (넓히고) → depthwise → SE → 좁히고 → 잔차. SE 가 좁히는 폭은 `plan.se` — 블록 **입력**의
 * 4분의 1 이고, 넓힌 채널의 4분의 1 이 아니다. 그 함정은 `efficientnet.ts` 첫 문단에 있다.
 */
export function efficientnetCpuGraph(plan: EfficientNetPlan, st: cpu.HostStateDict, opts: CpuGraphOptions): cpu.CpuGraph {
  const g = new cpu.GraphBuilder();
  const x = g.input(3);
  let h = g.conv(x, { weight: need(st, "conv_stem.weight"), cout: plan.stem, cin: 3, k: 3, stride: 2, pad: 1, bn: bn(st, "bn1"), act: cpu.ACT.swish });
  plan.stages.forEach((blocks, si) => {
    blocks.forEach((b, bi) => {
      const p = `blocks.${si}.${bi}`, pad = (b.kernel - 1) / 2;
      const skip = b.stride === 1 && b.cin === b.cout;
      let out: number;
      if (b.kind === "dw") {
        let d = g.dwconv(h, { weight: need(st, `${p}.conv_dw.weight`), cout: b.cin, cin: b.cin, k: b.kernel, stride: b.stride, pad, bn: bn(st, `${p}.bn1`), act: cpu.ACT.swish });
        d = g.se(d, need(st, `${p}.se.conv_reduce.weight`), need(st, `${p}.se.conv_reduce.bias`), need(st, `${p}.se.conv_expand.weight`), need(st, `${p}.se.conv_expand.bias`), b.se);
        out = g.conv(d, { weight: need(st, `${p}.conv_pw.weight`), cout: b.cout, cin: b.cin, k: 1, stride: 1, pad: 0, bn: bn(st, `${p}.bn2`) });
      } else {
        const e = g.conv(h, { weight: need(st, `${p}.conv_pw.weight`), cout: b.mid, cin: b.cin, k: 1, stride: 1, pad: 0, bn: bn(st, `${p}.bn1`), act: cpu.ACT.swish });
        let d = g.dwconv(e, { weight: need(st, `${p}.conv_dw.weight`), cout: b.mid, cin: b.mid, k: b.kernel, stride: b.stride, pad, bn: bn(st, `${p}.bn2`), act: cpu.ACT.swish });
        d = g.se(d, need(st, `${p}.se.conv_reduce.weight`), need(st, `${p}.se.conv_reduce.bias`), need(st, `${p}.se.conv_expand.weight`), need(st, `${p}.se.conv_expand.bias`), b.se);
        out = g.conv(d, { weight: need(st, `${p}.conv_pwl.weight`), cout: b.cout, cin: b.mid, k: 1, stride: 1, pad: 0, bn: bn(st, `${p}.bn3`) });
      }
      h = skip ? g.add(out, h) : out;
    });
  });
  const last = plan.stages[plan.stages.length - 1];
  const cin = last?.[last.length - 1]?.cout ?? plan.stem;
  h = g.conv(h, { weight: need(st, "conv_head.weight"), cout: plan.head, cin, k: 1, stride: 1, pad: 0, bn: bn(st, "bn2"), act: cpu.ACT.swish });
  const pooled = g.gap(h);
  if (opts.features) return g.finish(pooled);
  return g.finish(g.linear(pooled, need(st, "classifier.weight"), need(st, "classifier.bias"), opts.numClasses));
}

/**
 * timm 의 ImageNet ResNet 을 그래프로. `plan` 은 `resnetPlan(name)`.
 *
 * `plan.block` 이 `basic` 이면 3×3 둘, `bottleneck` 이면 1×1 → 3×3 → 1×1 — 둘 다
 * `resnet50.ts` 의 `forward` 순서 그대로이고, `downsample` 은 plan 이 `null` 이 아닌
 * 첫 블록에만 있다.
 */
export function resnetCpuGraph(plan: ResNetPlan, st: cpu.HostStateDict, opts: CpuGraphOptions): cpu.CpuGraph {
  const g = new cpu.GraphBuilder();
  const x = g.input(3);
  let h = g.conv(x, { weight: need(st, "conv1.weight"), cout: plan.stem, cin: 3, k: 7, stride: 2, pad: 3, bn: bn(st, "bn1"), act: cpu.ACT.relu });
  h = g.maxpool(h, 3, 2, 1);
  plan.layers.forEach((blocks, li) => {
    blocks.forEach((b, bi) => {
      const p = `layer${li + 1}.${bi}`;
      let main: number;
      if (plan.block === "basic") {
        const a = g.conv(h, { weight: need(st, `${p}.conv1.weight`), cout: b.width, cin: b.cin, k: 3, stride: b.stride, pad: 1, bn: bn(st, `${p}.bn1`), act: cpu.ACT.relu });
        main = g.conv(a, { weight: need(st, `${p}.conv2.weight`), cout: b.cout, cin: b.width, k: 3, stride: 1, pad: 1, bn: bn(st, `${p}.bn2`) });
      } else {
        const a = g.conv(h, { weight: need(st, `${p}.conv1.weight`), cout: b.width, cin: b.cin, k: 1, stride: 1, pad: 0, bn: bn(st, `${p}.bn1`), act: cpu.ACT.relu });
        const c = g.conv(a, { weight: need(st, `${p}.conv2.weight`), cout: b.width, cin: b.width, k: 3, stride: b.stride, pad: 1, bn: bn(st, `${p}.bn2`), act: cpu.ACT.relu });
        main = g.conv(c, { weight: need(st, `${p}.conv3.weight`), cout: b.cout, cin: b.width, k: 1, stride: 1, pad: 0, bn: bn(st, `${p}.bn3`) });
      }
      const shortcut = b.downsample === null ? h
        : g.conv(h, { weight: need(st, `${p}.downsample.0.weight`), cout: b.downsample.cout, cin: b.downsample.cin, k: 1, stride: b.downsample.stride, pad: 0, bn: bn(st, `${p}.downsample.1`) });
      h = g.add(main, shortcut, cpu.ACT.relu);
    });
  });
  const pooled = g.gap(h);
  if (opts.features) return g.finish(pooled);
  return g.finish(g.linear(pooled, need(st, "fc.weight"), need(st, "fc.bias"), opts.numClasses));
}

/** `cpuGraphFor` 가 받는 이름. 카탈로그의 부분집합이고, 검사가 그 포함을 지킨다. */
export const CPU_FACTORIES: readonly FactoryName[] = [
  ...Object.keys(SCALES).map((factory) => ({ library: "timm", factory })),
  ...RESNETS.map((factory) => ({ library: "timm", factory })),
];

/**
 * 카탈로그 이름 하나를 그래프로. 매니페스트의 `arch` 가 말하는 것을 그대로 넘긴다.
 *
 * `library` 는 `timm` 이어야 한다 — `borchvision/resnet18_cifar` 는 CIFAR 판이라 스템이
 * 다르고, 여기 아직 없다.
 */
export function cpuGraphFor(name: FactoryName, st: cpu.HostStateDict, opts: CpuGraphOptions): cpu.CpuGraph {
  if (name.library === "timm") {
    const scale = SCALES[name.factory];
    if (scale) return efficientnetCpuGraph(efficientnetPlan(scale[0], scale[1]), st, opts);
    if (RESNETS.includes(name.factory)) return resnetCpuGraph(resnetPlan(name.factory), st, opts);
  }
  throw new BimmError(
    `${name.library}/${name.factory} 는 cpu 그래프로 지을 수 없습니다.\n`
    + `  지을 수 있는 이름: ${CPU_FACTORIES.map((f) => `${f.library}/${f.factory}`).join(", ")}`,
  );
}
