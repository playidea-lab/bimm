/**
 * 팩토리 인자를 검사한다. **여기에는 `borch` 임포트가 없다** — 일부러다.
 *
 * 모델을 하나라도 만들려면 WebGPU 어댑터가 필요하고, 그건 브라우저에서만 잡힌다.
 * 검사까지 그 안에 묶여 있으면 "매니페스트의 args 가 틀렸다"는 것을 확인하는 데
 * GPU 가 있는 기계가 필요해진다. 값을 보는 일과 물건을 만드는 일을 갈라 두면
 * 앞쪽은 어디서나, 특히 레지스트리 CI 에서 돈다.
 */

import { BimmError } from "./errors.js";

/** 지금 인자는 전부 정수다. 실수나 문자열이 필요해지면 그때 늘린다. */
export interface ArgSpec {
  readonly kind: "int";
  readonly min: number;
}

export type FactoryArgs = Readonly<Record<string, ArgSpec>>;

/**
 * 매니페스트가 준 `arch.args` 를 팩토리의 규격에 맞춰 좁힌다.
 *
 * ## 왜 기본값을 두지 않는가
 *
 * 빠진 인자를 기본값으로 메우면, 나중에 그 기본값을 바꾸는 순간 **이미 배포된
 * 매니페스트가 다른 모델을 만든다.** 가중치는 모양이 맞으니 실리고, 실린 뒤에
 * 틀린 수를 낸다. 그래서 전부 필수다 — 매니페스트가 길어지는 편이 낫다.
 *
 * ## 왜 모르는 인자를 거절하는가
 *
 * `numClases` 라고 적어도 조용히 무시하면 기본값 모델이 만들어지고, 올린 사람은
 * 자기가 무엇을 올렸는지 모른 채 통과한다. 오타는 병합 전에 잡혀야 한다.
 */
export function checkArgs(
  factory: string,
  spec: FactoryArgs,
  given: Readonly<Record<string, unknown>>,
): Record<string, number> {
  const known = Object.keys(spec);

  const extra = Object.keys(given).filter((k) => !known.includes(k));
  if (extra.length > 0) {
    throw new BimmError(
      `${factory} 가 모르는 인자입니다: ${extra.sort().join(", ")}\n` +
        `  받는 인자: ${known.sort().join(", ")}`,
    );
  }

  const out: Record<string, number> = {};
  for (const key of known) {
    const argSpec = spec[key];
    if (argSpec === undefined) continue;
    const value = given[key];
    if (value === undefined) {
      throw new BimmError(
        `${factory} 에 ${key} 가 없습니다 — 이 인자는 기본값이 없습니다.\n` +
          "  기본값으로 메우면 그 값을 바꾸는 날 이미 배포된 매니페스트가 다른 모델이 됩니다.",
      );
    }
    if (typeof value !== "number" || !Number.isInteger(value)) {
      throw new BimmError(`${factory}.${key} 는 정수여야 합니다 — ${String(value)} 를 받았습니다`);
    }
    if (value < argSpec.min) {
      throw new BimmError(`${factory}.${key} 는 ${argSpec.min} 이상이어야 합니다 — ${value}`);
    }
    out[key] = value;
  }
  return out;
}
