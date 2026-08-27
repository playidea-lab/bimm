/**
 * **선언한 범위가 코어의 새 마이너를 막지 않는지** 본다.
 *
 * ## 왜 이런 검사가 있나
 *
 * `bimm-ts@0.2.0` 이 peer 를 `^0.1.0` 으로 적고 나갔다. **0.x 에서 `^` 는 마이너를
 * 메이저처럼 다룬다** — `>=0.1.0 <0.2.0` 이다. 코어가 0.2.0 을 내자 그 둘을 같이
 * 요구하는 설치가 통째로 거부됐다:
 *
 *     Conflicting peer dependency: borch-ts@0.1.1
 *       peer borch-ts@"^0.1.0" from bimm-ts@0.2.0
 *
 * **이쪽에서는 안 보이는 실패다.** 컴파일도 되고 검사도 통과한다 — 설치하는 사람의
 * 자리에서만 드러난다. 이웃 저장소가 같은 사고를 먼저 냈고 같은 검사를 세웠는데,
 * 그 규칙이 여기에는 없어서 두 번째로 냈다.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

/** `dist/test` 에서 두 칸 위가 저장소 뿌리다. */
const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

interface Manifest {
  readonly peerDependencies: Readonly<Record<string, string>>;
  readonly devDependencies: Readonly<Record<string, string>>;
}

function manifest(): Manifest {
  return JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8")) as Manifest;
}

test("peer 범위가 0.x 의 새 마이너를 배제하지 않는다", () => {
  for (const [name, range] of Object.entries(manifest().peerDependencies)) {
    // `^0.` 하나가 이 사고의 전부였다. 문자열로 잡는 것이 얕지만, 잡으려는 것이
    // 정확히 그 문자열이다.
    assert.ok(
      !range.startsWith("^0."),
      `${name}: '${range}' — 0.x 에서 ^ 는 마이너를 배제한다. '>=x.y.z <1.0.0' 로 적을 것`,
    );
    assert.match(
      range,
      /<1\.0\.0/,
      `${name}: '${range}' — 0.x 안에서만 열어야 한다. 1.0.0 은 깨지는 판이다`,
    );
  }
});

test("검사가 도는 코어도 같은 규칙을 따른다", () => {
  // dev 가 좁으면 CI 는 옛 코어를 보고 초록을 켠다 — 사용자가 받는 것과 다른 것을
  // 본다는 뜻이고, 그것이 이 사고의 절반이었다.
  const range = manifest().devDependencies["borch-ts"];
  assert.ok(range !== undefined, "borch-ts 가 devDependencies 에 있어야 합니다");
  assert.ok(!range.startsWith("^0."), `borch-ts: '${range}' — 위와 같은 이유`);
  assert.match(range, /<1\.0\.0/, `borch-ts: '${range}' — 0.x 안에서만`);
});

test("판 번호가 적힌 두 곳이 같은 수를 말한다", () => {
  // **`npm ci` 는 이 불일치로 안 멈춘다**(borch 에서 실측). 그래서 CI 는 초록이고,
  // 다음 사람이 `npm install` 을 하는 날 잠금이 조용히 바뀌어 자기 변경과 무관한
  // diff 로 올라온다 — 그 사람은 안 건드린 것을 커밋하거나 되돌리느라 시간을 쓴다.
  //
  // 0.7.0 을 낼 때 실제로 그렇게 됐다. `package.json` 은 올렸고 잠금은 0.6.0 에
  // 남았다. borch 쪽에서 같은 자리를 찾고 나서야 여기도 보게 됐다.
  const read = (name: string): { version?: string; packages?: Record<string, { version?: string }> } =>
    JSON.parse(readFileSync(join(ROOT, name), "utf8")) as never;
  const declared = read("package.json").version;
  const lock = read("package-lock.json");

  assert.equal(lock.version, declared,
    `package-lock.json 의 루트는 ${lock.version}, package.json 은 ${declared} — 같이 고쳐라`);
  assert.equal(lock.packages?.[""]?.version, declared,
    `package-lock.json 의 packages[""] 도 같아야 한다`);
});
