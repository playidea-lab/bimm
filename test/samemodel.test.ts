/**
 * 코어의 벤치가 든 ResNet 과 **여기 실린 ResNet 이 같은 모델인지** 본다.
 *
 * ## 왜 두 벌인가, 그리고 왜 합치지 않는가
 *
 * `borch/borch-ts/test/bench.ts` 에 같은 모델이 있다. 합칠 수가 없다 — 코어가 이
 * 패키지를 의존하면 순환이 되기 때문이다(이쪽이 코어를 peer 로 잡는다). 갈리는 것을
 * 막을 수 없으면 **갈린 것을 잡는다.** 그 자리가 여기다.
 *
 * ## 왜 `stateDict` 가 아니라 소스를 보는가
 *
 * 열쇠와 모양을 실제로 뽑으려면 모델을 만들어야 하고, 층이 곧 텐서이므로 만드는 데
 * WebGPU 어댑터가 든다 — `new nn.Conv2d(...)` 는 `init()` 없이 그 자리에서 멈춘다.
 * 즉 진짜 `stateDict` 대조는 브라우저 하네스가 있어야 하고, 그것은 코어에만 있다.
 *
 * 소스를 보는 대조는 그보다 얕지만 **비어 있는 것보다 얕지 않다.** 실제로 두 벌이
 * 갈린 첫 사건은 코어의 `Conv2d` 가 `dilation` 과 `groups` 를 `bias` 앞에 들이면서
 * 이쪽 호출 넷이 여섯 인자에 남은 것이었고, 그것은 정확히 이 층에서 보인다.
 *
 * ## 이름이 다른 것은 이름만 다르다
 *
 * 코어는 `Block`·`ResNet18`·`classes`, 여기는 `BasicBlock`·`ResNet18Cifar`·
 * `numClasses` 다. 이름은 두 저장소의 사정이고 모델이 아니므로 대조 전에 맞춰 준다.
 * 반대로 **수는 맞춰 주지 않는다** — 스템이 3 에서 7 로 바뀌면 그것은 다른 모델이고
 * 여기서 걸려야 한다.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
/** `dist/test` → `dist` → 저장소 뿌리. 검사는 빌드된 자리에서 돈다. */
const ROOT = join(HERE, "..", "..");
const OURS = join(ROOT, "src", "resnet.ts");
/**
 * 코어를 **옆에 나란히** 받아둔 것으로 본다.
 *
 * ## 벤치를 두 자리에서 찾는다
 *
 * **설치된 `borch-ts` 안**을 먼저 본다. 벤치는 코어의 진입점이 아니지만(`exports` 에
 * 없어 아무도 임포트할 수 없다) 배포에는 실리므로, 경로로 읽는 이 검사에는 그것으로
 * 충분하다 — 그러면 `npm ci` 한 번으로 CI 에서도 대조가 돈다.
 *
 * 없으면 **옆에 나란히 받아둔 코어**로 물러난다. 코어 0.1.0 에는 벤치가 아직 안 실려
 * 있어서 지금은 대개 이쪽이 쓰인다. 그 판이 나가면 이 갈래는 저절로 앞의 것으로 옮겨
 * 간다 — 무엇을 봤는지는 실패했을 때 경로로 말한다.
 *
 * 두 자리는 서로 다른 판을 가리킬 수 있다. 설치본은 **우리가 지금 서 있는 판**이고
 * 옆 저장소는 **코어의 최신 판**이라, 뒤쪽에서 갈렸다고 나오면 "오늘 깨졌다"가 아니라
 * "다음 코어 릴리스에서 깨진다"는 뜻이다. 둘 다 값이 있고, 앞의 것이 더 정확하다.
 *
 * 어느 자리에도 없으면 조용히 건너뛰지 않고 그 사실을 말한다: 건너뛴 대조는 통과한
 * 대조와 초록색이 같아서, 갈린 것을 잡으라고 둔 검사가 아무것도 안 보게 된다.
 */
const CORE_CANDIDATES = [
  join(ROOT, "node_modules", "borch-ts", "borch-ts", "test", "bench.ts"),
  join(ROOT, "..", "borch", "borch-ts", "test", "bench.ts"),
] as const;

function readCore(): string {
  for (const path of CORE_CANDIDATES) {
    try {
      return readFileSync(path, "utf8");
    } catch {
      // 다음 자리를 본다. 어느 자리에도 없을 때만 말한다.
    }
  }
  throw new Error(
    "대조할 코어의 벤치를 못 찾았습니다. 본 자리:\n" +
      CORE_CANDIDATES.map((p) => `  ${p}`).join("\n") +
      "\n  설치된 borch-ts 에 벤치가 실리기 전 판이라면, 코어를 이 저장소 옆에 받아 두어야 합니다.",
  );
}

function read(path: string): string {
  try {
    return readFileSync(path, "utf8");
  } catch {
    throw new Error(`대조할 소스를 못 읽었습니다: ${path}`);
  }
}

/** 주석은 대조 대상이 아니다. 두 저장소가 서로 다른 말로 같은 모델을 적는다. */
function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
}

/** 이름만 다른 것들을 코어 쪽 이름으로 맞춘다. 수는 건드리지 않는다. */
function rename(text: string): string {
  return text
    .replace(/\bBasicBlock\b/g, "Block")
    .replace(/\bResNet18Cifar\b/g, "ResNet18")
    .replace(/\bnumClasses\b/g, "classes")
    .replace(/\bSTEM_CHANNELS\s*\*\s*8\b/g, "512")
    .replace(/\bSTEM_CHANNELS\b/g, "64")
    .replace(/\bFINAL_CHANNELS\b/g, "512");
}

/** 중괄호를 세어 클래스 본문만 떼어 낸다. */
function classBody(source: string, name: string): string {
  const at = source.search(new RegExp(`class\\s+${name}\\b`));
  assert.ok(at >= 0, `${name} 을 소스에서 못 찾았습니다`);
  const open = source.indexOf("{", at);
  let depth = 0;
  for (let i = open; i < source.length; i += 1) {
    if (source[i] === "{") depth += 1;
    else if (source[i] === "}") {
      depth -= 1;
      if (depth === 0) return source.slice(open + 1, i);
    }
  }
  throw new Error(`${name} 의 본문이 안 닫혔습니다`);
}

/**
 * 필드 선언을 **적힌 순서대로** 뽑는다.
 *
 * `stateDict` 의 열쇠가 여기서 나온다. 층 구성이 똑같아도 `stem` 을 `head` 로 고치면
 * **이미 배포된 가중치가 안 실린다** — 아래 `layers` 는 `new nn.Conv2d(...)` 호출만
 * 보므로 그 갈림을 못 본다. 열쇠를 실제로 뽑으려면 모델을 세워야 하고 그러려면
 * WebGPU 어댑터가 드는데, 열쇠가 어디서 오는지는 어댑터 없이도 안다.
 */
function fields(body: string): readonly string[] {
  const found = [];
  const decl = /(?:private|protected|public)\s+(?:readonly\s+)?(\w+)\s*:\s*([^;]+);/g;
  let hit = decl.exec(body);
  while (hit !== null) {
    found.push(`${hit[1] ?? ""}: ${(hit[2] ?? "").replace(/\s+/g, " ").trim()}`);
    hit = decl.exec(body);
  }
  return found;
}

/**
 * 만들어지는 층을 **나오는 순서대로** 뽑는다.
 *
 * `nn.Sequential([...])` 은 인자에 다시 `new` 가 들어 있어 여기서 앞이 잘리는데,
 * 잘린 조각도 양쪽에서 같은 방식으로 잘리고 그 안의 블록들은 바로 다음 항목으로
 * 이어 나온다. 그래서 대조는 성립한다.
 */
function layers(body: string): readonly string[] {
  const found: string[] = [];
  const call = /new\s+(?:nn\.)?(\w+)\(([^)]*)\)/g;
  let hit: RegExpExecArray | null = call.exec(body);
  while (hit !== null) {
    found.push(`${hit[1] ?? ""}(${(hit[2] ?? "").replace(/\s+/g, "")})`);
    hit = call.exec(body);
  }
  return found;
}

/** `forward` 본문. 공백은 두 저장소의 줄바꿈 취향이므로 지운다. */
function forwardBody(body: string): string {
  const at = body.indexOf("forward(");
  assert.ok(at >= 0, "forward 를 못 찾았습니다");
  const open = body.indexOf("{", at);
  let depth = 0;
  for (let i = open; i < body.length; i += 1) {
    if (body[i] === "{") depth += 1;
    else if (body[i] === "}") {
      depth -= 1;
      if (depth === 0) return body.slice(open + 1, i).replace(/\s+/g, "");
    }
  }
  throw new Error("forward 본문이 안 닫혔습니다");
}

const ours = rename(stripComments(read(OURS)));
const core = stripComments(readCore());

test("블록이 만드는 층이 코어의 것과 같다", () => {
  // 코어의 Conv2d 가 dilation·groups 를 bias 앞에 들였을 때 갈린 자리가 여기다.
  assert.deepEqual(
    layers(classBody(ours, "Block")),
    layers(classBody(core, "Block")),
  );
});

test("ResNet-18 이 만드는 층이 코어의 것과 같다", () => {
  assert.deepEqual(
    layers(classBody(ours, "ResNet18")),
    layers(classBody(core, "ResNet18")),
  );
});

test("블록의 필드 이름과 형이 코어의 것과 같다", () => {
  // stateDict 열쇠가 이 이름에서 나온다. 여기서 갈리면 가중치가 안 실린다.
  assert.deepEqual(
    fields(classBody(ours, "Block")),
    fields(classBody(core, "Block")),
  );
});

test("ResNet-18 의 필드 이름과 형이 코어의 것과 같다", () => {
  assert.deepEqual(
    fields(classBody(ours, "ResNet18")),
    fields(classBody(core, "ResNet18")),
  );
});

test("블록의 forward 가 코어의 것과 같다", () => {
  // 층이 같아도 이어붙이는 순서가 다르면 다른 모델이고, 손실은 그래도 내려간다.
  assert.equal(
    forwardBody(classBody(ours, "Block")),
    forwardBody(classBody(core, "Block")),
  );
});

test("ResNet-18 의 forward 가 코어의 것과 같다", () => {
  assert.equal(
    forwardBody(classBody(ours, "ResNet18")),
    forwardBody(classBody(core, "ResNet18")),
  );
});
