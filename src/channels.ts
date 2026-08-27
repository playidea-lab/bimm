/**
 * 채널 수를 8 의 배수로 맞추는 규칙 — timm 의 `make_divisible` 과 `round_channels`.
 *
 * **여기에는 `borch` 임포트가 없다.** `args.ts` 와 같은 이유다 — 이것은 수를 세는
 * 일이지 텐서를 만드는 일이 아니므로, GPU 없는 자리에서도 검사할 수 있어야 한다.
 * 이 파일이 갈라져 나온 첫 값어치가 그것이다: 모델 코드에 붙은 검사 중 CI 에서
 * 실제로 도는 것이 여기서 시작한다.
 *
 * ## 왜 한 곳이어야 하는가
 *
 * 전에는 같은 세 줄이 두 곳에 있었다 — `efficientnet.ts` 의 `rounded` 와
 * `mobilenetv3.ts` 의 `reduced`. 이름도 설명도 달랐지만 계산은 글자까지 같았다.
 *
 * 이 규칙이 어긋나면 그 블록만 채널이 달라지고, **가중치는 모양이 안 맞아 실리지
 * 않는다.** 실리지 않는 것은 그나마 낫다 — 어긋난 채로 모양이 우연히 맞으면 조용히
 * 틀린 수가 나온다. 갈리면 안 되는 규칙이 두 벌인 것은 그래서 미룰 일이 아니었다.
 */

/**
 * `v` 를 `divisor` 의 배수로 맞춘다.
 *
 * 반올림이 원래 값의 `roundLimit` 아래로 떨어지면 한 칸 올린다 — 내림이 10% 를 넘지
 * 않게 하는 timm 의 규칙이다. `72 * 0.25 = 18` 은 16 으로 내려가는데 16 은 18 의
 * 90%(16.2) 보다 작으므로 24 가 된다. 이 한 줄이 없으면 MobileNetV3 의 여러 블록이
 * 어긋난다.
 *
 * 인자 이름과 기본값은 timm 의 것을 그대로 쓴다 — 옮겨온 규칙이므로 옮겨온 자리의
 * 이름으로 부르는 편이, 나중에 둘을 나란히 놓고 볼 사람에게 낫다.
 */
export function makeDivisible(
  v: number, divisor = 8, minValue?: number, roundLimit = 0.9,
): number {
  const floor = minValue ?? divisor;
  let out = Math.max(floor, Math.floor(Math.floor(v + divisor / 2) / divisor) * divisor);
  if (out < roundLimit * v) out += divisor;
  return out;
}

/**
 * 배율을 곱한 뒤 배수로 맞춘다 — timm 의 `round_channels`.
 *
 * **곱셈이 먼저다.** 배수로 맞춘 뒤 곱하면 다른 수가 나오고, 그 차이는 판이 커질수록
 * 벌어진다.
 *
 * ## 배율 1 도 그냥 지나가지 않는다
 *
 * timm 이 원래 값을 그대로 돌려주는 것은 배율이 **0** 일 때뿐이고, 1 일 때는
 * `make_divisible` 을 통과한다. 그래서 `round_channels(17, 1.0)` 은 17 이 아니라
 * 16 이다.
 *
 * 전에 이 자리에 `if (width === 1) return channels` 가 있었다. 지금 표의 채널이
 * **전부 8 의 배수라서** 결과가 같았을 뿐이고, 8 의 배수가 아닌 수가 표에 들어오는
 * 날 조용히 갈렸을 것이다. 규칙을 옮길 때는 지금 맞는 것보다 **원본과 같은 것**이
 * 안전하다.
 */
export function roundChannels(
  channels: number, multiplier = 1.0, divisor = 8,
  channelMin?: number, roundLimit = 0.9,
): number {
  if (!multiplier) return channels;
  return makeDivisible(channels * multiplier, divisor, channelMin, roundLimit);
}
