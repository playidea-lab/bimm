/**
 * `bimm` 이 던지는 오류 하나.
 *
 * **`BorchHubError` 를 쓰지 않는다.** 카탈로그가 허브에서 갈라져 나온 이유가 정확히
 * 이것이다 — 매니페스트를 모르는 쪽이 매니페스트 패키지의 오류 형을 들고 있으면,
 * 모델을 만들고 싶을 뿐인 사람이 배포·검증 계층을 통째로 끌어오게 된다.
 *
 * 두 이름이 갈리는 값은 잡는 쪽에 있다. `catch (e) { if (e instanceof BimmError) }`
 * 가 "카탈로그가 거절했다" 를 뜻하고, 그것은 "매니페스트가 틀렸다" 와 다른 사건이다.
 */
export class BimmError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BimmError";
  }
}
