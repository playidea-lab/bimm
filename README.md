# bimm

[`borch`](https://github.com/playidea-lab/borch) 런타임 위의 **모델 아키텍처
카탈로그.**

```
borch        ←→  torch          (npm 에서는 `borch-ts` — 아래를 보라)
borchvision  ←→  torchvision
bimm         ←→  timm
borch-hub    ←→  (아무것도 아니다 — 아래를 보라)
```

`borch` 가 torch 이고 `borchvision` 이 torchvision 인 것과 **같은 뜻으로** 이
패키지는 timm 이다: 이름으로 부를 수 있는 아키텍처 표와, 그것을 실제 층으로 만드는
함수 하나.

## 쓰는 법

```ts
import { init } from "borch-ts";
import { createModel, listModels } from "bimm-ts";

await init();                                   // 층이 곧 텐서다
listModels();                                   // [{ library: "borchvision", factory: "resnet18_cifar" }]
const model = createModel("borchvision", "resnet18_cifar", { numClasses: 10 });
```

**`await init()` 이 먼저다.** 층이 곧 텐서이고 텐서는 WebGPU 어댑터 위에 선다. 안
부르고 오면 코어가 그 자리에서 멈추는데, 그 진단을 가로채 우리 말로 바꾸지 않는다 —
원인은 코어 쪽이고 코어의 문구가 더 정확하다.

## 이름을 둘로 받는다 — timm 과 갈리는 유일한 자리

timm 은 `create_model("resnet18")` 처럼 이름 하나를 받는다. 여기는 둘을 받는다.

흉내를 깨는 자리이므로 까닭을 적어 둔다. **실제 생태계에서 torchvision 의
`resnet18` 과 timm 의 `resnet18` 은 다른 모델이고 가중치가 안 호환된다.** timm 은
이름공간이 없어서 그 충돌을 문서로만 다룬다. 우리 카탈로그는 처음부터 두 라이브러리를
동시에 들도록 만들어졌고, 이름공간 없이 시작하면 **이미 배포된 매니페스트**가 그
이름을 박은 뒤에는 못 고친다.

즉 timm 을 덜 흉내 낸 것이 아니라, timm 이 나중에 아쉬워한 자리를 먼저 잡은 것이다.

## 패키지가 어디 있는지와 `library` 이름은 별개다

이 파일이 `bimm` 안에 있다고 해서 여기 실린 것이 전부 `bimm` 의 모델은 아니다.
`library` 는 **아키텍처의 출신**을 가리키는 규약이고, 코드가 사는 곳과 상관없다.
그래서 `bimm` 이 `borchvision/resnet18_cifar` 를 들고 있는 것은 모순이 아니라 설계다.

**이름은 늘릴 수 있지만 지우거나 뜻을 바꿀 수 없다.** 그 이름을 적어둔 매니페스트가
이미 남의 페이지에서 돌고 있다.

## 인자에 기본값이 없다

빠뜨리면 거절한다. 기본값으로 메우면 나중에 그 값을 바꾸는 순간 이미 배포된
매니페스트가 **다른 모델**을 만들고, 가중치는 모양이 맞으니 실린 다음 틀린 수를 낸다.
모르는 인자도 거절한다 — `numClases` 라고 적어도 조용히 통과하면 올린 사람은 자기가
무엇을 올렸는지 모른다.

## 여기 없는 것: 가중치를 받아오는 일

timm 의 `create_model(pretrained=True)` 에 해당하는 일 — 매니페스트를 읽고, 해시를
대조하고, 이 브라우저가 그 모델을 돌릴 수 있는지 **먼저** 판정하는 것 — 은
[`borch-hub`](https://github.com/playidea-lab/borch-hub) 에 있다.

갈라 둔 이유는 **의존이 한 방향으로만 흐르게** 하기 위해서다. 허브는 카탈로그를
알아야 하지만 카탈로그는 매니페스트를 몰라도 된다. 모델 하나 만들려는 사람이
배포·검증 계층을 통째로 끌어오지 않는다.

그 방향은 실제 생태계와도 같다. timm 은 아키텍처를 알고, 어디서 받아오는지는 그
바깥의 일이다.

## npm 에서는 둘 다 `-ts` 가 붙는다

```
npm i bimm-ts borch-ts
```

저장소 이름은 `bimm` 이고 코어는 `borch` 인데, 설치하는 이름은 둘 다 뒤에 `-ts` 가
붙는다. 갈린 것이 아니라 **그 두 이름을 못 쓴다** — npm 레지스트리의 유사성 필터가
막는다. 코어가 먼저 겪었고, 이 패키지도 게시하려다 같은 자리에서 403 을 받았다:

```
Package name too similar to existing packages bigi,bili,boom,jimp,mime,viem
```

네 글자 이름이 이미 붐비는 곳이라 그렇다. 스코프(`@playidealab/bimm`)를 쓰면 필터를
아예 우회하지만, 그러면 코어와 설치 줄의 모양이 갈린다. **같은 규칙으로 읽히는 편이
낫다고 보고 코어가 쓴 해법을 그대로 따랐다.**

문서와 대화에서 부르는 이름은 계속 `bimm` 이다 — 설치하는 이름만 다르다.

## 코어는 peerDependency 다

이 패키지가 코어를 끌고 오면 사용자의 것과 두 벌이 된다. 텐서 두 벌은 같은 GPU
장치를 공유하지 않아서 **예외 없이 조용히 안 맞는다.** 쓰는 쪽이 이미 깔아둔 것을 쓴다.

`optional` peer 가 아닌 것은 `borch-ts` 0.1.0 이 실제로 나가 있기 때문이다. 없는
패키지를 필수 peer 로 두면 npm 이 lockfile 조차 못 만들어서 한동안 optional 이었는데,
그 조건이 아니게 됐다.

## 지금 카탈로그에 있는 것

`borchvision/resnet18_cifar` 하나다. **빈 표에 자리만 잡아두지 않은 것은 일부러다** —
코어 저장소가 여러 번 적어둔 대로 사용자 없는 표면은 케이스가 안 생기고, 케이스 없는
표면이 조용히 틀린다. 이름은 첫 모델을 끝까지 통과시키면서 같이 나온다.
