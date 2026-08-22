/**
 * ResNet — CIFAR 판.
 *
 * ## 왜 코어가 아니라 여기 있나
 *
 * 모델 구조는 **유통되는 물건**이지 런타임의 일부가 아니다. 코어가 싣는 ES 모듈은
 * 압축 전 770KB 이고 그 수를 붙잡는 검사가 있는데, 모델이 늘 때마다 그 수가 오르면
 * 안 된다. ResNet 을 안 쓰는 사람이 ResNet 을 받을 이유도 없다.
 *
 * ## 코어의 벤치에 같은 모델이 있다
 *
 * `borch/borch-ts/test/bench.ts` 다. 두 벌이고, 두 벌은 갈린다 — 그런데 합칠 수가
 * 없다. 코어가 이 패키지를 의존하면 순환이 되기 때문이다(이쪽이 코어를 peer 로
 * 잡는다). 그래서 합치는 대신 **대조한다**: 두 모델의 `stateDict` 열쇠와 모양이
 * 같은지 보는 검사가 붙는다. 갈리는 것을 막을 수 없으면 갈린 것을 잡는다.
 *
 * ## 이름이 CIFAR 판임을 말한다

 * `resnet18` 이라고 부르면 ImageNet 판이 올 자리를 미리 먹는다. 둘은 스템부터
 * 다르고 가중치가 안 호환되므로, 같은 이름 아래 인자로 가르면 옛 매니페스트가
 * 어느 날 다른 모델을 만든다.

 *
 * ## 왜 CIFAR 판인가
 *
 * 3×3 스템에 맥스풀이 없다. 32×32 를 7×7 스템과 풀링으로 받으면 8×8 로 줄어
 * 남는 것이 별로 없다. ImageNet 판이 필요해지면 그때 별도 팩토리로 온다 —
 * 같은 이름에 인자로 가르면 옛 매니페스트가 다른 모델을 만들게 된다.
 */
import { nn, type Tensor } from "borch";
/**
 * ResNet 의 기본 블록. 지름길이 모양을 바꿔야 할 때만 1×1 을 둔다.
 *
 * ## 지름길 층은 반드시 **필드**여야 한다
 *
 * 코어 저장소가 값으로 치른 교훈이다. 전에는 `{ conv, bn }` 이라는 평범한 객체에
 * 담겨 있었고 `children()` 에만 적혀 있었는데, `namedChildren()` 은 `instanceof
 * Module` 인 **필드**만 훑으므로 그 둘을 못 봤다. **지름길 층 여섯이 한 번도 안
 * 배웠다.** 예외는 안 났고 손실은 내려갔다 — 나머지 층이 대신 맞추기 때문이다.
 * 정확도만 조용히 낮았다(65.5% 로 적혀 있던 수가 실은 그 상태였다).
 *
 * torch 도 파이썬 dict 에 담은 층은 등록하지 않는다. 그래서 `nn.ModuleDict` 가
 * 있는 것이고, 라이브러리가 옳았고 모델 쪽이 틀렸던 자리다.
 */
export declare class BasicBlock extends nn.Module {
    private readonly conv1;
    private readonly bn1;
    private readonly conv2;
    private readonly bn2;
    private readonly downConv;
    private readonly downBn;
    constructor(cin: number, cout: number, stride: number);
    forward(x: Tensor): Tensor;
}
/**
 * ResNet-18(CIFAR 판).
 *
 * 필드 넷이 그대로 자식이다 — `children()` 을 덮어쓰지 않는다. 덮어쓰면
 * `namedChildren()` 과 어긋날 자리가 생기고, 그 어긋남이 위 블록의 지름길 여섯을
 * 안 배우게 만든 것이다. `stateDict` 열쇠도 이 필드 이름에서 나오므로, 이름을
 * 바꾸면 **이미 배포된 가중치가 안 실린다.**
 */
export declare class ResNet18Cifar extends nn.Module {
    private readonly stem;
    private readonly bn;
    private readonly body;
    private readonly fc;
    constructor(numClasses: number);
    forward(x: Tensor): Tensor;
}
