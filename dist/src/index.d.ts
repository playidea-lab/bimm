/**
 * bimm — **timm 의 자리.** 아키텍처 카탈로그와, 이름으로 그것을 만드는 함수.
 *
 * ```ts
 * import { init } from "borch";
 * import { createModel, listModels } from "bimm";
 *
 * await init();                                   // 층이 곧 텐서다
 * listModels();                                   // [{ library, factory }, …]
 * const model = createModel("borchvision", "resnet18_cifar", { numClasses: 10 });
 * ```
 *
 * ## 여기 없는 것과, 그것이 어디 있는지
 *
 * **가중치를 받아오지 않는다.** timm 의 `create_model(pretrained=True)` 에 해당하는
 * 일 — 매니페스트를 읽고, 해시를 대조하고, 브라우저가 그 모델을 돌릴 수 있는지
 * 먼저 판정하는 것 — 은 `borch-hub` 에 있다. 갈라 둔 이유는 **의존이 한 방향으로만
 * 흐르게** 하기 위해서다: 허브는 카탈로그를 알아야 하지만, 카탈로그는 매니페스트를
 * 몰라도 된다. 모델 하나 만들려는 사람이 배포·검증 계층을 통째로 끌어오지 않는다.
 *
 * 그 방향은 실제 생태계와도 같다. timm 은 아키텍처를 알고, 어디서 받아오는지는
 * 그 바깥의 일이다.
 */
export { createModel, factorySpec, listModels, type FactoryName } from "./registry.js";
export { checkArgs, type ArgSpec, type FactoryArgs } from "./args.js";
export { BasicBlock, ResNet18Cifar } from "./resnet.js";
export { BimmError } from "./errors.js";
