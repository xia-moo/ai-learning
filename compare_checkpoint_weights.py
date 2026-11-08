from pathlib import Path
import torch

project_dir = Path(__file__).resolve().parent

original = torch.load(
    project_dir / "bert_sentiment_best.pt",
    map_location="cpu",
    weights_only=True
)

control = torch.load(
    project_dir / "bert_sentiment_negation_v1.pt",
    map_location="cpu",
    weights_only=True
)

for component in ["bert_state", "classifier_state"]:
    old_state = original[component]
    new_state = control[component]

    if old_state.keys() != new_state.keys():
        raise ValueError(f"{component} 的参数名称不一致")

    different = [
        name
        for name in old_state
        if not torch.equal(old_state[name], new_state[name])
    ]

    print(f"{component}：")
    print("  状态张量数量：", len(old_state))
    print("  不同的张量数量：", len(different))

    if different:
        print("  部分不同项：", different[:5])
        