from sentiment_policy import add_decision

# 概率、是否截断、预期动作、预期原因
cases = [
    (0.95, False, "自动通过", "正面概率达到阈值"),
    (0.90, False, "自动通过", "正面概率达到阈值"),
    (0.89, False, "转人工审核", "正面概率未达到阈值"),
    (0.99, True, "转人工审核", "输入发生截断"),
]

for probability, truncated, expected_action, expected_reason in cases:
    prediction = {
        "probabilities": {"正面": probability},
        "truncated": truncated
    }

    result = add_decision(prediction, threshold=0.90)

    assert result["action"] == expected_action, (
        f"动作不符合预期：概率={probability}，截断={truncated}"
    )
    assert result["reason"] == expected_reason, "处理原因不符合预期"

    print(f"通过：概率={probability}，截断={truncated}")

print("全部规则检查通过")