def add_decision(prediction, threshold):
    if prediction["truncated"]:
        action = "转人工审核"
        reason = "输入发生截断"
    elif prediction["probabilities"]["正面"] >= threshold:
        action = "自动通过"
        reason = "正面概率达到阈值"
    else:
        action = "转人工审核"
        reason = "正面概率未达到阈值"

    return {
        **prediction,
        "decision_threshold": threshold,
        "action": action,
        "reason": reason
    }