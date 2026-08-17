"""Presentation helpers for rule-based pose classifications."""

from __future__ import annotations

from pose_classifier import Prediction


def humanize_label(label: str) -> str:
    """Turn a classifier label into UI text without changing its identity."""

    return label.replace("_", " ").title()


def reference_diagnostic(
    rule_label: str,
    prediction: Prediction | None,
) -> str:
    """Describe the legacy embedding result without treating it as confidence.

    The angle-rule label drives the UI action.  Embedding similarity is only a
    relative nearest-reference score and must not be displayed as confidence
    in the rule-based decision.
    """

    if prediction is None:
        return "reference comparison unavailable"
    nearest = humanize_label(prediction.label)
    if prediction.label == rule_label:
        return f"nearest reference agrees: {nearest}"
    return f"nearest reference (diagnostic only): {nearest}"
