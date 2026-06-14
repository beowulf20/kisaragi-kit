package llm

// ReasoningEffort constrains how much reasoning a supported model should use.
type ReasoningEffort string

const (
	// ReasoningEffortNone disables reasoning on models that support it.
	ReasoningEffortNone ReasoningEffort = "none"
	// ReasoningEffortMinimal uses the smallest available reasoning budget on models that support it.
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	// ReasoningEffortLow uses a low reasoning budget on models that support it.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortMedium uses a medium reasoning budget on models that support it.
	ReasoningEffortMedium ReasoningEffort = "medium"
	// ReasoningEffortHigh uses a high reasoning budget on models that support it.
	ReasoningEffortHigh ReasoningEffort = "high"
	// ReasoningEffortXHigh uses the highest available reasoning budget on models that support it.
	ReasoningEffortXHigh ReasoningEffort = "xhigh"
)

func (effort ReasoningEffort) valid() bool {
	switch effort {
	case "", ReasoningEffortNone, ReasoningEffortMinimal, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh:
		return true
	default:
		return false
	}
}
