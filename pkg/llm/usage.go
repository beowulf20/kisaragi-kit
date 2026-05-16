package llm

// TokenUsage contains provider-neutral token usage for one or more generations.
type TokenUsage struct {
	// PromptTokens is the number of input tokens billed by the provider.
	PromptTokens int64
	// CompletionTokens is the number of output tokens billed by the provider.
	CompletionTokens int64
	// TotalTokens is the total number of billed tokens reported by the provider.
	TotalTokens int64
	// PromptTokenDetails contains provider-specific input token details.
	PromptTokenDetails map[string]int64
	// CompletionTokenDetails contains provider-specific output token details.
	CompletionTokenDetails map[string]int64
}

func (usage *TokenUsage) add(other TokenUsage) {
	if usage == nil {
		return
	}
	usage.PromptTokens += other.PromptTokens
	usage.CompletionTokens += other.CompletionTokens
	usage.TotalTokens += other.TotalTokens
	usage.PromptTokenDetails = addTokenDetails(usage.PromptTokenDetails, other.PromptTokenDetails)
	usage.CompletionTokenDetails = addTokenDetails(usage.CompletionTokenDetails, other.CompletionTokenDetails)
}

func (usage TokenUsage) clone() TokenUsage {
	usage.PromptTokenDetails = cloneTokenDetails(usage.PromptTokenDetails)
	usage.CompletionTokenDetails = cloneTokenDetails(usage.CompletionTokenDetails)
	return usage
}

func addTokenDetails(dst map[string]int64, src map[string]int64) map[string]int64 {
	if len(src) == 0 {
		return dst
	}
	if dst == nil {
		dst = make(map[string]int64, len(src))
	}
	for key, value := range src {
		dst[key] += value
	}
	return dst
}

func cloneTokenDetails(src map[string]int64) map[string]int64 {
	if len(src) == 0 {
		return nil
	}
	dst := make(map[string]int64, len(src))
	for key, value := range src {
		dst[key] = value
	}
	return dst
}
