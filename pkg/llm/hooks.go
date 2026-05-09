package llm

type CompletionHooks struct {
	OnContentDelta func(string)
	OnToolCall     func(ToolCall)
	OnToolResult   func(ToolCall)
}

func (hooks CompletionHooks) EmitToolCall(toolCall ToolCall) {
	if hooks.OnToolCall != nil {
		hooks.OnToolCall(toolCall)
	}
}

func (hooks CompletionHooks) EmitContentDelta(delta string) {
	if hooks.OnContentDelta != nil {
		hooks.OnContentDelta(delta)
	}
}

func (hooks CompletionHooks) EmitToolResult(toolCall ToolCall) {
	if hooks.OnToolResult != nil {
		hooks.OnToolResult(toolCall)
	}
}
