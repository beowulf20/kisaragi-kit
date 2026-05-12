package llm

// CompletionHooks contains optional callbacks for streaming and tool events.
type CompletionHooks struct {
	// OnContentDelta receives streamed assistant text chunks.
	OnContentDelta func(string)
	// OnToolCall runs before a requested tool is executed.
	OnToolCall func(ToolCall)
	// OnToolResult runs after a tool returns or fails.
	OnToolResult func(ToolCall)
}

// EmitToolCall invokes OnToolCall when it is configured.
func (hooks CompletionHooks) EmitToolCall(toolCall ToolCall) {
	if hooks.OnToolCall != nil {
		hooks.OnToolCall(toolCall)
	}
}

// EmitContentDelta invokes OnContentDelta when it is configured.
func (hooks CompletionHooks) EmitContentDelta(delta string) {
	if hooks.OnContentDelta != nil {
		hooks.OnContentDelta(delta)
	}
}

// EmitToolResult invokes OnToolResult when it is configured.
func (hooks CompletionHooks) EmitToolResult(toolCall ToolCall) {
	if hooks.OnToolResult != nil {
		hooks.OnToolResult(toolCall)
	}
}
