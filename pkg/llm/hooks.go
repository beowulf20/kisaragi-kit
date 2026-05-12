package llm

// CompletionHooks contains optional callbacks for streaming and tool events.
type CompletionHooks struct {
	// OnContentDelta receives streamed assistant text chunks.
	OnContentDelta func(string)
	// OnCallError runs after a provider chat call fails.
	OnCallError func(error)
	// OnToolCall runs before a requested tool is executed.
	OnToolCall func(ToolCall)
	// OnToolError runs after a tool returns an error or invalid result.
	OnToolError func(ToolCall, error)
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

// EmitCallError invokes OnCallError when it is configured.
func (hooks CompletionHooks) EmitCallError(err error) {
	if hooks.OnCallError != nil {
		hooks.OnCallError(err)
	}
}

// EmitToolError invokes OnToolError when it is configured.
func (hooks CompletionHooks) EmitToolError(toolCall ToolCall, err error) {
	if hooks.OnToolError != nil {
		hooks.OnToolError(toolCall, err)
	}
}

// EmitToolResult invokes OnToolResult when it is configured.
func (hooks CompletionHooks) EmitToolResult(toolCall ToolCall) {
	if hooks.OnToolResult != nil {
		hooks.OnToolResult(toolCall)
	}
}
