package llm

// MessageType identifies the role of a chat message.
type MessageType string

const (
	// System is a system instruction message.
	System = "system"
	// User is an end-user message.
	User = "user"
	// Assistant is an assistant response message.
	Assistant = "assistant"
	// Tool is a tool result message.
	Tool = "tool"
)

// Message is a provider-neutral chat message.
type Message struct {
	// Type is the role or category of this message.
	Type MessageType
	// Content is the text payload.
	Content string
	// ToolCallID links a tool result message to its assistant tool call.
	ToolCallID string
	// ToolCalls contains assistant-requested tool calls.
	ToolCalls []ToolCall
}

// NewSystemMessage returns a system instruction message.
func NewSystemMessage(content string) Message {
	return Message{Type: System, Content: content}
}

// NewUserMessage returns a user message.
func NewUserMessage(content string) Message {
	return Message{Type: User, Content: content}
}

// NewAssistantMessage returns an assistant text message.
func NewAssistantMessage(content string) Message {
	return Message{Type: Assistant, Content: content}
}

// NewAssistantToolCallMessage returns an assistant message containing tool calls.
func NewAssistantToolCallMessage(content string, toolCalls []ToolCall) Message {
	return Message{
		Type:      Assistant,
		Content:   content,
		ToolCalls: append([]ToolCall(nil), toolCalls...),
	}
}

// NewToolMessage returns a tool result message for a tool call ID.
func NewToolMessage(toolCallID, content string) Message {
	return Message{Type: Tool, ToolCallID: toolCallID, Content: content}
}

// ToolCall describes a tool call requested by an assistant or executed locally.
type ToolCall struct {
	// ID is the provider-assigned tool call ID.
	ID string
	// Name is the registered tool name.
	Name string
	// Arguments is the JSON argument payload.
	Arguments string
	// Result is the JSON result or error payload after execution.
	Result string
}
