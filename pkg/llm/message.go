package llm

type MessageType string

const (
	System    = "system"
	User      = "user"
	Assistant = "assistant"
	Tool      = "tool"
)

type Message struct {
	Type       MessageType
	Content    string
	ToolCallID string
	ToolCalls  []ToolCall
}

func NewSystemMessage(content string) Message {
	return Message{Type: System, Content: content}
}

func NewUserMessage(content string) Message {
	return Message{Type: User, Content: content}
}

func NewAssistantMessage(content string) Message {
	return Message{Type: Assistant, Content: content}
}

func NewAssistantToolCallMessage(content string, toolCalls []ToolCall) Message {
	return Message{
		Type:      Assistant,
		Content:   content,
		ToolCalls: append([]ToolCall(nil), toolCalls...),
	}
}

func NewToolMessage(toolCallID, content string) Message {
	return Message{Type: Tool, ToolCallID: toolCallID, Content: content}
}

type ToolCall struct {
	ID        string
	Name      string
	Arguments string
	Result    string
}
