package agent

import (
	"context"
	"errors"
	"strings"
	"unicode"

	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

// AsTool exposes the agent as a callable LLM tool.
func (a *Agent) AsTool() llmtool.Tool {
	if a == nil {
		return llmtool.NewTool("agent", "Calls an agent.", func(_ context.Context, input agentToolInput) (agentToolOutput, error) {
			return agentToolOutput{}, errors.New("agent is nil: " + input.Query)
		})
	}

	description := a.Description
	if description == "" {
		description = "Calls the " + a.Name + " agent."
	}

	return llmtool.NewTool("agent_"+toolName(a.Name), description, func(ctx context.Context, input agentToolInput) (agentToolOutput, error) {
		output, err := a.CallWithUserMessageContext(ctx, input.Query)
		if err != nil {
			return agentToolOutput{}, err
		}
		return agentToolOutput{Response: output.Content}, nil
	})
}

type agentToolInput struct {
	Query string `json:"query" description:"Query to send to the agent."`
}

type agentToolOutput struct {
	Response string `json:"response"`
}

func toolName(name string) string {
	var out strings.Builder
	lastUnderscore := false
	for _, r := range strings.TrimSpace(name) {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r):
			out.WriteRune(unicode.ToLower(r))
			lastUnderscore = false
		case !lastUnderscore:
			out.WriteByte('_')
			lastUnderscore = true
		}
	}

	value := strings.Trim(out.String(), "_")
	if value == "" {
		return "agent"
	}
	return value
}
