package openrouter

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
	openaiadapter "github.com/beowulf20/kisaragi-kit/pkg/llm/provider/openai"
)

const DefaultBaseURL = "https://openrouter.ai/api/v1"

var typedProviderFields = map[string]struct{}{
	"allow_fallbacks": {},
	"ignore":          {},
	"only":            {},
	"order":           {},
}

// ProviderPreferences controls which OpenRouter providers may serve requests.
type ProviderPreferences struct {
	// Only restricts requests to these provider slugs.
	Only []string
	// Ignore excludes these provider slugs.
	Ignore []string
	// Order prioritizes provider slugs without restricting membership by itself.
	Order []string
	// AllowFallbacks controls OpenRouter fallback behavior. Nil uses OpenRouter's default.
	AllowFallbacks *bool
	// ExtraFields adds advanced OpenRouter provider fields not modeled above.
	ExtraFields map[string]any
}

// ClientConfig configures an OpenRouter client.
type ClientConfig struct {
	// BaseURL defaults to DefaultBaseURL when empty.
	BaseURL string
	// APIKey is the OpenRouter API key.
	APIKey string
	// Timeout limits individual API requests when greater than zero.
	Timeout time.Duration
	// Provider contains client-wide provider routing preferences.
	Provider ProviderPreferences
	// ChatCompletionExtraFields adds advanced top-level OpenRouter request fields.
	// The provider key is reserved for Provider.
	ChatCompletionExtraFields map[string]any
}

// Client is an OpenRouter-configured OpenAI-compatible client.
type Client struct {
	client *openaiadapter.Client
}

// NewClient validates and normalizes OpenRouter configuration.
func NewClient(config ClientConfig) (*Client, ClientConfig, error) {
	config.BaseURL = strings.TrimRight(strings.TrimSpace(config.BaseURL), "/")
	if config.BaseURL == "" {
		config.BaseURL = DefaultBaseURL
	}
	if strings.TrimSpace(config.APIKey) == "" {
		return nil, config, errors.New("API key cannot be empty")
	}

	provider, providerFields, err := normalizeProviderPreferences(config.Provider)
	if err != nil {
		return nil, config, err
	}
	config.Provider = provider

	extraFields := cloneFields(config.ChatCompletionExtraFields)
	if _, exists := extraFields["provider"]; exists {
		return nil, config, errors.New("chat completion extra field provider is reserved; use Provider")
	}
	config.ChatCompletionExtraFields = cloneFields(extraFields)
	if len(providerFields) > 0 {
		if extraFields == nil {
			extraFields = make(map[string]any)
		}
		extraFields["provider"] = providerFields
	}

	client, _, err := openaiadapter.NewClient(openaiadapter.ClientConfig{
		BaseURL:                   config.BaseURL,
		APIKey:                    config.APIKey,
		Timeout:                   config.Timeout,
		ChatCompletionExtraFields: extraFields,
	})
	if err != nil {
		return nil, config, err
	}
	return &Client{client: client}, config, nil
}

// Complete sends one streaming chat completion through OpenRouter.
func (client *Client) Complete(ctx context.Context, request llm.ChatRequest, hooks llm.CompletionHooks) (*llm.ChatResponse, error) {
	if client == nil || client.client == nil {
		return nil, errors.New("openrouter client is nil")
	}
	return client.client.Complete(ctx, request, hooks)
}

// ListModels returns model IDs available from OpenRouter.
func (client *Client) ListModels(ctx context.Context) ([]string, error) {
	if client == nil || client.client == nil {
		return nil, errors.New("openrouter client is nil")
	}
	return client.client.ListModels(ctx)
}

func normalizeProviderPreferences(input ProviderPreferences) (ProviderPreferences, map[string]any, error) {
	only, err := normalizeSlugs("only", input.Only)
	if err != nil {
		return ProviderPreferences{}, nil, err
	}
	ignore, err := normalizeSlugs("ignore", input.Ignore)
	if err != nil {
		return ProviderPreferences{}, nil, err
	}
	order, err := normalizeSlugs("order", input.Order)
	if err != nil {
		return ProviderPreferences{}, nil, err
	}

	onlySet := stringSet(only)
	ignoreSet := stringSet(ignore)
	for _, slug := range only {
		if _, exists := ignoreSet[slug]; exists {
			return ProviderPreferences{}, nil, fmt.Errorf("provider %q cannot appear in both only and ignore", slug)
		}
	}
	for _, slug := range order {
		if _, exists := ignoreSet[slug]; exists {
			return ProviderPreferences{}, nil, fmt.Errorf("ordered provider %q cannot be ignored", slug)
		}
		if len(onlySet) > 0 {
			if _, exists := onlySet[slug]; !exists {
				return ProviderPreferences{}, nil, fmt.Errorf("ordered provider %q must appear in only", slug)
			}
		}
	}

	extraFields := cloneFields(input.ExtraFields)
	for key := range extraFields {
		if strings.TrimSpace(key) == "" {
			return ProviderPreferences{}, nil, errors.New("provider extra field name cannot be empty")
		}
		if _, reserved := typedProviderFields[key]; reserved {
			return ProviderPreferences{}, nil, fmt.Errorf("provider extra field %q is reserved by typed preferences", key)
		}
	}

	normalized := ProviderPreferences{
		Only:           only,
		Ignore:         ignore,
		Order:          order,
		AllowFallbacks: cloneBool(input.AllowFallbacks),
		ExtraFields:    cloneFields(extraFields),
	}
	fields := cloneFields(extraFields)
	if len(only) > 0 || len(ignore) > 0 || len(order) > 0 || normalized.AllowFallbacks != nil {
		if fields == nil {
			fields = make(map[string]any)
		}
	}
	if len(only) > 0 {
		fields["only"] = append([]string(nil), only...)
	}
	if len(ignore) > 0 {
		fields["ignore"] = append([]string(nil), ignore...)
	}
	if len(order) > 0 {
		fields["order"] = append([]string(nil), order...)
	}
	if normalized.AllowFallbacks != nil {
		fields["allow_fallbacks"] = *normalized.AllowFallbacks
	}
	return normalized, fields, nil
}

func normalizeSlugs(field string, values []string) ([]string, error) {
	if len(values) == 0 {
		return nil, nil
	}
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		slug := strings.TrimSpace(value)
		if slug == "" {
			return nil, fmt.Errorf("provider %s contains an empty slug", field)
		}
		if _, exists := seen[slug]; exists {
			continue
		}
		seen[slug] = struct{}{}
		result = append(result, slug)
	}
	return result, nil
}

func stringSet(values []string) map[string]struct{} {
	result := make(map[string]struct{}, len(values))
	for _, value := range values {
		result[value] = struct{}{}
	}
	return result
}

func cloneBool(value *bool) *bool {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func cloneFields(fields map[string]any) map[string]any {
	if len(fields) == 0 {
		return nil
	}
	cloned := make(map[string]any, len(fields))
	for key, value := range fields {
		cloned[key] = value
	}
	return cloned
}
