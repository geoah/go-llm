package llm

import (
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"
)

type ChatGPT4 struct {
	client *openai.Client
}

func NewChatGPT4(client *openai.Client) *ChatGPT4 {
	return &ChatGPT4{
		client: client,
	}
}

func (o *ChatGPT4) Evaluate(ctx context.Context, prefix, prompt string) (string, error) {
	msgs := []openai.ChatCompletionMessage{}
	if prefix != "" {
		msgs = append(msgs, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: prefix,
		})
	}
	if prompt != "" {
		msgs = append(msgs, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: prompt,
		})
	}
	req := openai.ChatCompletionRequest{
		Model:    openai.GPT4,
		Messages: msgs,
	}
	resp, err := o.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("failed to create completion: %w", err)
	}

	return resp.Choices[0].Message.Content, nil
}
