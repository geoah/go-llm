package llm

import (
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"
)

type ChatGPT3p5 struct {
	client *openai.Client
}

func NewChatGPT3p5(client *openai.Client) *ChatGPT3p5 {
	return &ChatGPT3p5{
		client: client,
	}
}

func (o *ChatGPT3p5) Evaluate(ctx context.Context, prefix, prompt string) (string, error) {
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
		Model:    openai.GPT3Dot5Turbo,
		Messages: msgs,
	}
	resp, err := o.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("failed to create completion: %w", err)
	}

	return resp.Choices[0].Message.Content, nil
}
