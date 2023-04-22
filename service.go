package llm

import (
	"context"
	"fmt"
)

type (
	Service struct {
		evaluators map[string]Evaluator
		prefix     string
	}
	Evaluator interface {
		Evaluate(context.Context, string, string) (string, error)
	}
)

func NewService(prefix string, evaluators map[string]Evaluator) *Service {
	return &Service{
		prefix:     prefix,
		evaluators: evaluators,
	}
}

func (s *Service) Evaluate(ctx context.Context, evaluator, expr string) (string, error) {
	e, ok := s.evaluators[evaluator]
	if !ok {
		return "", fmt.Errorf("unknown evaluator: %s", evaluator)
	}
	return e.Evaluate(ctx, s.prefix, expr)
}
