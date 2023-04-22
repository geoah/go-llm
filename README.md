# LLM

## API Usage
```sh
curl --request POST \
     --header "Content-Type: application/json" \
     --data '{"prompt": "What is 42?", "model": "gpt4"}' \
     http://localhost:8080/api/v1/evaluate
```