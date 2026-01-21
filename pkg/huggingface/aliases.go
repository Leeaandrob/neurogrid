package huggingface

import (
	"flag"
	"os"
)

// Model aliases mapping short names to full HuggingFace repo IDs.
var modelAliases = map[string]string{
	"tinyllama":     "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
	"llama7b":       "meta-llama/Llama-2-7b-hf",
	"llama7b-chat":  "meta-llama/Llama-2-7b-chat-hf",
	"llama13b":      "meta-llama/Llama-2-13b-hf",
	"llama13b-chat": "meta-llama/Llama-2-13b-chat-hf",
}

// ResolveAlias resolves a model alias to a full HuggingFace repo ID.
// If the alias is not found, it returns the input unchanged.
func ResolveAlias(alias string) string {
	if repoID, ok := modelAliases[alias]; ok {
		return repoID
	}
	return alias
}

// ParseCLIArgs parses command-line arguments for the download CLI.
func ParseCLIArgs(args []string) CLIArgs {
	fs := flag.NewFlagSet("download", flag.ContinueOnError)

	var cli CLIArgs
	fs.StringVar(&cli.Repo, "repo", "", "HuggingFace repo ID or alias")
	fs.StringVar(&cli.Output, "output", "./models", "Output directory")
	fs.StringVar(&cli.Token, "token", os.Getenv("HF_TOKEN"), "HuggingFace token")

	fs.Parse(args)

	// Resolve alias if provided
	cli.Repo = ResolveAlias(cli.Repo)

	return cli
}
