// Defaults for agent metadata when upstream does not supply them.
// claude-cli uses the Claude Code CLI with Pro/Max subscription (no API credits).
// clearEnv strips ANTHROPIC_API_KEY so it uses subscription auth.
// To revert to API-based usage, change to "anthropic".
export const DEFAULT_PROVIDER = "claude-cli";
export const DEFAULT_MODEL = "opus";
// Conservative fallback used when model metadata is unavailable.
export const DEFAULT_CONTEXT_TOKENS = 200_000;
