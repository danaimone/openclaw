// Defaults for agent metadata when upstream does not supply them.
// claude-sdk uses the Claude Agent SDK (Pro/Max subscription, no API credits).
// To revert to API-based usage, change to "anthropic".
export const DEFAULT_PROVIDER = "claude-sdk";
export const DEFAULT_MODEL = "sonnet";
// Conservative fallback used when model metadata is unavailable.
export const DEFAULT_CONTEXT_TOKENS = 200_000;
