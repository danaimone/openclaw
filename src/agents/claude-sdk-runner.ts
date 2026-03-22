/**
 * Claude Agent SDK runner — replaces the raw CLI subprocess approach with the
 * programmatic `query()` API from `@anthropic-ai/claude-agent-sdk`.
 *
 * Uses the Claude Pro/Max subscription via the SDK (no API credits needed).
 * Returns `EmbeddedPiRunResult` to slot into the existing reply pipeline.
 */

import { query } from "@anthropic-ai/claude-agent-sdk";
import type { SDKMessage, SDKResultSuccess, SDKResultError } from "@anthropic-ai/claude-agent-sdk";

import { resolveHeartbeatPrompt } from "../auto-reply/heartbeat.js";
import type { ThinkLevel } from "../auto-reply/thinking.js";
import type { OpenClawConfig } from "../config/config.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
import { resolveSessionAgentIds } from "./agent-scope.js";
import {
	analyzeBootstrapBudget,
	buildBootstrapInjectionStats,
	buildBootstrapPromptWarning,
	buildBootstrapTruncationReportMeta,
	prependBootstrapPromptWarning,
} from "./bootstrap-budget.js";
import { makeBootstrapWarn, resolveBootstrapContextForRun } from "./bootstrap-files.js";
import { resolveCliBackendConfig } from "./cli-backends.js";
import {
	buildSystemPrompt,
	normalizeCliModel,
} from "./cli-runner/helpers.js";
import { resolveOpenClawDocsPath } from "./docs-path.js";
import { FailoverError, resolveFailoverStatus } from "./failover-error.js";
import {
	resolveBootstrapMaxChars,
	resolveBootstrapPromptTruncationWarningMode,
	resolveBootstrapTotalMaxChars,
} from "./pi-embedded-helpers.js";
import type { EmbeddedPiRunResult } from "./pi-embedded-runner.js";
import { buildSystemPromptReport } from "./system-prompt-report.js";
import { redactRunIdentifier, resolveRunWorkspaceDir } from "./workspace-run.js";

const log = createSubsystemLogger("agent/claude-sdk");

// Track SDK session IDs per OpenClaw session key for conversation continuity.
const sdkSessionMap = new Map<string, string>();

export async function runClaudeSdkAgent(params: {
	sessionId: string;
	sessionKey?: string;
	agentId?: string;
	sessionFile: string;
	workspaceDir: string;
	config?: OpenClawConfig;
	prompt: string;
	provider: string;
	model?: string;
	thinkLevel?: ThinkLevel;
	timeoutMs: number;
	runId: string;
	extraSystemPrompt?: string;
	ownerNumbers?: string[];
	cliSessionId?: string;
	bootstrapPromptWarningSignaturesSeen?: string[];
	bootstrapPromptWarningSignature?: string;
}): Promise<EmbeddedPiRunResult> {
	const started = Date.now();

	// --- Resolve workspace, model, system prompt (reuse existing helpers) ---

	const workspaceResolution = resolveRunWorkspaceDir({
		workspaceDir: params.workspaceDir,
		sessionKey: params.sessionKey,
		agentId: params.agentId,
		config: params.config,
	});
	const workspaceDir = workspaceResolution.workspaceDir;

	const backendResolved = resolveCliBackendConfig(params.provider, params.config);
	const modelId = (params.model ?? "default").trim() || "default";
	const normalizedModel = backendResolved
		? normalizeCliModel(modelId, backendResolved.config)
		: modelId;
	const modelDisplay = `${params.provider}/${modelId}`;

	const { bootstrapFiles, contextFiles } = await resolveBootstrapContextForRun({
		workspaceDir,
		config: params.config,
		sessionKey: params.sessionKey,
		sessionId: params.sessionId,
		warn: makeBootstrapWarn({
			sessionLabel: params.sessionKey ?? params.sessionId,
			warn: (message) => log.warn(message),
		}),
	});

	const bootstrapMaxChars = resolveBootstrapMaxChars(params.config);
	const bootstrapTotalMaxChars = resolveBootstrapTotalMaxChars(params.config);
	const bootstrapAnalysis = analyzeBootstrapBudget({
		files: buildBootstrapInjectionStats({ bootstrapFiles, injectedFiles: contextFiles }),
		bootstrapMaxChars,
		bootstrapTotalMaxChars,
	});
	const bootstrapPromptWarningMode = resolveBootstrapPromptTruncationWarningMode(params.config);
	const bootstrapPromptWarning = buildBootstrapPromptWarning({
		analysis: bootstrapAnalysis,
		mode: bootstrapPromptWarningMode,
		seenSignatures: params.bootstrapPromptWarningSignaturesSeen,
		previousSignature: params.bootstrapPromptWarningSignature,
	});

	const { defaultAgentId, sessionAgentId } = resolveSessionAgentIds({
		sessionKey: params.sessionKey,
		config: params.config,
		agentId: params.agentId,
	});
	const heartbeatPrompt =
		sessionAgentId === defaultAgentId
			? resolveHeartbeatPrompt(params.config?.agents?.defaults?.heartbeat?.prompt)
			: undefined;
	const docsPath = await resolveOpenClawDocsPath({
		workspaceDir,
		argv1: process.argv[1],
		cwd: process.cwd(),
		moduleUrl: import.meta.url,
	});

	const systemPrompt = buildSystemPrompt({
		workspaceDir,
		config: params.config,
		defaultThinkLevel: params.thinkLevel,
		extraSystemPrompt: params.extraSystemPrompt,
		ownerNumbers: params.ownerNumbers,
		heartbeatPrompt,
		docsPath: docsPath ?? undefined,
		tools: [],
		contextFiles,
		modelDisplay,
		agentId: sessionAgentId,
	});

	const systemPromptReport = buildSystemPromptReport({
		source: "run",
		generatedAt: Date.now(),
		sessionId: params.sessionId,
		sessionKey: params.sessionKey,
		provider: params.provider,
		model: modelId,
		workspaceDir,
		bootstrapMaxChars,
		bootstrapTotalMaxChars,
		bootstrapTruncation: buildBootstrapTruncationReportMeta({
			analysis: bootstrapAnalysis,
			warningMode: bootstrapPromptWarningMode,
			warning: bootstrapPromptWarning,
		}),
		sandbox: { mode: "off", sandboxed: false },
		systemPrompt,
		bootstrapFiles,
		injectedFiles: contextFiles,
		skillsPrompt: "",
		tools: [],
	});

	// --- Prepare prompt (with bootstrap warning prepended) ---

	const prompt = prependBootstrapPromptWarning(params.prompt, bootstrapPromptWarning.lines, {
		preserveExactPrompt: heartbeatPrompt,
	});

	// --- Execute via Claude Agent SDK ---

	const sessionKey = params.sessionKey ?? params.sessionId;
	const existingSdkSession = sdkSessionMap.get(sessionKey);

	log.info(
		`sdk exec: provider=${params.provider} model=${normalizedModel} promptChars=${prompt.length} resume=${!!existingSdkSession}`,
	);

	let fullText = "";
	let sdkSessionId: string | undefined;
	let sdkUsage: { input?: number; output?: number; total?: number } | undefined;

	try {
		const abortController = new AbortController();
		const timeoutId = setTimeout(() => abortController.abort(), params.timeoutMs);

		const q = query({
			prompt,
			options: {
				model: normalizedModel,
				systemPrompt,
				permissionMode: "bypassPermissions",
				allowDangerouslySkipPermissions: true,
				maxTurns: 25,
				cwd: workspaceDir,
				abortController,
				env: (() => {
					const env: Record<string, string | undefined> = { ...process.env, IS_SANDBOX: "1" };
					// Strip API keys so Claude Code uses subscription auth, not credits.
					delete env.ANTHROPIC_API_KEY;
					delete env.ANTHROPIC_API_KEY_OLD;
					return env;
				})(),
				stderr: (data: string) => {
					if (data.trim()) {
						log.debug(`sdk stderr: ${data.trim()}`);
					}
				},
				...(existingSdkSession ? { resume: existingSdkSession } : {}),
			},
		});

		for await (const message of q) {
			handleSdkMessage(message, {
				onSessionId(id) {
					sdkSessionId = id;
				},
				onText(text) {
					fullText += text;
				},
				onResultText(text) {
					// Final result replaces any accumulated text
					fullText = text;
				},
				onUsage(usage) {
					sdkUsage = usage;
				},
			});
		}

		clearTimeout(timeoutId);
	} catch (err) {
		const msg = err instanceof Error ? err.message : String(err);

		// Classify known failure modes
		if (msg.includes("aborted") || msg.includes("AbortError")) {
			throw new FailoverError(`SDK query timed out after ${Math.round(params.timeoutMs / 1000)}s`, {
				reason: "timeout",
				provider: params.provider,
				model: modelId,
				status: resolveFailoverStatus("timeout"),
			});
		}
		if (msg.includes("rate limit") || msg.includes("429")) {
			throw new FailoverError(`SDK rate limited: ${msg}`, {
				reason: "rate_limit",
				provider: params.provider,
				model: modelId,
				status: resolveFailoverStatus("rate_limit"),
			});
		}

		log.error(`sdk error: ${msg}`);
		throw new FailoverError(msg, {
			reason: "unknown",
			provider: params.provider,
			model: modelId,
			status: resolveFailoverStatus("unknown"),
		});
	}

	// Store SDK session for future conversation continuity
	if (sdkSessionId) {
		sdkSessionMap.set(sessionKey, sdkSessionId);
		log.debug(
			`sdk session stored: key=${redactRunIdentifier(sessionKey)} sdkSession=${sdkSessionId}`,
		);
	}

	const text = fullText.trim();
	const payloads = text ? [{ text }] : undefined;

	return {
		payloads,
		meta: {
			durationMs: Date.now() - started,
			systemPromptReport,
			agentMeta: {
				sessionId: sdkSessionId ?? params.cliSessionId ?? params.sessionId ?? "",
				provider: params.provider,
				model: normalizedModel,
				usage: sdkUsage
					? {
							input: sdkUsage.input,
							output: sdkUsage.output,
							total: sdkUsage.total,
						}
					: undefined,
			},
		},
	};
}

// --- SDK message handler ---

type SdkCallbacks = {
	onSessionId: (id: string) => void;
	onText: (text: string) => void;
	onResultText: (text: string) => void;
	onUsage: (usage: { input?: number; output?: number; total?: number }) => void;
};

function handleSdkMessage(
	message: SDKMessage,
	callbacks: SdkCallbacks,
): void {
	switch (message.type) {
		case "system":
			if ("session_id" in message && message.session_id) {
				callbacks.onSessionId(message.session_id);
			}
			break;

		case "assistant":
			if ("message" in message && message.message?.content) {
				for (const block of message.message.content) {
					if ("text" in block && block.text) {
						callbacks.onText(block.text);
					}
				}
			}
			break;

		case "result": {
			const result = message as SDKResultSuccess | SDKResultError;
			if ("session_id" in result) {
				callbacks.onSessionId(result.session_id);
			}
			if (result.subtype === "success") {
				const success = result as SDKResultSuccess;
				if (success.result) {
					callbacks.onResultText(success.result);
				}
				if (success.usage) {
					callbacks.onUsage({
						input: success.usage.input_tokens,
						output: success.usage.output_tokens,
						total: (success.usage.input_tokens ?? 0) + (success.usage.output_tokens ?? 0),
					});
				}
			} else {
				const error = result as SDKResultError;
				log.error(`sdk result error: subtype=${error.subtype} errors=${error.errors?.join(", ")}`);
			}
			break;
		}

		default:
			// Ignore other message types (status, rate_limit, tool_progress, etc.)
			break;
	}
}
