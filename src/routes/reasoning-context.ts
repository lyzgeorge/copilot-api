import type { AnthropicMessagesPayload } from "~/routes/messages/anthropic-types"
import type { ChatCompletionsPayload } from "~/services/copilot/create-chat-completions"
import type { Model } from "~/services/copilot/get-models"

export interface ReasoningContext {
  reasoningEffort?: "low" | "medium" | "high"
  thinkingBudget?: number
}

export function buildAnthropicReasoningContext(
  payload: AnthropicMessagesPayload,
  model: Model | undefined,
): ReasoningContext {
  const adaptiveThinkingSupported =
    model?.capabilities.adaptive_thinking === true
  const thinkingEnabled = payload.thinking?.type === "enabled"
  return {
    reasoningEffort:
      thinkingEnabled && adaptiveThinkingSupported ? "high" : undefined,
    thinkingBudget:
      thinkingEnabled && adaptiveThinkingSupported ?
        payload.thinking?.budget_tokens
      : undefined,
  }
}

export function buildOpenAIReasoningContext(
  payload: ChatCompletionsPayload,
  model: Model | undefined,
): ReasoningContext {
  const adaptiveThinkingSupported =
    model?.capabilities.adaptive_thinking === true
  return {
    reasoningEffort: payload.reasoning_effort ?? undefined,
    thinkingBudget:
      adaptiveThinkingSupported ?
        (payload.thinking_budget ?? undefined)
      : undefined,
  }
}
