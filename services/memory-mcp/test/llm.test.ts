// The hand-rolled Anthropic Messages client (src/llm.ts) — PAID-API guard: every test
// stubs globalThis.fetch; nothing here ever touches the network.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../src/config.js", () => ({
  config: {
    anthropicApiKey: "test-key",
    consolidateModel: "consolidate-model",
    distillModel: "distill-model",
    defaultProjectId: "test",
  },
}));

import { judgeComplete, llmFetch } from "../src/llm.js";
import { config } from "../src/config.js";

function jsonResponse(
  body: unknown,
  status = 200,
  headers: Record<string, string> = {},
) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

describe("llm client (anthropic messages, mocked fetch)", () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    (config as { anthropicApiKey: string }).anthropicApiKey = "test-key";
  });

  it("sends the messages request with auth + version headers and returns joined text", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        content: [
          { type: "text", text: "hello " },
          { type: "tool_use", id: "x" },
          { type: "text", text: "world" },
        ],
        stop_reason: "end_turn",
      }),
    );

    const text = await judgeComplete("sys", "user prompt");
    expect(text).toBe("hello world");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(String(url)).toBe("https://api.anthropic.com/v1/messages");
    const headers = init!.headers as Record<string, string>;
    expect(headers["x-api-key"]).toBe("test-key");
    expect(headers["anthropic-version"]).toBe("2023-06-01");
    const body = JSON.parse(String(init!.body)) as Record<string, unknown>;
    expect(body.model).toBe("consolidate-model"); // default = config.consolidateModel
    expect(body.system).toBe("sys");
    expect(body.messages).toEqual([{ role: "user", content: "user prompt" }]);
  });

  it("honors an explicit model override (distill passes config.distillModel)", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ content: [{ type: "text", text: "ok" }] }),
    );
    await judgeComplete("s", "u", { model: "distill-model" });
    const body = JSON.parse(String(fetchMock.mock.calls[0]![1]!.body)) as {
      model: string;
    };
    expect(body.model).toBe("distill-model");
  });

  it("retries 429 (Retry-After honored) then succeeds", async () => {
    fetchMock
      .mockResolvedValueOnce(
        new Response("rate limited", {
          status: 429,
          headers: { "retry-after": "0.01" },
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({ content: [{ type: "text", text: "ok" }] }),
      );

    const text = await judgeComplete("s", "u");
    expect(text).toBe("ok");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("does NOT retry a 400 — throws with status + body", async () => {
    fetchMock.mockResolvedValueOnce(
      new Response("bad request", { status: 400 }),
    );
    await expect(llmFetch("/messages", {})).rejects.toThrow(
      /anthropic \/messages 400: bad request/,
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("fails with a clear operator message when ANTHROPIC_API_KEY is absent", async () => {
    (config as { anthropicApiKey: string }).anthropicApiKey = "";
    await expect(judgeComplete("s", "u")).rejects.toThrow(
      /ANTHROPIC_API_KEY is not set/,
    );
    expect(fetchMock).not.toHaveBeenCalled(); // refused before any network call
  });

  it("throws on an empty completion, naming the stop_reason", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ content: [], stop_reason: "refusal" }),
    );
    await expect(judgeComplete("s", "u")).rejects.toThrow(
      /stop_reason: refusal/,
    );
  });
});
