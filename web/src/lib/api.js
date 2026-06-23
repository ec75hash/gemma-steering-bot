// Thin client for the Gemma Test Kitchen steering API (steer_server.py).

async function req(url, options = {}) {
  const response = await fetch(url, {
    method: options.method || "GET",
    headers: options.body ? { "Content-Type": "application/json" } : undefined,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || data.error || `HTTP ${response.status}`);
  }
  return data;
}

export const api = {
  status: () => req("/api/status"),
  load: (model, promptMode) => req("/api/load", { method: "POST", body: { model, promptMode } }),
  humPrompt: () => req("/api/catalog/hum-prompt"),

  features: (q = "") => req(`/api/catalog/features?q=${encodeURIComponent(q)}`),
  bundles: (q = "") => req(`/api/catalog/bundles?q=${encodeURIComponent(q)}`),
  configs: (q = "") => req(`/api/catalog/configs?q=${encodeURIComponent(q)}`),
  aliases: (q = "") => req(`/api/catalog/aliases?q=${encodeURIComponent(q)}`),
  presets: () => req("/api/catalog/presets"),

  steering: () => req("/api/steering"),
  clearSteering: () => req("/api/steering", { method: "DELETE" }),
  removeSteering: (index) => req(`/api/steering/${index}`, { method: "DELETE" }),
  set: (name, level) => req("/api/steering/set", { method: "POST", body: { name, level } }),
  add: (name, level) => req("/api/steering/add", { method: "POST", body: { name, level } }),
  config: (name, level) => req("/api/steering/config", { method: "POST", body: { name, level } }),
  direct: (name, level) => req("/api/steering/direct", { method: "POST", body: { name, level } }),
  inject: (body) => req("/api/steering/inject", { method: "POST", body }),
  dim: (body) => req("/api/steering/dim", { method: "POST", body }),
  noHedge: (scale) => req("/api/steering/no-hedge", { method: "POST", body: { scale } }),
  injectvec: (form) => fetch("/api/steering/injectvec", { method: "POST", body: form }).then(async (r) => {
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || d.error || `HTTP ${r.status}`);
    return d;
  }),

  setConfig: (body) => req("/api/config", { method: "PUT", body }),

  history: () => req("/api/history"),
  resetHistory: () => req("/api/history", { method: "DELETE" }),
  stop: () => req("/api/chat/stop", { method: "POST" }),

  logits: (body) => req("/api/logits", { method: "POST", body }),
  snapshots: () => req("/api/logits/snapshots"),
  snapshot: (name) => req(`/api/logits/snapshots/${encodeURIComponent(name)}`),
};

// Stream a chat completion. onPiece(text) per token; resolves when done; rejects on error.
export async function streamChat({ content, useHum }, onPiece, signal) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, useHum }),
    signal,
  });
  if (!response.ok || !response.body) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || data.error || `HTTP ${response.status}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      let event;
      try {
        event = JSON.parse(payload);
      } catch {
        continue;
      }
      if (event.error) throw new Error(event.error);
      if (event.done) return;
      if (typeof event.content === "string") onPiece(event.content);
    }
  }
}
