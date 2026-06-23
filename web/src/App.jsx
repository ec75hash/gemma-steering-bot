import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import gsap from "gsap";
import {
  Beaker, Bell, BookOpen, ChefHat, CookingPot, Eraser, Flame, FlaskConical, Gauge,
  Layers3, Loader2, Plus, Power, RotateCcw, Salad, SlidersHorizontal,
  Soup, Square, Thermometer, Upload, Utensils, Wand2, X,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { api, streamChat } from "@/lib/api";
import { BeginnerSidebar } from "@/Beginner";
import { STARTERS } from "@/lib/vibes";

function App() {
  const [status, setStatus] = useState(null);
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState(null);
  const [mode, setMode] = useState(() => localStorage.getItem("kitchen-mode") || "beginner");
  const shellRef = useRef(null);
  const endRef = useRef(null);

  const loaded = Boolean(status?.loaded);
  const loading = Boolean(status?.loading);
  const steering = status?.steering || [];

  const notify = useCallback((text, kind = "error") => setToast({ text, kind, id: Math.random() }), []);

  const refresh = useCallback(async () => {
    try {
      setStatus(await api.status());
    } catch (error) {
      setStatus((s) => ({ ...(s || {}), loaded: false, loadError: error.message }));
    }
  }, []);

  // wrap an API action, adopting any returned status payload, surfacing errors
  const run = useCallback(async (fn, okMsg) => {
    try {
      const result = await fn();
      if (result && typeof result === "object" && "steering" in result) setStatus(result);
      else await refresh();
      if (okMsg) notify(okMsg, "ok");
      return result;
    } catch (error) {
      notify(error.message);
      return null;
    }
  }, [refresh, notify]);

  useEffect(() => {
    refresh();
    api.history().then((h) => {
      setMessages(h.map((m, i) => ({ id: `h${i}`, role: m.role, content: m.content })));
    }).catch(() => {});
    const interval = window.setInterval(refresh, 2500);
    return () => window.clearInterval(interval);
  }, [refresh]);

  useEffect(() => {
    if (!toast) return undefined;
    const t = window.setTimeout(() => setToast(null), 4200);
    return () => window.clearTimeout(t);
  }, [toast]);

  useEffect(() => { localStorage.setItem("kitchen-mode", mode); }, [mode]);

  useLayoutEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const ctx = gsap.context(() => {
      gsap.from(".hud-card", { opacity: 0, y: 24, duration: 0.7, ease: "power3.out", stagger: 0.06 });
      gsap.from(".brand-mark", { opacity: 0, x: -16, duration: 0.7, ease: "power3.out" });
      if (reduced) return;
      // slow ken-burns drift on the kitchen photo + flickering range glow
      gsap.fromTo(".kx-photo", { scale: 1.04 }, { scale: 1.12, duration: 34, repeat: -1, yoyo: true, ease: "sine.inOut" });
      gsap.to(".kx-glow", { opacity: 0.62, scale: 1.12, duration: 1.9, repeat: -1, yoyo: true, ease: "sine.inOut", stagger: 0.4 });
    }, shellRef);
    return () => ctx.revert();
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
    const latest = document.querySelector("[data-msg='true']:last-of-type");
    if (latest) {
      gsap.fromTo(latest, { opacity: 0, y: 14 }, { opacity: 1, y: 0, duration: 0.4, ease: "power3.out" });
    }
  }, [messages.length]);

  async function send({ content, useHum }) {
    if (busy) return;
    if (!loaded) { notify("Fire up the range first — load the model."); return; }
    const userText = useHum ? "⟨ hum prompt ⟩" : content;
    const assistantId = `a${Date.now()}`;
    setMessages((m) => [
      ...m,
      { id: `u${Date.now()}`, role: "user", content: userText },
      { id: assistantId, role: "assistant", content: "" },
    ]);
    setBusy(true);
    try {
      await streamChat({ content, useHum }, (piece) => {
        setMessages((m) => m.map((msg) => msg.id === assistantId
          ? { ...msg, content: msg.content + piece } : msg));
      });
    } catch (error) {
      setMessages((m) => m.map((msg) => msg.id === assistantId
        ? { ...msg, role: "error", content: error.message } : msg));
    } finally {
      setBusy(false);
      refresh();
    }
  }

  async function stop() { await api.stop().catch(() => {}); }
  async function newChat() {
    await run(() => api.resetHistory());
    setMessages([]);
  }

  return (
    <div ref={shellRef} className="relative min-h-screen overflow-hidden bg-background text-foreground">
      <KitchenScene />
      <div className="kitchen-vignette" />
      <div className="kitchen-haze" />

      <div className={cn(
        "relative z-10 grid min-h-screen grid-cols-1",
        mode === "lab"
          ? "xl:grid-cols-[minmax(310px,360px)_minmax(0,1fr)_minmax(340px,400px)]"
          : "xl:grid-cols-[minmax(330px,400px)_minmax(0,1fr)]",
      )}>
        {/* LEFT: beginner Vibes, or Lab range/seasoning/steering */}
        {mode === "beginner" ? (
          <BeginnerSidebar status={status} loaded={loaded} loading={loading} run={run} notify={notify} />
        ) : (
          <aside className="flex min-h-0 flex-col gap-4 border-b border-border bg-black/45 p-4 backdrop-blur-md xl:h-screen xl:overflow-auto xl:border-b-0 xl:border-r">
            <header className="brand-mark flex items-center gap-3">
              <div className="grid h-12 w-12 place-items-center rounded-md border border-primary bg-primary/15 text-primary shadow-[0_0_28px_rgba(240,104,58,0.32)]">
                <ChefHat className="size-6" />
              </div>
              <div className="min-w-0">
                <h1 className="text-xl font-black uppercase leading-none text-primary kitchen-glow">
                  Gemma's Test Kitchen
                </h1>
                <div className="mt-1 text-[11px] font-bold uppercase text-muted-foreground">
                  SAE steering line · {status?.model || "gemma-3-4b"}
                </div>
              </div>
            </header>

            <RangePanel status={status} loading={loading} loaded={loaded} run={run} />
            <SeasoningPanel status={status} run={run} disabled={!status} />
            <ActiveSteering steering={steering} run={run} />
          </aside>
        )}

        {/* CENTER: chat (shared by both modes) */}
        <ChatColumn
          mode={mode} setMode={setMode}
          messages={messages} busy={busy} loaded={loaded} loading={loading}
          status={status} steering={steering} endRef={endRef}
          onSend={send} onStop={stop} onNew={newChat}
        />

        {/* RIGHT: the pantry (Lab only) */}
        {mode === "lab" && (
          <aside className="min-h-0 border-t border-border bg-black/45 p-4 backdrop-blur-md xl:h-screen xl:overflow-auto xl:border-l xl:border-t-0">
            <Pantry status={status} run={run} notify={notify} />
          </aside>
        )}
      </div>

      {toast && (
        <div className={cn(
          "fixed bottom-4 left-1/2 z-50 -translate-x-1/2 rounded-md border px-4 py-2 text-sm font-bold shadow-lg backdrop-blur-md",
          toast.kind === "ok" ? "border-basil/60 bg-basil/15 text-basil" : "border-destructive/60 bg-destructive/15 text-destructive",
        )}>
          {toast.text}
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------- shared chat + mode toggle */

function ModeToggle({ mode, setMode }) {
  return (
    <div className="flex items-center rounded-md border border-border bg-black/40 p-0.5 text-[10px] font-black uppercase">
      <button type="button" onClick={() => setMode("beginner")}
        className={cn("flex items-center gap-1 rounded px-2 py-1 transition-colors",
          mode === "beginner" ? "bg-primary/20 text-primary" : "text-muted-foreground hover:text-foreground")}>
        <Wand2 className="size-3.5" /> Vibes
      </button>
      <button type="button" onClick={() => setMode("lab")}
        className={cn("flex items-center gap-1 rounded px-2 py-1 transition-colors",
          mode === "lab" ? "bg-primary/20 text-primary" : "text-muted-foreground hover:text-foreground")}>
        <FlaskConical className="size-3.5" /> Lab
      </button>
    </div>
  );
}

function ChatColumn({ mode, setMode, messages, busy, loaded, loading, status, steering, endRef, onSend, onStop, onNew }) {
  function submitPrompt(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const value = String(new FormData(form).get("prompt") || "").trim();
    if (!value) return;
    form.reset();
    onSend({ content: value, useHum: false });
  }

  return (
    <main className="grid min-h-[70vh] grid-rows-[auto_minmax(0,1fr)_auto] xl:h-screen xl:min-h-0">
      <header className="flex flex-col items-stretch justify-between gap-3 border-b border-border bg-black/55 px-4 py-3 backdrop-blur-md md:flex-row md:items-center md:px-6">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-black uppercase text-primary kitchen-glow">The Pass</h2>
            <StatusChip loaded={loaded} loading={loading} />
            {steering.length > 0 && <Badge variant="amber">{steering.length} steering</Badge>}
            {status?.busy && <Badge variant="amber">cooking</Badge>}
          </div>
          <p className="mt-0.5 break-anywhere text-[11px] font-bold text-muted-foreground">
            {status?.model} · {status?.promptMode} prompt · temp {status?.temp} ·{" "}
            {status?.tokens}{status?.soft ? " soft" : ""} tok · seed {status?.seed ?? "random"}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <ModeToggle mode={mode} setMode={setMode} />
          <div className="grid grid-cols-3 gap-2 md:flex">
            <Button variant="secondary" size="sm" onClick={() => onSend({ useHum: true })} disabled={busy || !loaded} type="button">
              <Soup className="size-4" /> Hum
            </Button>
            <Button variant="secondary" size="sm" onClick={onNew} type="button">
              <RotateCcw className="size-4" /> New
            </Button>
            <Button variant="destructive" size="sm" onClick={onStop} disabled={!busy} type="button">
              <Square className="size-4" /> Stop
            </Button>
          </div>
        </div>
      </header>

      <section className="relative min-h-0 overflow-hidden">
        <ScrollArea className="h-full">
          <div className="flex min-h-full flex-col gap-4 p-4 md:p-6">
            {messages.length === 0
              ? <EmptyState loaded={loaded} beginner={mode === "beginner"} onStarter={(t) => onSend({ content: t })} />
              : messages.map((m) => <Message key={m.id} message={m} />)}
            <div ref={endRef} />
          </div>
        </ScrollArea>
      </section>

      <form onSubmit={submitPrompt} className="border-t border-border bg-black/60 p-3 backdrop-blur-md md:p-4">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-[minmax(0,1fr)_120px]">
          <Textarea
            name="prompt" rows={3} placeholder="What are we cooking? (Enter to fire, Shift+Enter for a newline)"
            disabled={busy} className="min-h-[84px] resize-y text-base"
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); e.currentTarget.form?.requestSubmit(); }
            }}
          />
          <Button className="h-full min-h-12" type="submit" disabled={busy || !loaded}>
            {busy ? <Loader2 className="size-4 animate-spin" /> : <Flame className="size-4" />} Fire
          </Button>
        </div>
      </form>
    </main>
  );
}

/* ------------------------------------------------------------------ panels */

function RangePanel({ status, loading, loaded, run }) {
  const [model, setModel] = useState("it");
  const [promptMode, setPromptMode] = useState("auto");
  useEffect(() => { if (status?.modelKey) setModel(status.modelKey); }, [status?.modelKey]);

  return (
    <Card className="hud-card relative shrink-0">
      <div className="warm-sweep" />
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Flame className="size-4 text-primary" /> The Range</CardTitle>
        <CardDescription>{loaded ? "burners lit" : loading ? "heating up…" : "cold — fire it up"}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <Field label="Model">
            <Select value={model} onValueChange={setModel}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {(status?.modelChoices || ["it", "base"]).map((m) => (
                  <SelectItem key={m} value={m}>{m === "it" ? "gemma-3-4b-it" : "gemma-3-4b-base"}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          <Field label="Prompt">
            <Select value={promptMode} onValueChange={setPromptMode}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">auto</SelectItem>
                <SelectItem value="chat">chat</SelectItem>
                <SelectItem value="raw">raw</SelectItem>
              </SelectContent>
            </Select>
          </Field>
        </div>
        <Button className="w-full" disabled={loading} onClick={() => run(() => api.load(model, promptMode))} type="button">
          {loading ? <Loader2 className="size-4 animate-spin" /> : <Power className="size-4" />}
          {loaded ? "Reload Range" : "Fire Up The Range"}
        </Button>
        {status?.loadError && <p className="text-xs font-bold text-destructive">{status.loadError}</p>}
        <div className="grid grid-cols-2 gap-2 text-[11px] font-bold uppercase text-muted-foreground">
          <Stat label="SAE layers" value={(status?.saeLayers || []).join(" / ") || "—"} />
          <Stat label="Turns" value={status?.turns ?? 0} />
        </div>
      </CardContent>
    </Card>
  );
}

function SeasoningPanel({ status, run, disabled }) {
  const commit = (body) => run(() => api.setConfig(body));
  return (
    <Card className="hud-card shrink-0">
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Thermometer className="size-4 text-amber" /> Seasoning</CardTitle>
        <CardDescription>heat, salt &amp; timing</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <SliderField label="Temperature" suffix="°" value={Number(status?.temp ?? 0.9)} min={0} max={2} step={0.05}
          disabled={disabled} onCommit={(v) => commit({ temp: v })} />
        <div className="grid grid-cols-2 gap-3">
          <NumField label="Max tokens" value={status?.tokens ?? 150} min={1} max={4096} disabled={disabled}
            onCommit={(v) => commit({ tokens: v })} />
          <Field label="Token cap">
            <Select value={status?.soft ? "soft" : "hard"} onValueChange={(v) => commit({ soft: v === "soft" })}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="hard">hard</SelectItem>
                <SelectItem value="soft">soft (finish sentence)</SelectItem>
              </SelectContent>
            </Select>
          </Field>
          <NumField label="Seed" value={status?.seed ?? ""} placeholder="random" disabled={disabled}
            onCommit={(v) => commit({ seed: v === "" ? null : v })} />
          <Field label="Context turns">
            <Input defaultValue={status?.contextTurns ?? "all"} key={String(status?.contextTurns)} placeholder="all" disabled={disabled}
              onBlur={(e) => commit({ contextTurns: e.target.value.trim() || "all" })}
              onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()} />
          </Field>
        </div>
        <Field label="Steering phase">
          <Select value={status?.phase ?? "both"} onValueChange={(v) => commit({ phase: v })}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="both">both (prefill + decode)</SelectItem>
              <SelectItem value="prefill">prefill only</SelectItem>
              <SelectItem value="decode">decode only</SelectItem>
            </SelectContent>
          </Select>
        </Field>
      </CardContent>
    </Card>
  );
}

function ActiveSteering({ steering, run }) {
  return (
    <Card className="hud-card min-h-[180px] shrink-0">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2"><SlidersHorizontal className="size-4 text-primary" /> On The Pass</CardTitle>
          <Button variant="ghost" size="sm" disabled={!steering.length} onClick={() => run(() => api.clearSteering())} type="button">
            <Eraser className="size-4" /> Clear
          </Button>
        </div>
        <CardDescription>{steering.length ? `${steering.length} active op${steering.length > 1 ? "s" : ""}` : "no steering — plain Gemma"}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        {steering.map((s, i) => (
          <div key={i} className="flex items-start gap-2 rounded-md border border-border bg-black/40 p-2 text-xs">
            <Badge variant={s.kind === "dim" ? "amber" : "default"} className="shrink-0">{s.kind}</Badge>
            <div className="min-w-0 flex-1 font-bold">
              <div className="text-foreground">L{s.layer} {s.alias ? `· ${s.alias}` : ""}{" "}
                <span className="text-muted-foreground">
                  {s.kind === "dim" ? `×${s.scale}` : `@ ${s.strength ?? ""}`}
                </span>
              </div>
              <div className="break-anywhere text-[10px] text-muted-foreground">
                {s.label || `f${(s.features || []).slice(0, 8).join(",")}`}
              </div>
            </div>
            <button className="shrink-0 text-muted-foreground hover:text-destructive" title="remove"
              onClick={() => run(() => api.removeSteering(i))}>
              <X className="size-4" />
            </button>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

/* --------------------------------------------------------------- the pantry */

const PANTRY_TABS = [
  { key: "recipes", label: "Recipes", icon: BookOpen },
  { key: "spices", label: "Spice Rack", icon: Salad },
  { key: "handmix", label: "Hand-Mix", icon: Wand2 },
  { key: "tasting", label: "Tasting", icon: Beaker },
];

function Pantry({ status, run, notify }) {
  const [tab, setTab] = useState("recipes");
  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex items-center gap-2">
        <CookingPot className="size-5 text-primary" />
        <h2 className="text-lg font-black uppercase text-primary kitchen-glow">The Pantry</h2>
      </div>
      <div className="grid grid-cols-4 gap-1 rounded-md border border-border bg-black/40 p-1">
        {PANTRY_TABS.map(({ key, label, icon: Icon }) => (
          <button key={key} onClick={() => setTab(key)}
            className={cn("flex flex-col items-center gap-1 rounded px-1 py-1.5 text-[10px] font-black uppercase transition-colors",
              tab === key ? "bg-primary/18 text-primary" : "text-muted-foreground hover:text-foreground")}>
            <Icon className="size-4" /> {label}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1">
        {tab === "recipes" && <RecipesTab run={run} />}
        {tab === "spices" && <SpicesTab run={run} />}
        {tab === "handmix" && <HandMixTab status={status} run={run} />}
        {tab === "tasting" && <TastingTab notify={notify} />}
      </div>
    </div>
  );
}

function RecipesTab({ run }) {
  const [q, setQ] = useState("");
  const [rows, setRows] = useState([]);
  useEffect(() => { api.configs(q).then(setRows).catch(() => {}); }, [q]);
  return (
    <PantryList search={q} onSearch={setQ} placeholder="filter recipes…">
      {rows.map((c) => (
        <ItemRow key={c.name} title={c.name} subtitle={c.label}
          meta={c.steps.map((s) => `${s.target} ${s.level}`).join(" + ")}>
          <LevelApply defaultLevel={c.default} label="Cook"
            onApply={(lvl) => run(() => api.config(c.name, lvl), `cooked ${c.name}`)} />
        </ItemRow>
      ))}
    </PantryList>
  );
}

function SpicesTab({ run }) {
  const [q, setQ] = useState("");
  const [features, setFeatures] = useState([]);
  const [bundles, setBundles] = useState([]);
  useEffect(() => {
    api.features(q).then(setFeatures).catch(() => {});
    api.bundles(q).then(setBundles).catch(() => {});
  }, [q]);
  return (
    <PantryList search={q} onSearch={setQ} placeholder="filter spices & blends…">
      {bundles.length > 0 && <SectionLabel>Blends (bundles)</SectionLabel>}
      {bundles.map((b) => (
        <ItemRow key={`b-${b.name}`} title={b.name} badge={b.mode}
          subtitle={b.label} meta={`L${b.layers.join("/L")} · default ${b.default}`}>
          <LevelApply defaultLevel={b.default} label="Set"
            onApply={(lvl) => run(() => api.set(b.name, lvl), `set ${b.name}`)}
            secondaryLabel="Add" onSecondary={(lvl) => run(() => api.add(b.name, lvl), `added ${b.name}`)} />
        </ItemRow>
      ))}
      {features.length > 0 && <SectionLabel>Spices (single features)</SectionLabel>}
      {features.map((f) => (
        <ItemRow key={`f-${f.name}`} title={f.name} badge={f.ready ? null : "no SAE"}
          subtitle={f.label} meta={`L${f.layer}:${f.feature} · str ${f.strength}`}>
          <LevelApply defaultLevel={f.strength} step={10} label="Set"
            onApply={(lvl) => run(() => api.set(f.name, lvl), `set ${f.name}`)}
            secondaryLabel="Add" onSecondary={(lvl) => run(() => api.add(f.name, lvl), `added ${f.name}`)} />
        </ItemRow>
      ))}
    </PantryList>
  );
}

function HandMixTab({ status, run }) {
  const layers = status?.saeLayers || [9, 17, 22, 29];
  return (
    <ScrollArea className="h-full">
      <div className="space-y-4 pr-2">
        <MiniCard icon={Eraser} title="No-hedge (ablate negation)">
          <LevelApply defaultLevel={0} step={0.1} label="Apply"
            onApply={(v) => run(() => api.noHedge(v ?? 0), `no-hedge ${v ?? 0}`)} hint="0 = off · 0.5 = soften" />
        </MiniCard>

        <MiniCard icon={Gauge} title="Dim / boost preset">
          <PresetDim run={run} />
        </MiniCard>

        <MiniCard icon={Layers3} title="Raw dim (layer · features · scale)">
          <RawDim layers={layers} run={run} />
        </MiniCard>

        <MiniCard icon={Plus} title="Raw inject (layer · feature · strength)">
          <RawInject layers={layers} run={run} />
        </MiniCard>

        <MiniCard icon={Upload} title="Inject decoder vector (.safetensors)">
          <InjectVec layers={layers} run={run} />
        </MiniCard>
      </div>
    </ScrollArea>
  );
}

function PresetDim({ run }) {
  const [preset, setPreset] = useState("carriers");
  const [scale, setScale] = useState("0.8");
  return (
    <div className="flex items-end gap-2">
      <Field label="Preset" className="flex-1">
        <Select value={preset} onValueChange={setPreset}>
          <SelectTrigger><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="carriers">carriers</SelectItem>
            <SelectItem value="negation">negation</SelectItem>
          </SelectContent>
        </Select>
      </Field>
      <Field label="Scale" className="w-20"><Input value={scale} onChange={(e) => setScale(e.target.value)} /></Field>
      <Button size="sm" onClick={() => run(() => api.dim({ preset, scale: Number(scale) }), `dim ${preset}`)} type="button">Apply</Button>
    </div>
  );
}

function RawDim({ layers, run }) {
  const [layer, setLayer] = useState(String(layers[1] ?? layers[0] ?? 17));
  const [feats, setFeats] = useState("");
  const [scale, setScale] = useState("0");
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-3 gap-2">
        <Field label="Layer"><LayerSelect layers={layers} value={layer} onChange={setLayer} /></Field>
        <Field label="Scale"><Input value={scale} onChange={(e) => setScale(e.target.value)} /></Field>
        <div className="self-end">
          <Button size="sm" className="w-full" type="button"
            onClick={() => run(() => api.dim({ layer: Number(layer), features: csv(feats), scale: Number(scale) }), "dim applied")}>
            Apply
          </Button>
        </div>
      </div>
      <Field label="Features (comma-sep)"><Input value={feats} onChange={(e) => setFeats(e.target.value)} placeholder="4150, 15673, …" /></Field>
    </div>
  );
}

function RawInject({ layers, run }) {
  const [layer, setLayer] = useState(String(layers[1] ?? layers[0] ?? 17));
  const [feature, setFeature] = useState("");
  const [strength, setStrength] = useState("800");
  return (
    <div className="grid grid-cols-3 gap-2">
      <Field label="Layer"><LayerSelect layers={layers} value={layer} onChange={setLayer} /></Field>
      <Field label="Feature"><Input value={feature} onChange={(e) => setFeature(e.target.value)} placeholder="1087" /></Field>
      <Field label="Strength"><Input value={strength} onChange={(e) => setStrength(e.target.value)} /></Field>
      <div className="col-span-3">
        <Button size="sm" className="w-full" type="button"
          onClick={() => run(() => api.inject({ layer: Number(layer), feature: Number(feature), strength: Number(strength) }), "injected")}>
          Inject
        </Button>
      </div>
    </div>
  );
}

function InjectVec({ layers, run }) {
  const [layer, setLayer] = useState(String(layers[1] ?? layers[0] ?? 17));
  const [strength, setStrength] = useState("1");
  const [file, setFile] = useState(null);
  function apply() {
    if (!file) return;
    const form = new FormData();
    form.append("layer", layer);
    form.append("strength", strength);
    form.append("file", file);
    run(() => api.injectvec(form), `injected ${file.name}`);
  }
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 gap-2">
        <Field label="Layer"><LayerSelect layers={layers} value={layer} onChange={setLayer} /></Field>
        <Field label="Strength"><Input value={strength} onChange={(e) => setStrength(e.target.value)} /></Field>
      </div>
      <input type="file" accept=".safetensors" onChange={(e) => setFile(e.target.files?.[0] || null)}
        className="block w-full text-xs text-muted-foreground file:mr-2 file:rounded file:border file:border-border file:bg-secondary file:px-2 file:py-1 file:text-xs file:font-bold file:text-foreground" />
      <Button size="sm" className="w-full" type="button" disabled={!file} onClick={apply}>Inject vector</Button>
    </div>
  );
}

function TastingTab({ notify }) {
  const [source, setSource] = useState("current");
  const [topN, setTopN] = useState("20");
  const [saveIt, setSaveIt] = useState(false);
  const [result, setResult] = useState(null);
  const [snaps, setSnaps] = useState([]);
  const [busy, setBusy] = useState(false);
  const loadSnaps = () => api.snapshots().then(setSnaps).catch(() => {});
  useEffect(() => { loadSnaps(); }, []);
  async function taste() {
    setBusy(true);
    try {
      setResult(await api.logits({ topN: Number(topN), source, save: saveIt }));
      if (saveIt) loadSnaps();
    } catch (e) { notify(e.message); } finally { setBusy(false); }
  }
  return (
    <ScrollArea className="h-full">
      <div className="space-y-3 pr-2">
        <MiniCard icon={Beaker} title="Next-token tasting (logit lens)">
          <div className="grid grid-cols-2 gap-2">
            <Field label="Prompt">
              <Select value={source} onValueChange={setSource}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="current">current chat</SelectItem>
                  <SelectItem value="hum">hum prompt</SelectItem>
                </SelectContent>
              </Select>
            </Field>
            <Field label="Top N"><Input value={topN} onChange={(e) => setTopN(e.target.value)} /></Field>
          </div>
          <label className="flex items-center gap-2 text-[11px] font-bold uppercase text-muted-foreground">
            <input type="checkbox" checked={saveIt} onChange={(e) => setSaveIt(e.target.checked)} /> save snapshot to disk
          </label>
          <Button size="sm" className="w-full" disabled={busy} onClick={taste} type="button">
            {busy ? <Loader2 className="size-4 animate-spin" /> : <Beaker className="size-4" />} Taste
          </Button>
        </MiniCard>

        {result && (
          <div className="rounded-md border border-border bg-black/40 p-2 text-xs">
            <div className="mb-1 font-bold text-muted-foreground">
              {result.prompt_source} · {result.prompt_token_count} tok · {result.steering?.length || 0} steering
              {result.saved_path && <span className="text-basil"> · saved</span>}
            </div>
            <table className="w-full font-mono text-[11px]">
              <thead className="text-muted-foreground">
                <tr><th className="text-left">#</th><th className="text-left">token</th><th className="text-right">logit</th><th className="text-right">prob</th></tr>
              </thead>
              <tbody>
                {result.top_tokens.map((t) => (
                  <tr key={t.rank} className="border-t border-border/50">
                    <td className="text-muted-foreground">{t.rank}</td>
                    <td className="break-anywhere text-primary">{JSON.stringify(t.token_text)}</td>
                    <td className="text-right">{t.logit.toFixed(3)}</td>
                    <td className="text-right text-amber">{(t.probability * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {snaps.length > 0 && (
          <MiniCard icon={FlaskConical} title={`Saved snapshots (${snaps.length})`}>
            <div className="space-y-1">
              {snaps.slice(0, 12).map((s) => (
                <button key={s.name} onClick={() => api.snapshot(s.name).then(setResult).catch((e) => notify(e.message))}
                  className="block w-full break-anywhere rounded border border-border bg-black/30 px-2 py-1 text-left text-[10px] font-bold text-muted-foreground hover:text-foreground">
                  {s.name}
                </button>
              ))}
            </div>
          </MiniCard>
        )}
      </div>
    </ScrollArea>
  );
}

/* ----------------------------------------------------------- small helpers */

function Field({ label, children, className }) {
  return (
    <label className={cn("grid gap-1.5 text-[11px] font-black uppercase text-muted-foreground", className)}>
      <span>{label}</span>
      {children}
    </label>
  );
}

function NumField({ label, value, onCommit, placeholder, min, max, disabled }) {
  return (
    <Field label={label}>
      <Input type="number" min={min} max={max} placeholder={placeholder} disabled={disabled}
        defaultValue={value} key={String(value)}
        onBlur={(e) => onCommit(e.target.value === "" ? "" : Number(e.target.value))}
        onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()} />
    </Field>
  );
}

function SliderField({ label, value, min, max, step, onCommit, suffix, disabled }) {
  const [local, setLocal] = useState(value);
  useEffect(() => setLocal(value), [value]);
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-[11px] font-black uppercase text-muted-foreground">
        <span>{label}</span>
        <span className="font-mono text-primary kitchen-glow">{Number(local).toFixed(2)}{suffix}</span>
      </div>
      <Slider value={[Number(local)]} min={min} max={max} step={step} disabled={disabled}
        onValueChange={([v]) => setLocal(v)} onValueCommit={([v]) => onCommit(v)} />
    </div>
  );
}

function LayerSelect({ layers, value, onChange }) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger><SelectValue /></SelectTrigger>
      <SelectContent>{layers.map((l) => <SelectItem key={l} value={String(l)}>L{l}</SelectItem>)}</SelectContent>
    </Select>
  );
}

function LevelApply({ defaultLevel, onApply, label = "Apply", secondaryLabel, onSecondary, hint }) {
  const [level, setLevel] = useState(String(defaultLevel ?? ""));
  return (
    <div className="space-y-1">
      <div className="flex items-end gap-2">
        <Input className="h-8 w-20 text-xs" value={level} onChange={(e) => setLevel(e.target.value)} placeholder="level" />
        <Button size="sm" type="button" onClick={() => onApply(level === "" ? undefined : Number(level))}>{label}</Button>
        {secondaryLabel && (
          <Button size="sm" variant="secondary" type="button" onClick={() => onSecondary(level === "" ? undefined : Number(level))}>{secondaryLabel}</Button>
        )}
      </div>
      {hint && <p className="text-[10px] font-bold text-muted-foreground">{hint}</p>}
    </div>
  );
}

function PantryList({ search, onSearch, placeholder, children }) {
  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <Input value={search} onChange={(e) => onSearch(e.target.value)} placeholder={placeholder} className="h-9 shrink-0" />
      <ScrollArea className="min-h-0 flex-1"><div className="space-y-2 pr-2">{children}</div></ScrollArea>
    </div>
  );
}

function SectionLabel({ children }) {
  return <div className="pt-1 text-[10px] font-black uppercase tracking-wide text-amber">{children}</div>;
}

function ItemRow({ title, subtitle, meta, badge, children }) {
  return (
    <div className="rounded-md border border-border bg-black/40 p-2.5">
      <div className="flex items-center gap-2">
        <span className="font-black uppercase text-foreground">{title}</span>
        {badge && <Badge variant="amber" className="text-[9px]">{badge}</Badge>}
      </div>
      {subtitle && <p className="mt-0.5 break-anywhere text-[11px] font-bold text-muted-foreground">{subtitle}</p>}
      {meta && <p className="mt-0.5 break-anywhere font-mono text-[10px] text-muted-foreground/80">{meta}</p>}
      <div className="mt-2">{children}</div>
    </div>
  );
}

function MiniCard({ icon: Icon, title, children }) {
  return (
    <div className="rounded-md border border-border bg-black/40 p-3">
      <div className="mb-2 flex items-center gap-2 text-xs font-black uppercase text-foreground">
        <Icon className="size-4 text-primary" /> {title}
      </div>
      <div className="space-y-2">{children}</div>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="rounded border border-border bg-black/30 px-2 py-1.5">
      <div className="text-[9px] text-muted-foreground">{label}</div>
      <div className="font-mono text-xs text-foreground">{value}</div>
    </div>
  );
}

function StatusChip({ loaded, loading }) {
  const variant = loaded ? "default" : loading ? "amber" : "destructive";
  return <Badge variant={variant}>{loaded ? "open" : loading ? "heating" : "closed"}</Badge>;
}

function Message({ message }) {
  const isUser = message.role === "user";
  const isError = message.role === "error";
  return (
    <article data-msg="true" className={cn("grid max-w-[min(760px,94%)] gap-1.5", isUser ? "self-end" : "self-start")}>
      <div className={cn("flex items-center gap-2 text-[11px] font-black uppercase text-muted-foreground", isUser && "justify-end")}>
        {isUser ? <Utensils className="size-3.5 text-amber" /> : <ChefHat className="size-3.5 text-primary" />}
        {isUser ? "Order" : isError ? "Burned" : "Gemma"}
      </div>
      <div className={cn(
        "rounded-md border bg-card/85 p-3.5 text-sm font-medium leading-relaxed shadow-[0_16px_36px_rgba(0,0,0,0.30)] backdrop-blur-md",
        isUser && "border-amber/45 bg-amber/12 text-amber-foreground",
        !isUser && !isError && "border-primary/30 text-foreground",
        isError && "border-destructive/60 bg-destructive/12 text-destructive",
      )}>
        <div className="whitespace-pre-wrap break-words">{message.content || (isUser ? "" : "…")}</div>
      </div>
    </article>
  );
}

function EmptyState({ loaded, beginner, onStarter }) {
  return (
    <div className="grid min-h-[55vh] place-items-center">
      <div className="hud-card relative w-full max-w-md overflow-hidden rounded-md border border-primary/25 bg-black/50 p-8 text-center backdrop-blur-lg">
        <div className="mx-auto mb-4 grid h-16 w-16 place-items-center rounded-full border border-primary bg-primary/10 shadow-[0_0_40px_rgba(240,104,58,0.25)]">
          <Bell className="size-8 text-primary" />
        </div>
        <div className="text-xl font-black uppercase text-primary kitchen-glow">{loaded ? "Kitchen's open" : "Kitchen's cold"}</div>
        <div className="mt-1 text-sm font-bold text-muted-foreground">
          {loaded
            ? beginner ? "Pick a vibe on the left, then try a question:" : "Send an order, season with steering from the pantry."
            : beginner ? "Hit Start Gemma on the left to begin." : "Fire up the range to start the line."}
        </div>
        {loaded && beginner && onStarter && (
          <div className="mt-4 flex flex-wrap justify-center gap-2">
            {STARTERS.map((s) => (
              <button key={s} type="button" onClick={() => onStarter(s)}
                className="rounded-full border border-border bg-black/40 px-3 py-1 text-left text-[11px] font-bold text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground">
                {s}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------- decorative kitchen scene */

function KitchenScene() {
  return (
    <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden" aria-hidden="true">
      {/* real kitchen photo (Pexels, free) with a slow ken-burns drift */}
      <div className="kx-photo absolute inset-0 bg-cover bg-center will-change-transform"
        style={{ backgroundImage: "url(/scene/kitchen.jpg)" }} />
      {/* warm wash: pull the photo toward the flame/copper palette + darken for contrast */}
      <div className="absolute inset-0" style={{
        background:
          "radial-gradient(circle at 68% 58%, rgba(240,120,50,0.22), transparent 46%), " +
          "linear-gradient(180deg, rgba(40,22,8,0.34), rgba(18,11,5,0.5))",
        mixBlendMode: "multiply",
      }} />
      <div className="absolute inset-0 bg-background/25" />
      {/* lit-range glow — positions tuned to kitchen.jpg (stove sits center-right) */}
      <div className="kx-glow absolute h-44 w-44 rounded-full"
        style={{ left: "58%", top: "50%", background: "radial-gradient(circle, rgba(255,150,60,0.42), transparent 65%)" }} />
      <div className="kx-glow absolute h-24 w-24 rounded-full"
        style={{ left: "6%", top: "26%", background: "radial-gradient(circle, rgba(255,214,150,0.35), transparent 70%)" }} />
    </div>
  );
}

function csv(text) {
  return String(text).split(",").map((x) => parseInt(x.trim(), 10)).filter((n) => Number.isFinite(n));
}

export default App;
