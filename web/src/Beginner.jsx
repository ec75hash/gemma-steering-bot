// Beginner ("Vibes") sidebar for Gemma's Test Kitchen.
//
// Zero mech-interp knowledge required: fire up Gemma, click a vibe, chat. Each vibe is a
// vivid persona that maps onto a real steering preset (see web/src/lib/vibes.js); the
// "under the hood" reveal shows the actual feature family so a curious beginner picks up
// the real vocabulary. The chat column itself is shared with Lab mode (rendered by App).

import { useEffect, useState } from "react";
import {
  ChefHat, ChevronDown, Dice5, Eraser, HelpCircle, Power, Loader2, Sparkles, Thermometer, X,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import { VIBES, VIBES_BY_ID } from "@/lib/vibes";

const HELP_KEY = "kitchen-help-dismissed";

export function BeginnerSidebar({ status, loaded, loading, run, notify }) {
  // which vibe is "on", and whether we've temporarily flipped to plain Gemma to compare
  const [active, setActive] = useState(null); // { id, levelLabel }
  const [plain, setPlain] = useState(false);
  const [showHelp, setShowHelp] = useState(() => localStorage.getItem(HELP_KEY) !== "1");

  const steerCount = status?.steering?.length || 0;
  const activeVibe = active ? VIBES_BY_ID[active.id] : null;

  function dismissHelp() {
    localStorage.setItem(HELP_KEY, "1");
    setShowHelp(false);
  }

  async function applyVibe(vibe, levelLabel) {
    if (!loaded) { notify("Fire up Gemma first — hit Start Gemma."); return; }
    const level = vibe.levels[levelLabel];
    const result = await run(async () => {
      await api.clearSteering();
      return vibe.apply(api, level);
    }, `${vibe.title} — on`);
    if (result) { setActive({ id: vibe.id, levelLabel }); setPlain(false); }
  }

  async function goPlain() {
    await run(() => api.clearSteering(), "back to plain Gemma");
    setActive(null);
    setPlain(false);
  }

  async function togglePlain() {
    if (!activeVibe) return;
    if (!plain) {
      const r = await run(() => api.clearSteering());
      if (r) setPlain(true);
    } else {
      const level = activeVibe.levels[active.levelLabel];
      const r = await run(async () => {
        await api.clearSteering();
        return activeVibe.apply(api, level);
      });
      if (r) setPlain(false);
    }
  }

  function surprise() {
    const vibe = VIBES[Math.floor(Math.random() * VIBES.length)];
    applyVibe(vibe, vibe.defaultLevel);
  }

  return (
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
            click a vibe · rewire how Gemma talks
          </div>
        </div>
      </header>

      {showHelp && <WhatIsThis onClose={dismissHelp} />}

      {/* Start Gemma */}
      <Card className="hud-card relative shrink-0">
        <div className="warm-sweep" />
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Power className="size-4 text-primary" /> Start Gemma</CardTitle>
          <CardDescription>
            {loaded ? "ready — pick a vibe and chat" : loading ? "warming up… (~20s)" : "not running yet"}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          <Button className="w-full" disabled={loading} type="button"
            onClick={() => run(() => api.load("it", "auto"), "Gemma is warming up…")}>
            {loading ? <Loader2 className="size-4 animate-spin" /> : <Power className="size-4" />}
            {loaded ? "Restart Gemma" : "Start Gemma"}
          </Button>
          {status?.loadError && <p className="text-xs font-bold text-destructive">{status.loadError}</p>}
        </CardContent>
      </Card>

      {/* On now */}
      <Card className="hud-card shrink-0">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2"><Sparkles className="size-4 text-primary" /> On Now</CardTitle>
            <Button variant="ghost" size="sm" disabled={!steerCount && !active} onClick={goPlain} type="button">
              <Eraser className="size-4" /> Plain Gemma
            </Button>
          </div>
          <CardDescription>
            {activeVibe ? (plain ? `${activeVibe.title} — paused (plain)` : `${activeVibe.emoji} ${activeVibe.title}`)
              : steerCount ? `${steerCount} custom steering op${steerCount > 1 ? "s" : ""}` : "plain Gemma — no steering"}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="grid grid-cols-2 gap-2">
            <Button variant="secondary" size="sm" onClick={surprise} disabled={!loaded} type="button">
              <Dice5 className="size-4" /> Surprise me
            </Button>
            <Button variant={plain ? "amber" : "secondary"} size="sm" onClick={togglePlain} disabled={!activeVibe} type="button">
              {plain ? "Show steered" : "Compare: plain"}
            </Button>
          </div>
          {activeVibe && (
            <p className="text-[10px] font-bold text-muted-foreground">
              Tip: ask the same question, hit <span className="text-amber">Compare: plain</span>, ask again — watch it change.
            </p>
          )}
        </CardContent>
      </Card>

      {/* Vibes */}
      <div className="min-h-0 flex-1">
        <div className="mb-2 flex items-center gap-2">
          <Sparkles className="size-5 text-primary" />
          <h2 className="text-lg font-black uppercase text-primary kitchen-glow">Vibes</h2>
        </div>
        <div className="space-y-2">
          {VIBES.map((vibe) => (
            <VibeCard key={vibe.id} vibe={vibe}
              active={!plain && active?.id === vibe.id}
              activeLevelLabel={active?.id === vibe.id ? active.levelLabel : null}
              onApply={applyVibe} />
          ))}
        </div>
      </div>

      {/* Creativity (temperature) */}
      <Card className="hud-card shrink-0">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Thermometer className="size-4 text-amber" /> Creativity</CardTitle>
          <CardDescription>lower = focused · higher = wild</CardDescription>
        </CardHeader>
        <CardContent>
          <CreativitySlider status={status} run={run} />
        </CardContent>
      </Card>
    </aside>
  );
}

function VibeCard({ vibe, active, activeLevelLabel, onApply }) {
  return (
    <div className={cn(
      "vibe-card rounded-md border bg-black/40 p-3 transition-colors",
      active ? "border-primary bg-primary/10" : "border-border hover:border-primary/50",
    )}>
      <div className="flex items-start gap-2">
        <span className="text-xl leading-none">{vibe.emoji}</span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-black uppercase text-foreground">{vibe.title}</span>
            {active && <Badge variant="default" className="text-[9px]">on</Badge>}
          </div>
          <p className="mt-0.5 text-[11px] font-bold leading-snug text-muted-foreground">{vibe.blurb}</p>
        </div>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <span className="text-[10px] font-black uppercase text-muted-foreground">Strength</span>
        {Object.keys(vibe.levels).map((label) => (
          <button key={label} type="button" onClick={() => onApply(vibe, label)}
            className={cn(
              "rounded border px-2 py-0.5 text-[10px] font-black uppercase transition-colors",
              active && activeLevelLabel === label
                ? "border-primary bg-primary/25 text-primary"
                : "border-border bg-black/30 text-muted-foreground hover:border-primary/50 hover:text-foreground",
            )}>
            {label}
          </button>
        ))}
      </div>

      <details className="vibe-hood mt-2">
        <summary className="flex cursor-pointer list-none items-center gap-1 text-[10px] font-bold uppercase text-muted-foreground/80 hover:text-foreground">
          <ChevronDown className="size-3 transition-transform" /> under the hood
        </summary>
        <p className="mt-1 break-anywhere font-mono text-[10px] leading-relaxed text-muted-foreground/80">{vibe.underHood}</p>
      </details>
    </div>
  );
}

function CreativitySlider({ status, run }) {
  const [local, setLocal] = useState(Number(status?.temp ?? 0.9));
  useEffect(() => { setLocal(Number(status?.temp ?? 0.9)); }, [status?.temp]);
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-[11px] font-black uppercase text-muted-foreground">
        <span>Temperature</span>
        <span className="font-mono text-primary kitchen-glow">{local.toFixed(2)}</span>
      </div>
      <Slider value={[local]} min={0} max={1.5} step={0.05}
        onValueChange={([v]) => setLocal(v)} onValueCommit={([v]) => run(() => api.setConfig({ temp: v }))} />
    </div>
  );
}

function WhatIsThis({ onClose }) {
  return (
    <div className="hud-card relative rounded-md border border-amber/40 bg-amber/10 p-3 text-[12px] leading-relaxed text-amber-foreground">
      <button onClick={onClose} type="button" title="dismiss"
        className="absolute right-2 top-2 text-amber-foreground/70 hover:text-amber-foreground">
        <X className="size-4" />
      </button>
      <div className="mb-1 flex items-center gap-2 font-black uppercase text-amber">
        <HelpCircle className="size-4" /> What is this?
      </div>
      <p>
        This is a normal chat with a small AI (Gemma) running on your computer — but you can
        reach inside its head while it talks. Each <b>vibe</b> nudges the model's live wiring,
        like seasoning a dish, so it answers in a different mood. You can't break it. Hit
        <b> Plain Gemma</b> anytime to clear everything and go back to normal.
      </p>
    </div>
  );
}
