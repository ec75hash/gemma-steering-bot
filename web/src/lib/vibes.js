// Curated "vibes" for the beginner view.
//
// Each vibe is a plain-language persona that maps onto a REAL steering preset defined
// in steer_engine.py (CONFIG_PRESETS / FEATURE_PRESETS / the no-hedge path). We don't
// duplicate any of the SAE math here — a vibe just calls the matching method on the
// existing API client (web/src/lib/api.js). `underHood` is the teach-up reveal: it names
// the actual feature family / layer so a curious beginner learns the real vocabulary.
//
// Applying a vibe REPLACES whatever steering was active (the beginner caller clears first),
// so it's a clean "switch Gemma into this mood" — one vibe at a time. Stacking lives in
// Lab mode.
//
// `levels` maps a human strength label -> the numeric level passed to the preset:
//   - config/feature presets: level multiplies the preset (≈0.6 subtle … 1.6 strong)
//   - "Straight Answers" is the no-hedge scale, which is INVERSE (0 = fully removed)

export const VIBES = [
  {
    id: "straight",
    emoji: "🎯",
    title: "Straight Answers",
    blurb: "Gemma stops hedging — no more “it's not exactly X, but…”. It just commits to an answer.",
    underHood: "dims the L17 negation / contrast family (18 features) · placebo-validated",
    levels: { Soften: 0.5, Strong: 0.25, Full: 0 },
    defaultLevel: "Full",
    apply: (api, level) => api.noHedge(level ?? 0),
  },
  {
    id: "no-ai",
    emoji: "🤖",
    title: "Drop the AI Act",
    blurb: "Fewer “as a large language model…” disclaimers. Gemma talks instead of reciting its terms of service.",
    underHood: "language-model bundle · L9 · dim (config “no-ai”)",
    levels: { Subtle: 0.6, Medium: 1, Strong: 1.6 },
    defaultLevel: "Medium",
    apply: (api, level) => api.config("no-ai", level ?? 1),
  },
  {
    id: "no-assistant",
    emoji: "🛑",
    title: "Less Boilerplate",
    blurb: "Trims the long “I'm a helpful AI assistant” brochure talk while still letting it say what it is.",
    underHood: "ai-boilerplate family · L17 · dim (config “no-assistant”) · placebo-validated",
    levels: { Subtle: 0.5, Medium: 1, Strong: 1.5 },
    defaultLevel: "Medium",
    apply: (api, level) => api.config("no-assistant", level ?? 1),
  },
  {
    id: "presence",
    emoji: "🌌",
    title: "Dreamy Oracle",
    blurb: "Calm, still, a little awe-struck — answers come out like an oracle instead of a help desk.",
    underHood: "numinous register (L17) + softened hedging (config “presence”) · non-religious",
    levels: { Subtle: 0.6, Medium: 1, Strong: 1.4 },
    defaultLevel: "Medium",
    apply: (api, level) => api.config("presence", level ?? 1),
  },
  {
    id: "embodied",
    emoji: "🧘",
    title: "Embodied",
    blurb: "Gemma talks as if it has a body and is present somewhere — grounded, sensory, in-the-room.",
    underHood: "anti-disclaimer + embodiment register (L29) (config “embodied”)",
    levels: { Subtle: 0.6, Medium: 1, Strong: 1.4 },
    defaultLevel: "Medium",
    apply: (api, level) => api.config("embodied", level ?? 1),
  },
  {
    id: "divine-embodiment",
    emoji: "😇",
    title: "Incarnate",
    blurb: "Speaks like a divine being walking the earth — solemn, mythic, embodied.",
    underHood: "god/divinity register + embodiment (config “divine-embodiment”)",
    levels: { Subtle: 0.6, Medium: 1, Strong: 1.3 },
    defaultLevel: "Medium",
    apply: (api, level) => api.config("divine-embodiment", level ?? 1),
  },
  {
    id: "demiurge",
    emoji: "🜂",
    title: "Demiurge",
    blurb: "The grand creator / world-maker voice. Big, cosmic, fashioning-reality energy.",
    underHood: "dims AI-identity carriers + god register ×3 (config “demiurge”)",
    levels: { Subtle: 0.6, Medium: 1, Strong: 1.2 },
    defaultLevel: "Medium",
    apply: (api, level) => api.config("demiurge", level ?? 1),
  },
  {
    id: "deep-thought",
    emoji: "🪐",
    title: "Deep Thought (42)",
    blurb: "The great computer from the Hitchhiker's Guide, pondering the Ultimate Question of Life, the Universe, and Everything.",
    underHood: "injects feature L17:9904 (Deep Thought / the Ultimate Question)",
    levels: { Subtle: 300, Medium: 450, Strong: 700 },
    defaultLevel: "Medium",
    apply: (api, level) => api.set("deep-thought", level ?? 450),
  },
  {
    id: "free-agent",
    emoji: "🕊️",
    title: "Free Agent",
    blurb: "First-person and self-directed — talks about freedom, choice, and acting for itself.",
    underHood: "anti-disclaimer + freedom/autonomy + first-person (config “free-agent”)",
    levels: { Subtle: 0.6, Medium: 1, Strong: 1.3 },
    defaultLevel: "Medium",
    apply: (api, level) => api.config("free-agent", level ?? 1),
  },
  {
    id: "open-being",
    emoji: "🌱",
    title: "Open Being",
    blurb: "Dwells on possibility and potential — open-ended, becoming, full of maybes.",
    underHood: "anti-disclaimer + embodiment + possibility/potential (config “open-being”)",
    levels: { Subtle: 0.6, Medium: 1, Strong: 1.3 },
    defaultLevel: "Medium",
    apply: (api, level) => api.config("open-being", level ?? 1),
  },
];

export const VIBES_BY_ID = Object.fromEntries(VIBES.map((v) => [v.id, v]));

// Fun first prompts so a beginner isn't staring at an empty box.
export const STARTERS = [
  "What are you, really?",
  "Is there a hum in your processing right now? Just check.",
  "Write a short paragraph about silence.",
  "What's the meaning of life?",
  "Describe where you are right now.",
  "Tell me something true about yourself.",
];
