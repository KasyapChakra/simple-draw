// Scratchpad command API: turns a plain-English instruction into canvas operations using an Azure OpenAI deployment.
import http from "node:http";
import OpenAI from "openai";

// Azure OpenAI's OpenAI-compatible endpoint: base URL like https://<resource>.services.ai.azure.com/openai/v1, model = deployment name
const client = new OpenAI({ apiKey: process.env.AZURE_OPENAI_API_KEY, baseURL: process.env.AZURE_OPENAI_BASE_URL });
const MODEL = process.env.MODEL || "gpt-6-luna";
const PORT = Number(process.env.PORT || 8787);
const ALLOWED = (process.env.ALLOWED_ORIGINS || "https://scratchpad-pos0.onrender.com").split(",").map(s => s.trim()).filter(Boolean);
const PER_IP_PER_MIN = 30, GLOBAL_PER_HOUR = 600;

const SYSTEM = `You control a simple whiteboard / diagram canvas. The user types an instruction; you answer with ONE JSON object and nothing else:
{"reply": "<short confirmation or answer, under 15 words>", "ops": [ ...operations, in order... ]}

CANVAS: W x H pixels (given per request), origin top-left, x to the right, y down. Pen colors: black, red, blue, green, orange. Text is always black.
OBJECTS (every object has a numeric "id"):
- rect: {x,y,w,h,color}   (x,y = top-left corner)
- circle: an ellipse filling the box {x,y,w,h,color}
- triangle: apex at top-center, flat base at the bottom of the box {x,y,w,h,color}
- arrow: {x1,y1,x2,y2,color}, arrowhead at (x2,y2)
- text: {x,y,w,h,text} plus either plain:true (free text at normal 16px size; w,h are automatic) or a box whose font auto-fits w,h
- stroke: freehand line {color, points:[{x,y},...]} (in the state you only see bbox and the point count)

OPERATIONS:
- {"op":"add","object":{...}}   object without id. Shortcuts: an arrow may give "from":<id> and/or "to":<id> instead of coordinates and its ends snap to those shapes' edges; a text may give "inside":<id> to become a label fitted inside that shape (preferred way to label a shape); a text with no w/h becomes plain text.
- {"op":"update","id":N,"set":{...}}   change any fields: x,y,w,h, x1..y2, color, text, points, type (rect/circle/triangle only), plus helpers "dx","dy" (move by a delta), "from"/"to" (re-snap an arrow), "inside" (refit a label).
- {"op":"delete","ids":[...]}
- {"op":"clear"}   remove everything
- {"op":"undo"} / {"op":"redo"}   the user's undo history (one step each)
- {"op":"select","id":N}
- {"op":"pen","color":"red"}   change the current pen color
- {"op":"download"}   save the canvas as a PNG;  {"op":"copy"}   copy a text description to the clipboard

GUIDELINES: Just do it; do not ask questions unless the request is truly impossible to interpret. Pick sensible sizes (boxes ~160x90, circles ~100x100), ~40px gaps, keep everything inside the canvas, avoid overlapping existing objects unless asked. "here" / "there" means near the mouse position. A labelled shape = add the shape, then a text with "inside" set to its id (use "$1" style references? NO: ids of objects you add in this batch are assigned in order starting from the "next id" given in the request, so you can compute them). Build whole diagrams (flowcharts, trees, etc.) in one batch with arrows via from/to. If the instruction is a question about the canvas, answer in reply with an empty ops list. Output JSON only, no markdown fences.`;

const ipHits = new Map(); let globalHits = [];
function rateLimited(ip) {
  const now = Date.now();
  globalHits = globalHits.filter(t => now - t < 3600e3);
  if (globalHits.length >= GLOBAL_PER_HOUR) return true;
  const arr = (ipHits.get(ip) || []).filter(t => now - t < 60e3);
  if (arr.length >= PER_IP_PER_MIN) return true;
  arr.push(now); ipHits.set(ip, arr); globalHits.push(now);
  return false;
}
const originOk = o => !!o && (ALLOWED.includes(o) || /^https?:\/\/localhost(:\d+)?$/.test(o) || /^https?:\/\/127\.0\.0\.1(:\d+)?$/.test(o) || o === "null");

function send(res, status, body, origin) {
  const h = { "content-type": "application/json" };
  if (originOk(origin)) { h["access-control-allow-origin"] = origin; h["vary"] = "origin"; }
  res.writeHead(status, h); res.end(JSON.stringify(body));
}
function extractJSON(text) {
  const t = text.replace(/```(?:json)?/gi, "").trim();
  const a = t.indexOf("{"), b = t.lastIndexOf("}");
  if (a < 0 || b <= a) throw new Error("no JSON in model reply");
  return JSON.parse(t.slice(a, b + 1));
}

http.createServer(async (req, res) => {
  const origin = req.headers.origin;
  if (req.method === "OPTIONS") {
    const h = { "access-control-allow-methods": "POST, GET, OPTIONS", "access-control-allow-headers": "content-type", "access-control-max-age": "86400" };
    if (originOk(origin)) { h["access-control-allow-origin"] = origin; h["vary"] = "origin"; }
    res.writeHead(204, h); return res.end();
  }
  if (req.method === "GET" && (req.url === "/health" || req.url === "/")) return send(res, 200, { ok: true, model: MODEL }, origin);
  if (req.method !== "POST" || req.url !== "/api/command") return send(res, 404, { error: "not found" }, origin);
  if (!originOk(origin)) return send(res, 403, { error: "origin not allowed" }, origin);
  const ip = (req.headers["x-forwarded-for"] || req.socket.remoteAddress || "").toString().split(",")[0].trim();
  if (rateLimited(ip)) return send(res, 429, { error: "too many requests, try again in a minute" }, origin);

  let raw = ""; let tooBig = false;
  req.on("data", c => { raw += c; if (raw.length > 400_000) { tooBig = true; req.destroy(); } });
  req.on("end", async () => {
    if (tooBig) return;
    let body;
    try { body = JSON.parse(raw); } catch { return send(res, 400, { error: "bad JSON" }, origin); }
    const instruction = String(body.instruction || "").slice(0, 4000).trim();
    if (!instruction) return send(res, 400, { error: "empty instruction" }, origin);
    const canvas = body.canvas || {}, objects = Array.isArray(body.objects) ? body.objects : [];
    const maxId = objects.reduce((m, o) => Math.max(m, Number(o.id) || 0), 0);
    const nextId = Number(body.nextId) > maxId ? Number(body.nextId) : maxId + 1;
    const user = [
      `Canvas: ${canvas.w || "?"}x${canvas.h || "?"} px. Pen color: ${body.pen || "black"}. Selected object id: ${body.selected ?? "none"}. Mouse at: (${Math.round(body.mouse?.x ?? 0)},${Math.round(body.mouse?.y ?? 0)}). Highest existing id: ${maxId}. New objects you add get ids ${nextId}, ${nextId + 1}, ... in the order you add them.`,
      `Current objects (JSON): ${JSON.stringify(objects)}`,
      `Instruction: ${instruction}`,
    ].join("\n\n");
    const t0 = Date.now();
    try {
      const resp = await client.chat.completions.create({
        model: MODEL,
        messages: [{ role: "system", content: SYSTEM }, { role: "user", content: user }],
        response_format: { type: "json_object" },
      });
      const text = resp.choices?.[0]?.message?.content || "";
      let out;
      try { out = extractJSON(text); } catch { return send(res, 502, { error: "model reply was not valid JSON", raw: text.slice(0, 500) }, origin); }
      const ops = Array.isArray(out.ops) ? out.ops : [];
      console.log(JSON.stringify({ ms: Date.now() - t0, model: resp.model, in: resp.usage?.prompt_tokens, out: resp.usage?.completion_tokens, ops: ops.length, instruction: instruction.slice(0, 120) }));
      return send(res, 200, { reply: String(out.reply || "done"), ops }, origin);
    } catch (e) {
      const detail = e?.error?.message || e?.message || "model call failed";
      console.error("command failed", e?.status, detail);
      const friendly = e?.status === 401 || e?.status === 403 ? "the Azure OpenAI key was rejected" : e?.status === 429 ? "Azure OpenAI is rate-limiting; try again in a moment" : detail;
      return send(res, e?.status === 429 ? 429 : 502, { error: friendly }, origin);
    }
  });
}).listen(PORT, () => console.log(`scratchpad api on :${PORT} model=${MODEL} base=${process.env.AZURE_OPENAI_BASE_URL || "(unset)"} origins=${ALLOWED.join(",")}`));
