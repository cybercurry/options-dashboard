// Probe: confirms Cloudflare Pages Functions deploy from this directory and that /api/* routes.
export async function onRequestGet() {
  return new Response(JSON.stringify({ ok: true, pong: Date.now(), from: "functions/api/ping.js" }),
    { headers: { "content-type": "application/json", "access-control-allow-origin": "*" } });
}
