#!/usr/bin/env bash
# scripts/scaffolds/scaffold_035.sh
# Enhance Discord bot with control commands
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"; bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

# requirements already set by previous bot scaffold; ensure httpx present
grep -q "^httpx" requirements.txt || echo "httpx>=0.27.2" >> requirements.txt

# upgrade bot script if exists (backup then overwrite)
if [[ -f scripts/discord_bot.py ]]; then bk scripts/discord_bot.py; fi
cat > scripts/discord_bot.py <<'PY'
import os, asyncio, httpx, discord, re, textwrap
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
BASE  = os.getenv("DISCORD_FORWARD_URL", "http://localhost:8084")
CAPTURE_URL = BASE.rstrip("/") + "/capture"
API_STATS   = BASE.rstrip("/") + "/api/stats"
API_RECENT  = BASE.rstrip("/") + "/api/recent"
API_SEARCH  = BASE.rstrip("/") + "/api/search"
TAG_RENAME  = BASE.rstrip("/") + "/tags/rename"

BEARER = os.getenv("DISCORD_FORWARD_BEARER")
PREFIX = os.getenv("DISCORD_BOT_PREFIX", "!")
ALLOWED = set(filter(None, (os.getenv("DISCORD_ALLOWED_GUILDS","").split(","))))

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

def auth_headers():
    return {"Authorization": f"Bearer {BEARER}"} if BEARER else {}

async def http_json(url, method="GET", **kw):
    kw.setdefault("timeout", 30)
    headers = kw.pop("headers", {})
    headers.update(auth_headers())
    async with httpx.AsyncClient(timeout=kw.pop("timeout")) as cli:
        r = await cli.request(method, url, headers=headers, **kw)
        r.raise_for_status()
        if "application/json" in r.headers.get("content-type",""):
            return r.json()
        return r.text

async def forward_text(note:str, tags:str="discord"):
    data = {"note": note, "tags": tags}
    async with httpx.AsyncClient(timeout=30) as cli:
        r = await cli.post(CAPTURE_URL, data=data, headers=auth_headers())
        r.raise_for_status()

async def handle_command(msg: discord.Message):
    if msg.author.bot: return
    if ALLOWED and str(msg.guild.id) not in ALLOWED: return
    content = msg.content.strip()

    # !ping
    if content.startswith(PREFIX+"ping"):
        await msg.reply("pong"); return

    # !stats
    if content.startswith(PREFIX+"stats"):
        try:
            d = await http_json(API_STATS)
            total = d.get("total", 0); by = d.get("by_type", {})
            top = ", ".join([f"#{t['tag']}({t['count']})" for t in d.get("top_tags", [])[:5]]) or "—"
            out = f"**Stats** — total: {total}, notes: {by.get('note',0)}, audio: {by.get('audio',0)}\nTop tags: {top}"
            await msg.reply(out)
        except Exception as e:
            await msg.reply(f"stats error: {e}")
        return

    # !recent [n]
    if content.startswith(PREFIX+"recent"):
        m = re.match(rf"^{re.escape(PREFIX)}recent\s+(\d+)$", content)
        n = int(m.group(1)) if m else 5
        try:
            d = await http_json(API_RECENT+f"?limit={max(1,min(20,n))}")
            lines=[]
            for it in d.get("items", []):
                lines.append(f"- #{it['id']} {it['title']} · {it.get('ts','')}")
            await msg.reply("**Recent**\n" + "\n".join(lines) if lines else "No recent items.")
        except Exception as e:
            await msg.reply(f"recent error: {e}")
        return

    # !search <q...>
    if content.startswith(PREFIX+"search"):
        q = content[len(PREFIX)+len("search"):].strip()
        if not q: await msg.reply("Usage: !search <query>"); return
        try:
            d = await http_json(API_SEARCH+f"?q={httpx.QueryParams({'q':q})['q']}")
            items = d.get("items", [])[:10]
            if not items: await msg.reply("No matches."); return
            lines = [f"- #{it['id']} {it['title']} · {it.get('ts','')}" for it in items]
            await msg.reply("**Search**\n" + "\n".join(lines))
        except Exception as e:
            await msg.reply(f"search error: {e}")
        return

    # !renameTag <old> <new>
    if content.startswith(PREFIX+"renameTag"):
        parts = content.split()
        if len(parts) != 3:
            await msg.reply("Usage: !renameTag <old> <new>")
            return
        old, new = parts[1].lower(), parts[2].lower()
        try:
            async with httpx.AsyncClient(timeout=60) as cli:
                r = await cli.post(TAG_RENAME, data={"old": old, "new": new}, headers=auth_headers())
                if r.status_code == 200 or r.status_code == 302:
                    await msg.reply(f"Renamed #{old} → #{new}")
                else:
                    await msg.reply(f"rename failed: {r.status_code} {r.text}")
        except Exception as e:
            await msg.reply(f"rename error: {e}")
        return

    # !capture <text...>
    if content.startswith(PREFIX+"capture"):
        note = content[len(PREFIX)+len("capture"):].strip()
        if not note:
            await msg.reply("Usage: !capture <text>")
            return
        try:
            await forward_text(f"[{msg.author.display_name}] {note}", tags="discord")
            await msg.add_reaction("🧠")
        except Exception as e:
            await msg.reply(f"capture error: {e}")
        return

    # If no command: forward plain text as capture
    if content:
        try:
            await forward_text(f"[{msg.author.display_name}] {content}", tags="discord")
            await msg.add_reaction("🧠")
        except Exception:
            pass

@bot.event
async def on_ready():
    print(f"Bot ready: {bot.user} (latency {bot.latency*1000:.0f}ms)")

@bot.event
async def on_message(message: discord.Message):
    await handle_command(message)

if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("Set DISCORD_BOT_TOKEN in .env")
    bot.run(TOKEN)
PY

# 2) Makefile helper (if not already)
if [[ -f Makefile ]] && ! grep -q "^discord:" Makefile; then
  cat >> Makefile <<'MK'

# === Discord ===
discord:
	@. .venv/bin/activate && python scripts/discord_bot.py
MK
fi

echo "Done 035: Discord bot now supports: !stats, !recent [n], !search <q>, !renameTag <old> <new>, !capture <text>."
