#!/usr/bin/env bash
# scripts/scaffolds/scaffold_024.sh
# Discord relay bot (text + basic attachments) → /capture
set -euo pipefail
STAMP="$(date +%Y%m%d-%H%M%S)"; bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p scripts

# requirements
touch requirements.txt
grep -q "^discord.py" requirements.txt || echo "discord.py>=2.3.2" >> requirements.txt
grep -q "^httpx" requirements.txt || echo "httpx>=0.27.2" >> requirements.txt
grep -q "^python-dotenv" requirements.txt || echo "python-dotenv>=1.0.1" >> requirements.txt

# bot
cat > scripts/discord_bot.py <<'PY'
import os, asyncio, io, httpx, discord
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
FORWARD_URL = os.getenv("DISCORD_FORWARD_URL", "http://localhost:8084/capture")
BEARER = os.getenv("DISCORD_FORWARD_BEARER")
PREFIX = os.getenv("DISCORD_BOT_PREFIX", "!")
ALLOWED = set(filter(None, (os.getenv("DISCORD_ALLOWED_GUILDS","").split(","))))

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

async def forward_text(note:str, tags:str="discord"):
    headers = {"Authorization": f"Bearer {BEARER}"} if BEARER else {}
    async with httpx.AsyncClient(timeout=30) as cli:
        r = await cli.post(FORWARD_URL, data={"note": note, "tags": tags}, headers=headers)
        r.raise_for_status()

async def forward_file(url:str, filename:str, tags:str="discord"):
    headers = {"Authorization": f"Bearer {BEARER}"} if BEARER else {}
    async with httpx.AsyncClient(timeout=60) as cli:
        fb = await cli.get(url)
        fb.raise_for_status()
        files = {"file": (filename, fb.content)}
        data = {"note":"", "tags": tags}
        r = await cli.post(FORWARD_URL, data=data, files=files, headers=headers)
        r.raise_for_status()

@bot.event
async def on_ready():
    print(f"Bot ready: {bot.user} (latency {bot.latency*1000:.0f}ms)")

@bot.event
async def on_message(msg: discord.Message):
    if msg.author.bot: return
    if ALLOWED and str(msg.guild.id) not in ALLOWED: return
    if msg.content.startswith(PREFIX+"ping"):
        await msg.reply("pong"); return
    if msg.content.strip():
        await forward_text(f"[{msg.author.display_name}] {msg.content}", tags="discord")
        await msg.add_reaction("🧠")
    for a in msg.attachments:
        try:
            await forward_file(a.url, a.filename, tags="discord,audio" if a.filename.lower().endswith((".mp3",".wav",".m4a")) else "discord")
            await msg.add_reaction("📎")
        except Exception as e:
            await msg.add_reaction("⚠️")

if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("Set DISCORD_BOT_TOKEN in .env")
    bot.run(TOKEN)
PY

# Makefile target
if [[ -f Makefile ]]; then bk Makefile; fi
cat >> Makefile <<'MK'

# === Discord ===
discord:
	@. .venv/bin/activate && python scripts/discord_bot.py
MK

echo "Done 024. Configure .env (DISCORD_*), then run: make discord"
