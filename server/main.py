import os
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

# Optional OpenAI client (works if OPENAI_API_KEY is set)
try:
    from openai import OpenAI  # openai>=1.0.0
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI()


app = FastAPI(title="Game UI AI Backend")

# CORS: allow local file or localhost usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files under /static to avoid shadowing /api
STATIC_DIR = "/workspace"
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/prototype-3col.html")


@app.get("/prototype-3col.html")
async def serve_prototype():
    path = os.path.join(STATIC_DIR, "prototype-3col.html")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="prototype not found")
    return FileResponse(path)


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant" | "npc" | "player"
    content: str

class SuggestionRequest(BaseModel):
    chat: List[ChatMessage] = []
    world_events: List[Dict[str, Any]] = []
    n: int = 4

class SuggestionResponse(BaseModel):
    suggestions: List[str]

class ChatRequest(BaseModel):
    chat: List[ChatMessage] = []
    world_events: List[Dict[str, Any]] = []
    player_message: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str

class VoiceRequest(BaseModel):
    transcript: str
    chat: List[ChatMessage] = []
    world_events: List[Dict[str, Any]] = []

class VoiceResponse(BaseModel):
    world_event: Dict[str, Any]


MODEL_DEFAULT = os.getenv("MODEL", "gpt-4o-mini")


def _fallback_suggestions(n: int) -> List[str]:
    base = [
        "等等，先别说话，听——你听见了吗？",
        "向东转移队伍，躲开风暴眼。",
        "把消息压下去，免得引起恐慌。",
        "也许我们低估了天空的影子…",
        "别抬头，风从北边变冷了。",
        "嘘，嗡鸣在墙后回响。",
    ]
    return base[: max(1, min(n, len(base)))]


def _build_world_context(world_events: List[Dict[str, Any]]) -> str:
    if not world_events:
        return "(无世界事件)"
    lines = [f"- {ev.get('t','?')} · {ev.get('title','')} · {ev.get('desc','')}" for ev in world_events[:8]]
    return "\n".join(lines)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/suggestions", response_model=SuggestionResponse)
async def api_suggestions(req: SuggestionRequest):
    client = get_openai_client()
    n = max(1, min(8, req.n))

    if client is None:
        return SuggestionResponse(suggestions=_fallback_suggestions(n))

    world_ctx = _build_world_context(req.world_events)
    chat_tail = "\n".join([f"{m.role}: {m.content}" for m in req.chat[-8:]])

    system = (
        "你是插话建议生成器。根据世界事件与最近对话，给出短小、有暗示性、可点击的台词建议。"
        "要求：每条不超过16字，语义多样，有行动/情绪/信息差。只返回 JSON 对象，键为 suggestions。"
    )
    user = (
        f"世界事件:\n{world_ctx}\n\n最近对话:\n{chat_tail}\n\n"
        f"请生成 {n} 条中文插话建议。"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_DEFAULT,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=200,
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        suggestions = data.get("suggestions")
        if not isinstance(suggestions, list):
            raise ValueError("bad suggestions")
        suggestions = [str(s)[:30] for s in suggestions][:n]
        return SuggestionResponse(suggestions=suggestions)
    except Exception as e:
        # fallback on any error
        return SuggestionResponse(suggestions=_fallback_suggestions(n))


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    client = get_openai_client()

    player_msg = req.player_message or ""
    if client is None:
        # naive fallback: reflect player intent or continue generic line
        reply = "我听见了，但风暴不会等人。"
        if player_msg:
            reply = f"明白：{player_msg}。但命运仍在推进。"
        return ChatResponse(reply=reply)

    world_ctx = _build_world_context(req.world_events)
    chat_tail = "\n".join([f"{m.role}: {m.content}" for m in req.chat[-10:]])

    system = (
        "你是群聊中的 NPC 回复生成器。要求：\n"
        "- 语气自然简短（<= 30 字），不剧透，不强设定。\n"
        "- 优先参考左栏“世界事件/命运主线”，保持一致性。\n"
        "- 若玩家插话偏离命运，也要温和接住并暗示纠偏。"
    )

    user = (
        f"世界事件（命运主线摘要）：\n{world_ctx}\n\n"
        f"最近对话（含玩家）：\n{chat_tail}\n\n"
        f"玩家最新发言：{player_msg}\n"
        "请给出 NPC 的下一句简短中文回复。"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_DEFAULT,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.6,
            max_tokens=80,
        )
        reply = resp.choices[0].message.content.strip()
        return ChatResponse(reply=reply)
    except Exception:
        return ChatResponse(reply="先稳住阵脚，按既定路线走。")


@app.post("/api/voice", response_model=VoiceResponse)
async def api_voice(req: VoiceRequest):
    client = get_openai_client()

    if client is None:
        return VoiceResponse(world_event={
            "t": "此刻",
            "title": "低语如潮",
            "desc": "短暂的嗡鸣压过了争执，众人沉默片刻。",
        })

    world_ctx = _build_world_context(req.world_events)

    system = (
        "你是事件微扰生成器。输入为玩家的语音转写（文本），输出为一个微小世界事件。"
        "要求输出 JSON：{t,title,desc}。事件需细微但可感（如人群情绪、环境声响、视线变化）。"
    )
    user = (
        f"已知世界事件：\n{world_ctx}\n\n"
        f"玩家语音转写：{req.transcript}\n"
        "请生成一个新的细微世界事件。"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_DEFAULT,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=120,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        t = str(data.get("t") or "此刻")
        title = str(data.get("title") or "微弱回声")
        desc = str(data.get("desc") or "空气里浮起一阵细响，又迅速归于平静。")
        return VoiceResponse(world_event={"t": t, "title": title, "desc": desc})
    except Exception:
        return VoiceResponse(world_event={
            "t": "此刻",
            "title": "风停一瞬",
            "desc": "像是被谁按下了暂停键，所有人都稍稍屏住了气。",
        })