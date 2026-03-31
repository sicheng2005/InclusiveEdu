import os
import json
import uuid
import hashlib

from fastapi import (
    FastAPI, Request, Form, UploadFile, File, Query,
    WebSocket, WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from database import (
    init_db, create_user, get_user_by_username, get_user_by_id,
    save_courseware, get_courseware_by_id,
    get_courseware_by_classroom, get_courseware_orphan_by_teacher,
    create_classroom, get_classroom_by_code, get_active_classroom_by_teacher,
    get_classrooms_by_teacher, get_classroom_by_id_and_teacher, end_classroom_for_teacher,
    session_create, session_get_user_id, session_delete,
)

app = FastAPI(title="无障碍线上教学辅助平台")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "static", "css"), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "static", "js"), exist_ok=True)

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# ---------------------------------------------------------------------------
# Session & Auth helpers（会话持久化在 SQLite，避免进程重启后反复要求登录）
# ---------------------------------------------------------------------------
SESSION_COOKIE = "session_id"
SESSION_MAX_AGE = 60 * 60 * 24 * 14  # 14 天


def _hash_pw(password: str) -> str:
    return hashlib.sha256(f"edu_platform_{password}_salt".encode()).hexdigest()


def _verify_pw(password: str, hashed: str) -> bool:
    return _hash_pw(password) == hashed


def _attach_session_cookie(response, session_id: str):
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        max_age=SESSION_MAX_AGE,
        path="/",
        httponly=True,
        samesite="lax",
    )


def _user_id_from_session(session_id: str | None):
    if not session_id:
        return None
    return session_get_user_id(session_id)


def _current_user(request: Request):
    sid = request.cookies.get(SESSION_COOKIE)
    uid = _user_id_from_session(sid)
    if uid:
        return get_user_by_id(uid)
    return None


ROLE_LABELS = {
    "teacher": "老师",
    "student": "学生",
    # 兼容旧库中可能仍存在的取值（展示时统一为「学生」）
    "deaf_student": "学生",
    "blind_student": "学生",
}

# 凡可进入「学习端」的角色（含历史数据）
STUDENT_ROLES = frozenset({"student", "deaf_student", "blind_student"})


def _is_student(user) -> bool:
    return bool(user and user["role"] in STUDENT_ROLES)


def _normalize_room_code(room_code: str) -> str:
    """同一课堂内师生必须使用同一房间键；统一大写避免 WebSocket 分到不同房间。"""
    return (room_code or "").strip().upper()


# ---------------------------------------------------------------------------
# WebSocket 课堂连接管理
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self.rooms: dict[str, list[dict]] = {}

    async def connect(self, ws: WebSocket, room_code: str, user_info: dict):
        await ws.accept()
        self.rooms.setdefault(room_code, []).append({"ws": ws, "user": user_info})

    def disconnect(self, ws: WebSocket, room_code: str):
        if room_code in self.rooms:
            self.rooms[room_code] = [c for c in self.rooms[room_code] if c["ws"] is not ws]

    async def broadcast(self, room_code: str, message: dict, exclude_ws: WebSocket = None):
        for conn in self.rooms.get(room_code, []):
            if exclude_ws and conn["ws"] is exclude_ws:
                continue
            try:
                await conn["ws"].send_text(json.dumps(message, ensure_ascii=False))
            except Exception:
                pass


manager = ConnectionManager()

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    init_db()

# ---------------------------------------------------------------------------
# 页面路由
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "user": _current_user(request)})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
async def login(
    request: Request,
    username: str = Form(None),
    password: str = Form(None)
):
    if not username or not password:
        return templates.TemplateResponse("login.html", {"request": request, "error": "请填写用户名和密码"})
    user = get_user_by_username(username)
    if not user or not _verify_pw(password, user["password_hash"]):
        return templates.TemplateResponse("login.html", {"request": request, "error": "用户名或密码错误"})

    sid = str(uuid.uuid4())
    session_create(sid, user["id"])

    if user["role"] == "teacher":
        dest = "/teacher"
    elif _is_student(user):
        dest = "/student"
    else:
        dest = "/"
    resp = RedirectResponse(url=dest, status_code=303)
    _attach_session_cookie(resp, sid)
    return resp


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "error": None})


@app.post("/register")
async def register(
    request: Request,
    username: str = Form(None),
    password: str = Form(None),
    role: str = Form(None)
):
    if not username or not password or not role:
        return templates.TemplateResponse("register.html", {"request": request, "error": "请填写完整信息"})
    if len(username) < 2 or len(password) < 4:
        return templates.TemplateResponse("register.html", {"request": request, "error": "用户名至少 2 字符，密码至少 4 字符"})
    if role not in ("teacher", "student"):
        return templates.TemplateResponse("register.html", {"request": request, "error": "请选择「老师」或「学生」"})

    ok = create_user(username, _hash_pw(password), role)
    if not ok:
        return templates.TemplateResponse("register.html", {"request": request, "error": "用户名已存在"})
    return RedirectResponse(url="/login", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    sid = request.cookies.get(SESSION_COOKIE)
    if sid:
        session_delete(sid)
    resp = RedirectResponse(url="/")
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp

# ---------------------------------------------------------------------------
# 老师端
# ---------------------------------------------------------------------------

@app.get("/teacher", response_class=HTMLResponse)
async def teacher_page(request: Request):
    user = _current_user(request)
    if not user or user["role"] != "teacher":
        return RedirectResponse(url="/login")
    tid = user["id"]
    active = get_active_classroom_by_teacher(tid)
    cls_rows = get_classrooms_by_teacher(tid)
    classrooms_display = []
    for row in cls_rows:
        classrooms_display.append({
            "id": row["id"],
            "name": row["name"],
            "room_code": row["room_code"],
            "is_active": row["is_active"],
            "created_at": row["created_at"],
            "cw_count": len(get_courseware_by_classroom(row["id"])),
        })
    cw_active = get_courseware_by_classroom(active["id"]) if active else []
    return templates.TemplateResponse("teacher.html", {
        "request": request,
        "user": user,
        "courseware_list": cw_active,
        "courseware_orphan": get_courseware_orphan_by_teacher(tid),
        "classroom": active,
        "classrooms": classrooms_display,
    })


@app.get("/teacher/live", response_class=HTMLResponse)
async def teacher_live_page(request: Request):
    user = _current_user(request)
    if not user or user["role"] != "teacher":
        return RedirectResponse(url="/login")
    classroom = get_active_classroom_by_teacher(user["id"])
    if not classroom:
        return RedirectResponse(url="/teacher", status_code=303)
    tid = user["id"]
    return templates.TemplateResponse("teacher_live.html", {
        "request": request,
        "user": user,
        "classroom": classroom,
        "courseware_list": get_courseware_by_classroom(classroom["id"]),
    })


@app.post("/end-classroom")
async def end_classroom_route(
    request: Request,
    classroom_id: int = Form(None)
):
    user = _current_user(request)
    if not user or user["role"] != "teacher":
        return RedirectResponse(url="/login")
    if classroom_id is None:
        return RedirectResponse(url="/teacher", status_code=303)
    row = get_classroom_by_id_and_teacher(classroom_id, user["id"])
    if not row or not row["is_active"]:
        return RedirectResponse(url="/teacher", status_code=303)
    room_key = _normalize_room_code(row["room_code"])
    await manager.broadcast(room_key, {
        "type": "system",
        "content": "老师已结束本课堂，学生将无法再通过房间号加入。",
    })
    end_classroom_for_teacher(user["id"], classroom_id)
    return RedirectResponse(url="/teacher", status_code=303)


@app.post("/create-classroom")
async def create_classroom_route(
    request: Request,
    room_name: str = Form(None)
):
    user = _current_user(request)
    if not user or user["role"] != "teacher":
        return RedirectResponse(url="/login")
    if not room_name:
        return RedirectResponse(url="/teacher", status_code=303)
    room_code = uuid.uuid4().hex[:6].upper()
    create_classroom(user["id"], room_code, room_name)
    return RedirectResponse(url="/teacher", status_code=303)


def _safe_redirect_path(redirect_to: str, default: str = "/teacher") -> str:
    if not redirect_to or not isinstance(redirect_to, str):
        return default
    redirect_to = redirect_to.strip()
    if not redirect_to.startswith("/") or redirect_to.startswith("//"):
        return default
    return redirect_to


@app.post("/upload-courseware")
async def upload_courseware(
    request: Request,
    file: UploadFile = File(...),
    classroom_id: int = Form(...),
    redirect_to: str = Form("/teacher"),
):
    user = _current_user(request)
    if not user or user["role"] != "teacher":
        return RedirectResponse(url="/login")

    c = get_classroom_by_id_and_teacher(classroom_id, user["id"])
    if not c:
        return RedirectResponse(url="/teacher", status_code=303)

    ext = os.path.splitext(file.filename)[1]
    saved_name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, saved_name)
    with open(path, "wb") as f:
        f.write(await file.read())

    text_content = _extract_text(path, ext)
    save_courseware(user["id"], saved_name, file.filename, text_content, classroom_id, file_ext=ext)
    dest = _safe_redirect_path(redirect_to, "/teacher")
    return RedirectResponse(url=dest, status_code=303)

# ---------------------------------------------------------------------------
# 学生端
# ---------------------------------------------------------------------------

@app.get("/student", response_class=HTMLResponse)
async def student_page(request: Request):
    user = _current_user(request)
    if not _is_student(user):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("student.html", {"request": request, "user": user})


@app.get("/deaf-student")
async def legacy_deaf_student():
    return RedirectResponse(url="/student", status_code=307)


@app.get("/blind-student")
async def legacy_blind_student():
    return RedirectResponse(url="/student", status_code=307)

# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, description="用户名")
    password: str = Field(..., min_length=4, description="密码")
    role: str = Field(..., description="角色: teacher 或 student")

class LoginRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")

@app.post("/api/register", summary="用户注册接口")
async def api_register(req: RegisterRequest):
    if req.role not in ("teacher", "student"):
        return JSONResponse({"success": False, "error": "角色只能是 teacher 或 student"}, status_code=400)
    
    hashed_pw = _hash_pw(req.password)
    ok = create_user(req.username, hashed_pw, req.role)
    if not ok:
        return JSONResponse({"success": False, "error": "用户名已存在"}, status_code=409)
        
    return {"success": True, "message": "注册成功"}

@app.post("/api/login", summary="用户登录接口")
async def api_login(req: LoginRequest):
    user = get_user_by_username(req.username)
    if not user or not _verify_pw(req.password, user["password_hash"]):
        return JSONResponse({"success": False, "error": "用户名或密码错误"}, status_code=401)
    
    sid = str(uuid.uuid4())
    session_create(sid, user["id"])

    response = JSONResponse({
        "success": True, 
        "message": "登录成功",
        "data": {
            "token": sid,
            "user_id": user["id"],
            "username": user["username"],
            "role": user["role"]
        }
    })
    _attach_session_cookie(response, sid)
    return response

@app.get("/api/user/me", summary="获取当前用户信息")
async def api_get_current_user(request: Request):
    user = _current_user(request)
    if not user:
        return JSONResponse({"success": False, "error": "未登录或登录已过期"}, status_code=401)
        
    return {
        "success": True,
        "data": {
            "user_id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "display_name": user["display_name"]
        }
    }

@app.post("/api/join-classroom")
async def api_join_classroom(
    request: Request,
    room_code: str = Form(None)
):
    user = _current_user(request)
    if not user:
        return JSONResponse({"error": "未登录"}, status_code=401)
    if not room_code:
        return JSONResponse({"error": "请输入房间号"}, status_code=400)
    classroom = get_classroom_by_code(room_code.strip().upper())
    if not classroom:
        return JSONResponse({"error": "课堂不存在或已结束"}, status_code=404)
    cw_list = get_courseware_by_classroom(classroom["id"])
    return JSONResponse({
        "success": True,
        "classroom_id": classroom["id"],
        "room_code": _normalize_room_code(classroom["room_code"]),
        "room_name": classroom["name"],
        "courseware": [{"id": c["id"], "name": c["original_name"]} for c in cw_list],
    })


@app.get("/api/courseware/{cw_id}/text")
async def api_courseware_text(
    cw_id: int,
    request: Request,
    classroom_id: int = Query(..., description="加入课堂时返回的 classroom_id，用于校验课件归属"),
):
    user = _current_user(request)
    if not user:
        return JSONResponse({"error": "未登录"}, status_code=401)
    cw = get_courseware_by_id(cw_id)
    if not cw or cw["classroom_id"] is None or int(cw["classroom_id"]) != int(classroom_id):
        return JSONResponse({"error": "课件不存在或无权访问"}, status_code=404)
    return JSONResponse({"text": cw["text_content"], "name": cw["original_name"]})

# ---------------------------------------------------------------------------
# WebSocket 实时课堂
# ---------------------------------------------------------------------------

@app.websocket("/ws/classroom/{room_code}")
async def ws_classroom(websocket: WebSocket, room_code: str):
    room_key = _normalize_room_code(room_code)
    if not room_key:
        await websocket.close(code=4000)
        return

    sid = websocket.cookies.get(SESSION_COOKIE)
    uid = _user_id_from_session(sid)
    user = get_user_by_id(uid) if uid else None
    if not user:
        await websocket.close(code=4001)
        return

    user_info = {"id": user["id"], "username": user["username"], "role": user["role"]}
    await manager.connect(websocket, room_key, user_info)
    await manager.broadcast(room_key, {
        "type": "system",
        "content": f"{user['username']}（{ROLE_LABELS.get(user['role'], '')}）加入了课堂",
        "username": user["username"],
        "role": user["role"],
    })

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg["username"] = user["username"]
            msg["role"] = user["role"]
            
            # WebRTC 信令消息，不广播给自己
            if msg.get("type") in ("webrtc_offer", "webrtc_answer", "webrtc_ice", "webrtc_ready"):
                await manager.broadcast(room_key, msg, exclude_ws=websocket)
            else:
                await manager.broadcast(room_key, msg)
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_key)
        await manager.broadcast(room_key, {
            "type": "system",
            "content": f"{user['username']} 离开了课堂",
            "username": user["username"],
            "role": user["role"],
        })

# ---------------------------------------------------------------------------
# 课件文本提取
# ---------------------------------------------------------------------------

def _extract_text(file_path: str, ext: str) -> str:
    ext = ext.lower()
    try:
        if ext == ".txt":
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        if ext == ".pptx":
            from pptx import Presentation
            prs = Presentation(file_path)
            parts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        parts.append(shape.text.strip())
            return "\n".join(parts)
        if ext == ".pdf":
            import fitz
            doc = fitz.open(file_path)
            parts = [page.get_text() for page in doc]
            doc.close()
            return "\n".join(parts)
        if ext in (".doc", ".docx"):
            from docx import Document
            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return f"[暂不支持 {ext} 格式的文本提取]"
    except ImportError as e:
        return f"[缺少依赖库: {e}，请 pip install 对应包]"
    except Exception as e:
        return f"[提取失败: {e}]"

# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("正在启动无障碍线上教学辅助平台...")
    print("请在浏览器中打开: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
