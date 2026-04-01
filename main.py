import os
import asyncio
import json
import uuid
import hashlib
import shutil
import subprocess

from fastapi import (
    FastAPI, Request, Form, UploadFile, File, Query,
    WebSocket, WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from database import (
    init_db, create_user, get_user_by_username, get_user_by_id,
    save_courseware, get_courseware_by_id, delete_courseware,
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


def _preview_pdf_path(file_path: str) -> str:
    root, _ = os.path.splitext(file_path)
    return f"{root}.preview.pdf"


def _convert_office_to_pdf(file_path: str) -> tuple[bool, str]:
    """
    将 doc/docx/pptx 转为 pdf 预览文件。
    需要系统可用的 LibreOffice/soffice。
    """
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        # Windows 兜底：尝试使用本机 Office COM 导出 PDF
        return _convert_office_to_pdf_windows(file_path)

    in_dir = os.path.dirname(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]
    generated_pdf = os.path.join(in_dir, f"{base}.pdf")
    preview_pdf = _preview_pdf_path(file_path)

    try:
        if os.path.exists(generated_pdf):
            os.remove(generated_pdf)
        cmd = [soffice, "--headless", "--convert-to", "pdf", "--outdir", in_dir, file_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
        if proc.returncode != 0 or not os.path.exists(generated_pdf):
            err = (proc.stderr or proc.stdout or "").strip() or "未知错误"
            return False, err
        if os.path.abspath(generated_pdf) != os.path.abspath(preview_pdf):
            if os.path.exists(preview_pdf):
                os.remove(preview_pdf)
            os.replace(generated_pdf, preview_pdf)
        return True, preview_pdf
    except Exception as e:
        return False, str(e)


def _convert_office_to_pdf_windows(file_path: str) -> tuple[bool, str]:
    """Windows 下使用本机 Office（Word/PPT）导出 PDF。"""
    if os.name != "nt":
        return False, "未检测到 LibreOffice/soffice，且当前系统不支持 Office COM 转换"

    ext = os.path.splitext(file_path)[1].lower()
    preview_pdf = _preview_pdf_path(file_path)
    abs_src = os.path.abspath(file_path)
    abs_pdf = os.path.abspath(preview_pdf)
    try:
        import win32com.client  # type: ignore
    except Exception:
        return False, "未安装 pywin32，且未检测到 LibreOffice/soffice，无法进行版式预览转换"

    try:
        if os.path.exists(abs_pdf):
            os.remove(abs_pdf)

        if ext in (".doc", ".docx"):
            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            doc = None
            try:
                doc = word.Documents.Open(abs_src, ReadOnly=True)
                # 17 = wdExportFormatPDF
                doc.ExportAsFixedFormat(abs_pdf, 17)
            finally:
                if doc is not None:
                    doc.Close(False)
                word.Quit()
        elif ext == ".pptx":
            ppt = win32com.client.DispatchEx("PowerPoint.Application")
            # 1 = msoTrue
            ppt.Visible = 1
            pres = None
            try:
                # WithWindow=False
                pres = ppt.Presentations.Open(abs_src, WithWindow=False)
                # 32 = ppSaveAsPDF
                pres.SaveAs(abs_pdf, 32)
            finally:
                if pres is not None:
                    pres.Close()
                ppt.Quit()
        else:
            return False, "仅支持 doc/docx/pptx 的 Office COM 转换"

        if not os.path.exists(abs_pdf):
            return False, "Office 已执行转换但未生成 PDF 文件"
        return True, abs_pdf
    except Exception as e:
        return False, f"Office COM 转换失败: {e}"


def _courseware_ext(cw) -> str:
    ext = (cw["file_ext"] or "").strip().lower()
    if ext:
        return ext
    return os.path.splitext(cw["filename"] or "")[1].lower()


def _courseware_file_path(cw) -> str:
    return os.path.join(UPLOAD_DIR, cw["filename"])


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

    # 文本提取放到线程并设置超时，避免个别文件解析卡住导致上传一直不返回
    try:
        text_content = await asyncio.wait_for(
            asyncio.to_thread(_extract_text, path, ext),
            timeout=12.0,
        )
    except asyncio.TimeoutError:
        text_content = "[课件文本提取超时，请稍后重试或更换文件]"

    # 对可转换格式预生成 PDF 预览（分页 + 尽量保留原布局）
    ext_lower = (ext or "").lower()
    if ext_lower in (".pptx", ".doc", ".docx"):
        try:
            await asyncio.wait_for(asyncio.to_thread(_convert_office_to_pdf, path), timeout=40.0)
        except Exception:
            # 预览转换失败时不影响上传主流程，前端会回退到文本展示
            pass

    save_courseware(user["id"], saved_name, file.filename, text_content, classroom_id, file_ext=ext)
    dest = _safe_redirect_path(redirect_to, "/teacher")
    return RedirectResponse(url=dest, status_code=303)


@app.post("/delete-courseware")
async def delete_courseware_route(
    request: Request,
    cw_id: int = Form(...),
    redirect_to: str = Form("/teacher"),
):
    user = _current_user(request)
    if not user or user["role"] != "teacher":
        return RedirectResponse(url="/login")

    cw = get_courseware_by_id(cw_id)
    if not cw or cw["teacher_id"] != user["id"]:
        return RedirectResponse(url="/teacher", status_code=303)

    delete_courseware(cw_id)
    
    # 可选：删除本地文件
    try:
        path = _courseware_file_path(cw)
        if os.path.exists(path):
            os.remove(path)
        preview_pdf = _preview_pdf_path(path)
        if os.path.exists(preview_pdf):
            os.remove(preview_pdf)
    except Exception:
        pass

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


@app.get("/api/courseware/{cw_id}/display")
async def api_courseware_display(
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

    ext = _courseware_ext(cw)
    file_path = _courseware_file_path(cw)
    preview_pdf = _preview_pdf_path(file_path)
    file_url_base = f"/api/courseware/{cw_id}/file?classroom_id={classroom_id}"

    # PDF：浏览器内直接分页展示，版式最接近原文件
    if ext == ".pdf":
        return JSONResponse({
            "mode": "pdf",
            "name": cw["original_name"],
            "file_url": f"{file_url_base}&variant=original",
        })

    # PPT/Word：优先使用预生成（或即时生成）的 PDF 预览
    if ext in (".pptx", ".doc", ".docx"):
        if not os.path.exists(preview_pdf):
            ok, _ = await asyncio.to_thread(_convert_office_to_pdf, file_path)
            if not ok:
                # 转换失败就回退文本展示，不阻塞课堂
                return JSONResponse({
                    "mode": "text",
                    "name": cw["original_name"],
                    "text": cw["text_content"] or "（暂无可展示文本）",
                })
        return JSONResponse({
            "mode": "pdf",
            "name": cw["original_name"],
            "file_url": f"{file_url_base}&variant=preview",
        })

    # 其他格式维持文本展示
    return JSONResponse({
        "mode": "text",
        "name": cw["original_name"],
        "text": cw["text_content"] or "（暂无可展示文本）",
    })


@app.get("/api/courseware/{cw_id}/file")
async def api_courseware_file(
    cw_id: int,
    request: Request,
    classroom_id: int = Query(..., description="加入课堂时返回的 classroom_id，用于校验课件归属"),
    variant: str = Query("auto", description="auto | original | preview"),
):
    user = _current_user(request)
    if not user:
        return JSONResponse({"error": "未登录"}, status_code=401)
    cw = get_courseware_by_id(cw_id)
    if not cw or cw["classroom_id"] is None or int(cw["classroom_id"]) != int(classroom_id):
        return JSONResponse({"error": "课件不存在或无权访问"}, status_code=404)

    ext = _courseware_ext(cw)
    original_path = _courseware_file_path(cw)
    preview_pdf = _preview_pdf_path(original_path)
    v = (variant or "auto").lower()

    if v == "preview":
        if not os.path.exists(preview_pdf):
            return JSONResponse({"error": "预览文件不存在"}, status_code=404)
        return FileResponse(preview_pdf, media_type="application/pdf", filename=f"{cw['original_name']}.pdf", content_disposition_type="inline")

    if v == "original":
        if not os.path.exists(original_path):
            return JSONResponse({"error": "原文件不存在"}, status_code=404)
        media_type = "application/pdf" if ext == ".pdf" else "application/octet-stream"
        disp = "inline" if ext == ".pdf" else "attachment"
        return FileResponse(original_path, media_type=media_type, filename=cw["original_name"], content_disposition_type=disp)

    # auto
    if ext == ".pdf" and os.path.exists(original_path):
        return FileResponse(original_path, media_type="application/pdf", filename=cw["original_name"], content_disposition_type="inline")
    if os.path.exists(preview_pdf):
        return FileResponse(preview_pdf, media_type="application/pdf", filename=f"{cw['original_name']}.pdf", content_disposition_type="inline")
    if os.path.exists(original_path):
        media_type = "application/octet-stream"
        return FileResponse(original_path, media_type=media_type, filename=cw["original_name"], content_disposition_type="attachment")
    return JSONResponse({"error": "文件不存在"}, status_code=404)

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
    # 先把房间当前成员发给新连接者，避免先后进入顺序导致老师/学生端一直显示“等待”
    members = [conn["user"] for conn in manager.rooms.get(room_key, [])]
    await websocket.send_text(json.dumps({
        "type": "presence_snapshot",
        "members": members,
    }, ensure_ascii=False))
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
