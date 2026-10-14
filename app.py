import json
import os
import random
import re
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
GROUPS_FILE = DATA_DIR / "groups.json"

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

client = OpenAI(
    api_key=GROQ_API_KEY or None,
    base_url="https://api.groq.com/openai/v1",
)


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not GROUPS_FILE.exists():
        GROUPS_FILE.write_text("{}", encoding="utf-8")


def load_all_groups():
    ensure_data_dir()
    try:
        return json.loads(GROUPS_FILE.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        return {}


def save_all_groups(store):
    ensure_data_dir()
    GROUPS_FILE.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_students(raw_students):
    students = []
    for item in raw_students or []:
        if not isinstance(item, dict):
            continue
        student_no = str(item.get("studentNo") or item.get("id") or "").strip()
        name = str(item.get("name") or "").strip()
        if not student_no and not name:
            continue
        students.append({
            "studentNo": student_no,
            "name": name or student_no,
            "note": str(item.get("note") or "").strip(),
        })
    return students


def split_into_groups(students, group_count=None, size_per_group=None):
    if not students:
        return []

    total = len(students)
    if size_per_group and size_per_group > 0:
        group_count = max(1, (total + size_per_group - 1) // size_per_group)
    else:
        group_count = max(1, min(int(group_count or 2), total))

    buckets = [[] for _ in range(group_count)]
    for index, student in enumerate(students):
        buckets[index % group_count].append(student)

    return [
        {
            "id": i + 1,
            "name": f"第 {i + 1} 組",
            "members": members,
        }
        for i, members in enumerate(buckets)
        if members
    ]


def payload_from_groups(course_id, groups, method, extra=None):
    data = {
        "course_id": course_id,
        "method": method,
        "group_count": len(groups),
        "groups": groups,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        data.update(extra)
    return data


def persist_course_groups(course_id, payload):
    store = load_all_groups()
    store[course_id] = payload
    save_all_groups(store)
    return payload


def extract_json(text):
    if not text:
        raise ValueError("Groq 沒有回傳內容")
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise ValueError("無法解析 Groq 回傳的分組結果")
        return json.loads(match.group(0))


def groq_group_students(students, group_count, instruction):
    if not GROQ_API_KEY:
        raise RuntimeError("尚未設定 GROQ_API_KEY，請在專案根目錄的 .env 填入金鑰")

    roster = [
        {
            "studentNo": s["studentNo"],
            "name": s["name"],
            "note": s["note"],
        }
        for s in students
    ]
    prompt = (
        "你是課堂助教，請把學生平均分成指定組數。"
        "請只輸出 JSON，不要 markdown、不要說明。"
        "JSON 格式必須是："
        '{"rationale":"簡短中文說明","groups":[{"name":"第 1 組","members":[{"studentNo":"學號","name":"姓名"}]}]}。'
        "每位學生只能出現一次，不可漏人。"
        f"\n組數：{group_count}"
        f"\n額外指示：{instruction or '盡量人數平均，若有班代或備註可適度分散到各組。'}"
        f"\n學生名單：{json.dumps(roster, ensure_ascii=False)}"
    )

    response = client.responses.create(
        input=prompt,
        model=GROQ_MODEL,
    )
    parsed = extract_json(getattr(response, "output_text", "") or "")
    raw_groups = parsed.get("groups") or []
    used = set()
    groups = []
    lookup = {(s["studentNo"], s["name"]): s for s in students}

    for i, group in enumerate(raw_groups):
        members = []
        for member in group.get("members") or []:
            key = (str(member.get("studentNo") or "").strip(), str(member.get("name") or "").strip())
            student = lookup.get(key)
            if not student:
                student = next((s for s in students if s["studentNo"] == key[0] or s["name"] == key[1]), None)
            if not student:
                continue
            identity = student["studentNo"] or student["name"]
            if identity in used:
                continue
            used.add(identity)
            members.append(student)
        if members:
            groups.append({
                "id": i + 1,
                "name": str(group.get("name") or f"第 {i + 1} 組").strip() or f"第 {i + 1} 組",
                "members": members,
            })

    leftover = [s for s in students if (s["studentNo"] or s["name"]) not in used]
    for index, student in enumerate(leftover):
        if not groups:
            groups.append({"id": 1, "name": "第 1 組", "members": []})
        groups[index % len(groups)]["members"].append(student)

    for i, group in enumerate(groups):
        group["id"] = i + 1
        if not group.get("name"):
            group["name"] = f"第 {i + 1} 組"

    return groups, str(parsed.get("rationale") or "").strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "ok": True,
        "groq_configured": bool(GROQ_API_KEY),
        "model": GROQ_MODEL,
    })


@app.route("/api/groups", methods=["GET"])
def get_groups():
    course_id = (request.args.get("course_id") or "").strip()
    store = load_all_groups()
    if course_id:
        return jsonify({"ok": True, "data": store.get(course_id)})
    return jsonify({"ok": True, "data": store})


@app.route("/api/groups/random", methods=["POST"])
def random_groups():
    body = request.get_json(silent=True) or {}
    course_id = str(body.get("course_id") or "default")
    students = normalize_students(body.get("students"))
    if not students:
        return jsonify({"ok": False, "error": "目前課程沒有學生可分組"}), 400

    shuffled = students[:]
    random.shuffle(shuffled)
    groups = split_into_groups(
        shuffled,
        group_count=body.get("group_count"),
        size_per_group=body.get("size_per_group"),
    )
    payload = persist_course_groups(
        course_id,
        payload_from_groups(course_id, groups, "random"),
    )
    return jsonify({"ok": True, "data": payload})


@app.route("/api/groups/manual", methods=["POST"])
def manual_groups():
    body = request.get_json(silent=True) or {}
    course_id = str(body.get("course_id") or "default")
    incoming = body.get("groups") or []
    groups = []
    for i, group in enumerate(incoming):
        members = normalize_students(group.get("members"))
        groups.append({
            "id": i + 1,
            "name": str(group.get("name") or f"第 {i + 1} 組").strip() or f"第 {i + 1} 組",
            "members": members,
        })
    payload = persist_course_groups(
        course_id,
        payload_from_groups(course_id, groups, "manual"),
    )
    return jsonify({"ok": True, "data": payload})


@app.route("/api/groups/ai", methods=["POST"])
def ai_groups():
    body = request.get_json(silent=True) or {}
    course_id = str(body.get("course_id") or "default")
    students = normalize_students(body.get("students"))
    if not students:
        return jsonify({"ok": False, "error": "目前課程沒有學生可分組"}), 400

    group_count = int(body.get("group_count") or 2)
    group_count = max(1, min(group_count, len(students)))
    instruction = str(body.get("instruction") or "").strip()

    try:
        groups, rationale = groq_group_students(students, group_count, instruction)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    payload = persist_course_groups(
        course_id,
        payload_from_groups(course_id, groups, "ai", {"rationale": rationale}),
    )
    return jsonify({"ok": True, "data": payload})


if __name__ == "__main__":
    ensure_data_dir()
    app.run(debug=True, host="127.0.0.1", port=5000)
