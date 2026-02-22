import streamlit as st
import pandas as pd
import requests
import json
import uuid
import re
import base64
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from io import BytesIO

# =========================
# CONFIG (연구소 리포지토리 설정)
# =========================
REPO_OWNER = "Haemi-Labs"
REPO_NAME = "Prosthetic-Education-AI"
CASES_JSONL_PATH = "data/clinical_cases.jsonl"
DEFAULT_REF = "main"
ASSISTANT_ID = "prosthetic_mentor"

# =========================
# Session 초기화
# =========================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "student_id" not in st.session_state:
    st.session_state.student_id = ""
if "event_log" not in st.session_state:
    st.session_state.event_log = []
if "last_event_ts" not in st.session_state:
    st.session_state.last_event_ts = None
if "event_seq" not in st.session_state:
    st.session_state.event_seq = 0
if "cases" not in st.session_state:
    st.session_state.cases = []
if "selected_case" not in st.session_state:
    st.session_state.selected_case = None

# =========================
# 시간 및 정밀 로깅 헬퍼
# =========================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_utc_iso() -> str:
    return now_utc().isoformat(timespec="microseconds")


def record_clinical_event(values: Dict[str, Any], case_id: str):
    st.session_state.event_seq += 1
    ts = now_utc()
    ts_iso = ts.isoformat(timespec="microseconds")
    delta_s = None
    if st.session_state.last_event_ts is not None:
        delta_s = (ts - st.session_state.last_event_ts).total_seconds()
    st.session_state.last_event_ts = ts

    entry = {
        "event_id": str(uuid.uuid4()),
        "seq": st.session_state.event_seq,
        "case_id": case_id,
        "student_id": st.session_state.student_id or "UNKNOWN_STUDENT",
        "session_id": st.session_state.thread_id,
        "timestamp": ts_iso,
        "delta_s": delta_s,
        "activity": values.get("routing_action") or "UNSPECIFIED",
        "step": values.get("step") or "UNKNOWN",
        "pqi_score": values.get("current_score"),
        "keyword_score_0_100": values.get("keyword_score_0_100"),
        "rationale_score_0_100": values.get("rationale_score_0_100"),
        "srl_code": values.get("srl_table1_code"),
        "mentor_feedback": values.get("mentor_reply"),
        "server_timestamp_iso": values.get("server_timestamp_iso"),
    }
    st.session_state.event_log.append(entry)


# =========================
# GitHub 케이스 로더
# =========================
def fetch_github_file_via_api(owner: str, repo: str, file_path: str, ref: str = "main") -> str:
    token = st.secrets.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GitHub Issue 생성을 위해 st.secrets['GITHUB_TOKEN']이 필요합니다.")
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    r = requests.get(url, headers=headers, params={"ref": ref}, timeout=30)
    r.raise_for_status()
    return base64.b64decode(r.json()["content"]).decode("utf-8")


def parse_jsonl(text: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_cases_from_github(owner: str, repo: str, path: str, ref: str) -> List[Dict[str, Any]]:
    text = fetch_github_file_via_api(owner, repo, path, ref)
    return parse_jsonl(text)


# =========================
# 임상 추론 단계별 데이터 생성
# =========================
def generate_step_data_from_case(case: Dict[str, Any]) -> Dict[str, Any]:
    gold = case.get("gold_standard") or {}

    def gs(*keys: str) -> str:
        for k in keys:
            if k in gold and isinstance(gold[k], str):
                return gold[k].strip()
        return ""

    return {
        "ORIENTATION": {
            "keywords": ["주호소(CC)", "병력", "방사선 소견"],
            "rationale": gs("orientation", "case_overview"),
        },
        "ASSESSMENT": {
            "keywords": ["프로빙 깊이(PD)", "임상 부착 소실(CAL)", "BOP"],
            "rationale": gs("assessment", "diagnosis"),
        },
        "PLANNING": {
            "keywords": ["치료 순서", "치료 목표", "EMD"],
            "rationale": gs("plan", "tx_sequence"),
        },
        "EVALUATION": {
            "keywords": ["자기평가", "예후", "근거"],
            "rationale": gs("evaluation", "reflection"),
        },
    }


# =========================
# LangGraph 스트리밍 및 SSE 파싱
# =========================
def call_langgraph_stream(user_input: str, step: str, step_data: dict):
    url = f"{st.secrets['LANGGRAPH_URL'].rstrip('/')}/v1/runs/stream"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {st.secrets['LANGGRAPH_TOKEN']}",
    }
    payload = {
        "assistant_id": ASSISTANT_ID,
        "input": {
            "messages": [{"role": "user", "content": user_input}],
            "metadata": {
                "student_id": st.session_state.student_id,
                "thread_id": st.session_state.thread_id,
            },
            "step": step,
            "step_data": step_data,
        },
        "config": {"configurable": {"thread_id": st.session_state.thread_id}},
        "stream_mode": "values",
    }

    final_values = None
    mentor_placeholder = st.empty()
    with requests.post(url, json=payload, headers=headers, stream=True, timeout=90) as resp:
        for line in resp.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                payload_str = line[5:].strip()
                if payload_str in ("[DONE]", ""):
                    continue
                chunk = json.loads(payload_str)
                values = chunk.get("data", {}).get("values") or chunk.get("values")
                if values:
                    final_values = values
                    if values.get("mentor_reply"):
                        mentor_placeholder.markdown(f"### Mentor\n{values['mentor_reply']}")
    return final_values


# =========================
# 데이터 저장 및 Export (GitHub Issue)
# =========================
def build_issue_body(records: list[dict], case_id: str) -> str:
    meta = {"case_id": case_id, "student_id": st.session_state.student_id, "created_at": now_utc_iso()}
    jsonl_log = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    return f"AUTO_PIPELINE_META:\n{json.dumps(meta)}\n\nSESSION_LOG:\n```jsonl\n{jsonl_log}\n```"


def create_github_issue(title: str, body: str):
    token = st.secrets["GITHUB_TOKEN"]
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    r = requests.post(url, headers=headers, json={"title": title, "body": body}, timeout=30)
    return r.json()


# =========================
# UI 구성
# =========================
st.set_page_config(layout="wide", page_title="보철 임상 시뮬레이터")
st.title("🦷 보철 임상 학습 프로토콜 v1.0")

left, right = st.columns([1, 1])

with left:
    st.subheader("1) 학생 정보 및 케이스 로드")
    st.session_state.student_id = st.text_input("학번(student_id)", value=st.session_state.student_id)
    if st.button("GitHub에서 케이스 불러오기"):
        try:
            st.session_state.cases = load_cases_from_github(REPO_OWNER, REPO_NAME, CASES_JSONL_PATH, DEFAULT_REF)
            st.success(f"{len(st.session_state.cases)}개의 증례를 로드했습니다.")
        except Exception as e:
            st.error(str(e))

    if st.session_state.cases:
        c_id = st.selectbox("증례 선택", [c["case_id"] for c in st.session_state.cases])
        st.session_state.selected_case = next(c for c in st.session_state.cases if c["case_id"] == c_id)

with right:
    st.subheader("2) 실습 진행")
    if st.session_state.selected_case:
        case = st.session_state.selected_case
        step = st.selectbox("학습 단계", ["ORIENTATION", "ASSESSMENT", "PLANNING", "EVALUATION"], index=1)
        u_input = st.text_area("임상 판단 입력", height=150)
        if st.button("전문가 피드백 받기"):
            sd = generate_step_data_from_case(case)
            res = call_langgraph_stream(u_input, step, sd)
            if res:
                record_clinical_event(res, case["case_id"])
                st.write(f"**PQI 점수:** {res.get('current_score')} | **SRL 코드:** {res.get('srl_table1_code')}")

st.divider()
st.subheader("3) 실습 로그 및 저장")
st.dataframe(pd.DataFrame(st.session_state.event_log), use_container_width=True)

if st.button("GitHub Issue로 실습 종료 및 저장"):
    if st.session_state.event_log:
        title = f"[Log] {st.session_state.student_id} | {st.session_state.selected_case['case_id']}"
        body = build_issue_body(st.session_state.event_log, st.session_state.selected_case["case_id"])
        res = create_github_issue(title, body)
        st.success(f"저장 완료! Issue 번호: {res.get('number')}")
    else:
        st.warning("저장할 로그가 없습니다.")
