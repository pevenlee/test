import streamlit as st
import pandas as pd
import json
import warnings
import os
import re
import numpy as np
import base64
import time
# 确保你已经安装了 google-genai 库
# pip install google-genai
from google import genai
from google.genai import types

# 忽略无关警告
warnings.filterwarnings('ignore')

# ================= 1. 基础配置 =================

st.set_page_config(
    page_title="ChatBI Pro", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 模型配置 ---
MODEL_FAST = "gemini-2.0-flash"       
MODEL_SMART = "gemini-3-pro-preview"      

# --- 常量定义 ---
JOIN_KEY = "药品索引"
FILE_FACT = "fact.csv"      
FILE_DIM = "ipmdata.xlsx"    

try:
    FIXED_API_KEY = st.secrets["GENAI_API_KEY"]
except:
    FIXED_API_KEY = ""

# ================= 2. 视觉体系 (Noir VI - 中文适配版) =================

def inject_custom_css():
    st.markdown("""
        <style>
        /* 引入等宽字体 (英文字体保留，中文使用系统默认无衬线以保清晰) */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');
        
        /* === 全局去圆角 & 线框化 === */
        :root {
            --bg-color: #000000;
            --border-color: #333333;
            --text-color: #E0E0E0;
        }

        /* 强制字体和直角 */
        .stApp, .element-container, .stMarkdown, .stDataFrame, .stButton, div[data-testid="stDataEditor"] {
            font-family: 'JetBrains Mono', "Microsoft YaHei", "SimHei", monospace !important;
        }
        
        div, button, input, select, textarea {
            border-radius: 0px !important;
        }

        /* === 顶部导航栏 (Logo 左上角) === */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
            border: none !important;
            pointer-events: none; 
            z-index: 99;
        }
        
        header[data-testid="stHeader"] > div:first-child {
            background: transparent;
        }

        /* 自定义顶部导航栏 */
        .fixed-header-container {
            position: fixed; top: 0; left: 0; width: 100%; height: 60px;
            background-color: rgba(0,0,0,0.95);
            border-bottom: 1px solid var(--border-color);
            z-index: 999990; 
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 24px;
            backdrop-filter: blur(5px);
        }
        
        .nav-left { display: flex; align-items: center; gap: 12px; }
        .nav-logo-icon { width: 24px; height: 24px; background: #fff; display: flex; align-items: center; justify-content: center; }
        .nav-logo-text { font-weight: 700; font-size: 20px; color: #FFF; letter-spacing: -1px; }
        .nav-right { display: flex; align-items: center; gap: 16px; }
        .nav-tag { font-size: 10px; background: #FFF; color: #000; padding: 2px 6px; font-weight: bold; }

        .block-container { padding-top: 80px !important; max-width: 1200px; }
        footer { display: none !important; }

        /* === 侧边栏折叠按钮位置修改 === */
        section[data-testid="stSidebar"] button[kind="header"] {
            position: absolute !important;
            bottom: 20px !important;
            right: 20px !important;
            top: auto !important;
            left: auto !important;
            background-color: #111 !important;
            border: 1px solid #333 !important;
            color: #fff !important;
            z-index: 999999 !important;
            width: 32px; height: 32px;
            pointer-events: auto;
        }
        section[data-testid="stSidebar"] button[kind="header"]:hover {
            border-color: #fff !important;
        }

        [data-testid="stSidebarCollapsedControl"] {
            position: fixed !important;
            bottom: 20px !important;
            left: 20px !important;
            top: auto !important;
            z-index: 999999 !important;
            background-color: #000 !important;
            border: 1px solid #333 !important;
            color: #fff !important;
            padding: 4px !important;
            pointer-events: auto;
            visibility: visible !important;
            display: flex !important;
        }

        /* === 底部输入框 === */
        [data-testid="stBottom"] {
            background-color: transparent !important;
            border-top: 1px solid var(--border-color);
            padding-bottom: 20px;
        }
        
        .stChatInputContainer textarea {
            background-color: #050505 !important;
            color: #fff !important;
            border: 1px solid #333 !important;
        }
        
        .stChatInputContainer textarea:focus {
            border-color: #fff !important;
            box-shadow: none !important;
        }

        /* === 侧边栏 & 表格 === */
        [data-testid="stSidebar"] {
            background-color: #050505;
            border-right: 1px solid var(--border-color);
        }
        
        [data-testid="stDataFrame"] {
            background-color: #000 !important;
            border: 1px solid #333;
        }
        
        /* === 消息气泡 & 头像 === */
        [data-testid="stChatMessage"] {
            background-color: transparent !important;
            border: none !important;
            padding: 10px 0 !important;
        }
        
        [data-testid="stChatMessageAvatarBackground"] {
            background-color: #111 !important;
            border: 1px solid #333;
            color: #fff !important;
        }

        /* === 按钮样式 === */
        div.stButton > button {
            background-color: #000 !important;
            color: #888 !important;
            border: 1px solid #333 !important;
            text-transform: uppercase;
            font-size: 12px;
            transition: all 0.3s;
        }
        div.stButton > button:hover {
            color: #fff !important;
            border-color: #fff !important;
            background-color: #111 !important;
        }

        /* 卡片样式 */
        .protocol-box {
            background-color: #0A0A0A; padding: 15px; 
            border: 1px solid #333; margin-bottom: 15px;
            font-size: 12px;
        }
        .protocol-title { 
            font-weight: 700; color: #FFF; margin-bottom: 10px; 
            border-bottom: 1px solid #222; padding-bottom: 5px;
        }
        .protocol-row { display: flex; margin-bottom: 5px; }
        .protocol-label { width: 80px; color: #666; font-size: 10px; }
        .protocol-val { color: #CCC; }

        .insight-box {
            background: #0A0A0A; padding: 20px; 
            border: 1px solid #333; border-left: 2px solid #FFF;
            color: #CCC; font-size: 14px; line-height: 1.6;
        }
        
        .mini-insight {
            color: #666; font-size: 12px; font-style: italic;
            border-top: 1px solid #222; margin-top: 10px; padding-top: 5px;
        }
        
        </style>
    """, unsafe_allow_html=True)

# ================= 3. 核心工具函数 =================

@st.cache_resource
def get_client():
    if not FIXED_API_KEY: return None
    try: return genai.Client(api_key=FIXED_API_KEY, http_options={'api_version': 'v1beta'})
    except Exception as e: st.error(f"SDK Error: {e}"); return None

# --- 数据读取 ---
@st.cache_data
def load_local_data(filename):
    if not os.path.exists(filename): return None
    df = None
    try:
        df = pd.read_excel(filename, engine='openpyxl')
    except:
        try: df = pd.read_csv(filename)
        except: 
            try: df = pd.read_csv(filename, encoding='gbk')
            except: return None

    if df is not None:
        df.columns = df.columns.str.strip()
        if JOIN_KEY in df.columns:
            df[JOIN_KEY] = df[JOIN_KEY].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
            
        for col in df.columns:
            if df[col].dtype == 'object': df[col] = df[col].astype(str)
            if any(k in str(col) for k in ['额', '量', 'Sales', 'Qty']):
                try: df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
                except: pass
            
            if any(k in str(col).lower() for k in ['日期', 'date', 'time', 'year', 'month', 'quarter', '年', '月', '季']):
                try: 
                    df[col] = pd.to_datetime(df[col], errors='coerce').fillna(df[col])
                    if df[col].dtype.kind == 'M' and any(x in str(col).lower() for x in ['季', 'quarter']):
                         df[col] = df[col].dt.to_period('Q').astype(str)
                except: pass
        return df
    return None

def get_dataframe_info(df, name="df"):
    if df is None: return f"{name}: NULL"
    info = [f"表名: `{name}` ({len(df)} 行)"]
    info.append("| 字段 | 类型 | 示例 |")
    info.append("|---|---|---|")
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = list(df[col].dropna().unique()[:5])
        info.append(f"| {col} | {dtype} | {str(sample)} |")
    return "\n".join(info)

def clean_json_string(text):
    try: return json.loads(text)
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except: pass
    return None

def safe_generate(client, model, prompt, mime_type="text/plain"):
    config = types.GenerateContentConfig(response_mime_type=mime_type)
    try: return client.models.generate_content(model=model, contents=prompt, config=config)
    except Exception as e: return type('obj', (object,), {'text': f"Error: {e}"})

def format_display_df(df):
    if not isinstance(df, pd.DataFrame): return df
    df_fmt = df.copy()
    for col in df_fmt.columns:
        col_str = str(col).lower()
        is_numeric = pd.api.types.is_numeric_dtype(df_fmt[col])
        if not is_numeric and df_fmt[col].dtype == 'object':
            try:
                temp = pd.to_numeric(df_fmt[col], errors='coerce')
                if temp.notnull().sum() > 0: is_numeric = True; df_fmt[col] = temp
            except: pass

        if is_numeric:
            if col_str in ['year', '年份', '年']:
                try: df_fmt[col] = df_fmt[col].fillna(0).astype(int).astype(str).replace('0', '-')
                except: pass
            elif any(x in col_str for x in ['率', '比', 'ratio', '%']):
                df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "-")
            else:
                df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "-")
        else:
            if pd.api.types.is_datetime64_any_dtype(df_fmt[col]):
                df_fmt[col] = df_fmt[col].dt.strftime('%Y-%m-%d')
    return df_fmt

def normalize_result(res):
    if res is None: return pd.DataFrame()
    if isinstance(res, pd.DataFrame): return res
    if isinstance(res, pd.Series): return res.to_frame(name='数值').reset_index()
    if isinstance(res, dict):
        try: return pd.DataFrame([res]) 
        except: return pd.DataFrame(list(res.items()), columns=['键', '值'])
    if isinstance(res, list): return pd.DataFrame(res)
    return pd.DataFrame([str(res)], columns=['结果'])

def safe_check_empty(df):
    if df is None: return True
    if not isinstance(df, pd.DataFrame): return True
    return df.empty

def get_history_context(limit=5):
    history_msgs = st.session_state.messages[:-1] 
    relevant_msgs = history_msgs[-(limit * 2):]
    context_str = ""
    if not relevant_msgs: return "无历史记录"
    for msg in relevant_msgs:
        role = "用户" if msg["role"] == "user" else "AI"
        content = msg["content"]
        if msg["type"] == "df": content = "[已展示数据表]"
        context_str += f"{role}: {content}\n"
    return context_str

def render_protocol_card(summary):
    st.markdown(f"""
    <div class="protocol-box">
        <div class="protocol-title">执行协议 (EXECUTION PROTOCOL)</div>
        <div class="protocol-row"><div class="protocol-label">意图</div><div class="protocol-val">{summary.get('intent', '-')}</div></div>
        <div class="protocol-row"><div class="protocol-label">范围</div><div class="protocol-val">{summary.get('scope', '-')}</div></div>
        <div class="protocol-row"><div class="protocol-label">关键词</div><div class="protocol-val">{summary.get('key_match', 'N/A')}</div></div>
        <div class="protocol-row"><div class="protocol-label">逻辑</div><div class="protocol-val">{summary.get('logic', '-')}</div></div>
    </div>
    """, unsafe_allow_html=True)

def handle_followup(question):
    st.session_state.messages.append({"role": "user", "type": "text", "content": question})

def safe_exec_code(code_str, context):
    context.update({"pd": pd, "np": np, "st": st})
    context['result'] = None
    pre_vars = set(context.keys())
    try:
        exec(code_str, context)
        if context.get('result') is not None: return context['result']
        # 启发式查找 DataFrame
        new_vars = set(context.keys()) - pre_vars
        candidates = []
        for var in new_vars:
            if var not in ["pd", "np", "st", "__builtins__", "result"]:
                val = context[var]
                if isinstance(val, (pd.DataFrame, pd.Series)): candidates.append(val)
        if candidates: return candidates[-1]
        return None
    except Exception as e: raise e

# ================= 4. 页面渲染 =================

inject_custom_css()
client = get_client()

df_sales = load_local_data(FILE_FACT)
df_product = load_local_data(FILE_DIM)

# --- Top Nav (With Logo) ---
svg_logo = """
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M4 4H20V20H4V4Z" fill="white"/>
<path d="M8 8H16V16H8V8Z" fill="black"/>
</svg>
"""
st.markdown(f"""
<div class="fixed-header-container">
    <div class="nav-left">
        <div class="nav-logo-icon">{svg_logo}</div>
        <div class="nav-logo-text">ChatBI.PRO</div>
    </div>
    <div class="nav-right">
        <div class="nav-tag">管理员</div>
        <button style="background:transparent; border:1px solid #333; color:#666; padding:4px 12px; cursor:pointer;" onclick="alert('已安全退出')">退出</button>
    </div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 系统状态")
    if df_sales is not None:
        st.markdown(f"[OK] {FILE_FACT} 已加载")
        
        # 尝试检测时间范围
        date_cols = df_sales.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns
        if len(date_cols) > 0:
            min_str = df_sales[date_cols[0]].min().strftime('%Y-%m-%d')
            max_str = df_sales[date_cols[0]].max().strftime('%Y-%m-%d')
            st.caption(f"数据范围: {min_str} -> {max_str}")
            
        st.divider()
        st.markdown("**字段结构:**")
        st.dataframe(pd.DataFrame(df_sales.columns, columns=["字段名"]), height=150, hide_index=True)
    else:
        st.markdown(f"[错误] {FILE_FACT} 文件缺失")

    if df_product is not None:
        st.markdown(f"[OK] {FILE_DIM} 已加载")
    else:
        st.markdown(f"[错误] {FILE_DIM} 文件缺失")

    st.divider()
    if st.button("清除会话记忆", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)

# --- Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=None):
        if msg["type"] == "text": 
            prefix = ">> " if msg["role"] == "assistant" else ""
            st.markdown(prefix + msg["content"])
        elif msg["type"] == "df": 
            st.dataframe(msg["content"], use_container_width=True)

# --- 猜你想问 ---
if not st.session_state.messages:
    st.markdown("### 初始化查询")
    c1, c2, c3 = st.columns(3)
    def handle_preset(question):
        st.session_state.messages.append({"role": "user", "type": "text", "content": question})
        st.rerun()
    if c1.button("肿瘤产品表现"): handle_preset("肿瘤产品的市场表现如何?")
    if c2.button("查询K药销售"): handle_preset("查一下K药最近的销售额")
    if c3.button("过亿独家品种"): handle_preset("销售额过亿的，独家创新药有哪些")

# --- Input ---
query = st.chat_input("请输入指令...")
if query:
    st.session_state.messages.append({"role": "user", "type": "text", "content": query})
    with st.chat_message("user", avatar=None):
        st.markdown(query)

# --- Core Logic ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    
    try:
        user_query = st.session_state.messages[-1]["content"]
        history_str = get_history_context(limit=5)

        with st.chat_message("assistant", avatar=None):
            if df_sales is None or df_product is None:
                st.error("数据源缺失，请检查上传文件。")
                st.stop()

            context_info = f"""
            {get_dataframe_info(df_sales, "df_sales")}
            {get_dataframe_info(df_product, "df_product")}
            关联键: `{JOIN_KEY}`
            """

            # 1. 意图识别
            with st.status("正在分析意图...", expanded=False) as status:
                prompt_router = f"""
                Classify intent based on history and query.
                History: {history_str}
                Query: "{user_query}"
                Categories: 
                1. inquiry (simple data fetch, filter)
                2. analysis (why, trend, breakdown, evaluation)
                3. irrelevant
                Output JSON: {{ "type": "inquiry/analysis/irrelevant" }}
                """
                resp = safe_generate(client, MODEL_FAST, prompt_router, "application/json")
                if "Error" in resp.text:
                    status.update(label="API 连接错误", state="error")
                    st.stop()
                intent = clean_json_string(resp.text).get('type', 'inquiry')
                status.update(label=f"意图识别: {intent.upper()}", state="complete")

            # 2. 简单查询
            if intent == 'inquiry':
                with st.spinner("正在构建代码..."):
                    prompt_code = f"""
                    Role: Python Data Expert.
                    History: {history_str}
                    Query: "{user_query}"
                    Context: {context_info}
                    
                    Rules:
                    1. Use `pd.merge` if needed.
                    2. Define ALL variables used.
                    3. No display functions (print/plot).
                    4. Assign final result to variable `result`.
                    
                    Output JSON (Translate summary values to Chinese): 
                    {{ "summary": {{ "intent": "数据查询", "scope": "...", "metrics": "...", "key_match": "...", "logic": "..." }}, "code": "..." }}
                    """
                    resp_code = safe_generate(client, MODEL_SMART, prompt_code, "application/json")
                    plan = clean_json_string(resp_code.text)
                
                if plan:
                    s = plan.get('summary', {})
                    render_protocol_card(s)

                    try:
                        exec_ctx = {"df_sales": df_sales, "df_product": df_product}
                        res_raw = safe_exec_code(plan['code'], exec_ctx)
                        res_df = normalize_result(res_raw)
                        
                        if not safe_check_empty(res_df):
                            formatted_df = format_display_df(res_df)
                            st.dataframe(formatted_df, use_container_width=True)
                            st.session_state.messages.append({"role": "assistant", "type": "df", "content": formatted_df})
                        else:
                            st.warning("无精确匹配，尝试模糊搜索...")
                            fallback_code = f"result = df_product[df_product.astype(str).apply(lambda x: x.str.contains('{user_query[:2]}', case=False, na=False)).any(axis=1)].head(10)"
                            try:
                                res_fallback = safe_exec_code(fallback_code, exec_ctx)
                                res_fallback = normalize_result(res_fallback)
                                if not safe_check_empty(res_fallback):
                                    st.dataframe(res_fallback)
                                    st.session_state.messages.append({"role": "assistant", "type": "df", "content": res_fallback})
                                else:
                                    st.error("未找到相关数据。")
                            except:
                                st.error("未找到相关数据。")
                    except Exception as e:
                        st.error(f"代码执行错误: {e}")

            # 3. 深度分析
            elif intent == 'analysis':
                shared_ctx = {
                    "df_sales": df_sales.copy(), 
                    "df_product": df_product.copy(), 
                }

                with st.spinner("规划分析路径..."):
                    prompt_plan = f"""
                    Role: Senior Analyst.
                    History: {history_str}
                    Query: "{user_query}"
                    Context: {context_info}
                    
                    Task: Create 2-4 analysis angles.
                    Rules:
                    1. Share context variables between steps.
                    2. Assign result of each step to `result`.
                    3. Ensure `result` is a DataFrame.
                    
                    Output JSON (Use Chinese for title/desc/intent_analysis): 
                    {{ "intent_analysis": "...", "angles": [ {{ "title": "...", "desc": "...", "summary": {{ "intent": "...", "scope": "...", "metrics": "...", "key_match": "...", "logic": "..." }}, "code": "..." }} ] }}
                    """
                    resp_plan = safe_generate(client, MODEL_SMART, prompt_plan, "application/json")
                    plan_json = clean_json_string(resp_plan.text)
                
                if plan_json:
                    intro = f"### 意图分析\n{plan_json.get('intent_analysis')}"
                    st.markdown(intro)
                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": intro})
                    
                    angles_data = []
                    
                    for angle in plan_json.get('angles', []):
                        with st.container():
                            st.markdown(f"**{angle['title']}**")
                            st.caption(angle['desc'])
                            
                            if 'summary' in angle:
                                render_protocol_card(angle['summary'])
                            
                            try:
                                res_raw = safe_exec_code(angle['code'], shared_ctx)
                                res_df = normalize_result(res_raw)
                                
                                if not safe_check_empty(res_df):
                                    formatted_df = format_display_df(res_df)
                                    st.dataframe(formatted_df, use_container_width=True)
                                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"**{angle['title']}**"})
                                    st.session_state.messages.append({"role": "assistant", "type": "df", "content": formatted_df})
                                    
                                    prompt_mini = f"Interpret data (1 sentence) in Chinese (用中文解释):\n{res_df.to_string()}"
                                    resp_mini = safe_generate(client, MODEL_FAST, prompt_mini)
                                    explanation = resp_mini.text
                                    st.markdown(f'<div class="mini-insight">>> {explanation}</div>', unsafe_allow_html=True)
                                    angles_data.append({"title": angle['title'], "explanation": explanation})
                                else:
                                    st.warning(f"无数据: {angle['title']}")
                            except Exception as e:
                                st.error(f"执行错误: {e}")

                    if angles_data:
                        with st.spinner("正在综合结论..."):
                            findings = "\n".join([f"[{a['title']}]: {a['explanation']}" for a in angles_data])
                            prompt_final = f"""Based on findings: {findings}, answer: "{user_query}". Professional tone, markdown format. RESPONSE IN CHINESE (中文回答)."""
                            resp_final = safe_generate(client, MODEL_SMART, prompt_final)
                            insight = resp_final.text
                            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
                            st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"### 分析总结\n{insight}"})

                        with st.spinner("..."):
                            prompt_next = f"""
                            Suggest 2 follow-up questions based on: {insight} in Chinese (中文).
                            Output JSON List: ["Q1", "Q2"]
                            """
                            resp_next = safe_generate(client, MODEL_FAST, prompt_next, "application/json")
                            next_questions = clean_json_string(resp_next.text)

                        if isinstance(next_questions, list) and len(next_questions) > 0:
                            st.markdown("### 建议追问")
                            c1, c2 = st.columns(2)
                            
                            c1.button(f"> {next_questions[0]}", use_container_width=True, on_click=handle_followup, args=(next_questions[0],))
                                
                            if len(next_questions) > 1:
                                c2.button(f"> {next_questions[1]}", use_container_width=True, on_click=handle_followup, args=(next_questions[1],))
            else:
                st.info("仅限数据查询")
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": "仅限数据查询"})

    except Exception as e:
        st.error(f"系统异常: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant", 
            "type": "text", 
            "content": "系统错误，请重试。"
        })