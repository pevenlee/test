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
    page_title="ChatBI by Pharmcube", 
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
LOGO_FILE = "logo.png"

# [头像定义]
USER_AVATAR = "clt.png"  # 用户头像文件名
BOT_AVATAR = "pmc.png"   # AI头像文件名

try:
    FIXED_API_KEY = st.secrets["GENAI_API_KEY"]
except:
    FIXED_API_KEY = ""

# ================= 2. 视觉体系 (Noir UI - 全中文版 - 圆角优化) =================

def get_base64_image(image_path):
    """读取本地图片并转为 Base64"""
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');
        
        :root {
            --bg-color: #050505;
            --sidebar-bg: #000000;
            --border-color: #333333;
            --text-primary: #E0E0E0;
            --accent-error: #FF3333;
            --radius-md: 8px; /* [修改] 定义通用圆角变量 */
        }

        /* 全局字体 */
        .stApp, .element-container, .stMarkdown, .stDataFrame, .stButton, div[data-testid="stDataEditor"] {
            font-family: "Microsoft YaHei", "SimHei", 'JetBrains Mono', monospace !important;
            background-color: var(--bg-color);
        }
        
        /* [修改] 全局圆角设置 */
        div, input, select, textarea { border-radius: var(--radius-md) !important; }
        
        /* [修改] 按钮样式：左对齐 + 圆角 */
        .stButton button {
            border-radius: var(--radius-md) !important;
            text-align: left !important;
            justify-content: flex-start !important; /* 强制内容左对齐 */
            padding-left: 15px !important;
            border: 1px solid #333 !important;
            background: #111 !important;
            color: #CCC !important;
            transition: all 0.2s ease;
        }
        .stButton button:hover {
            border-color: #666 !important;
            color: #FFF !important;
            background: #222 !important;
        }
        .stButton button p {
            text-align: left !important; /* 再次强制内部文字左对齐 */
            font-size: 13px !important;
        }

        /* === 顶部导航栏 === */
        header[data-testid="stHeader"] { background: transparent !important; pointer-events: none; z-index: 90; }
        header[data-testid="stHeader"] > div:first-child { background: transparent; }

        .fixed-header-container {
            position: fixed; top: 0; left: 0; width: 100%; height: 60px;
            background-color: rgba(0,0,0,0.95);
            border-bottom: 1px solid var(--border-color);
            z-index: 999990; 
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 24px;
        }
        
        .nav-left { display: flex; align-items: center; gap: 12px; }
        .nav-logo-img { height: 28px; width: auto; }
        .nav-logo-text { font-weight: 700; font-size: 18px; color: #FFF; letter-spacing: -0.5px; }
        
        /* [修改] 右上角头像容器 */
        .nav-right { display: flex; align-items: center; gap: 12px; }
        .user-avatar-circle {
            width: 36px; height: 36px;
            border-radius: 50%;
            border: 1px solid #444;
            overflow: hidden;
            display: flex; align-items: center; justify-content: center;
            background: #111;
        }
        .user-avatar-circle img { width: 100%; height: 100%; object-fit: cover; }

        .block-container { padding-top: 80px !important; max-width: 1200px; }
        footer { display: none !important; }

        /* === 侧边栏按钮位置 === */
        section[data-testid="stSidebar"] button[kind="header"] {
            position: absolute !important; bottom: 20px !important; right: 20px !important; top: auto !important; left: auto !important;
            background-color: #111 !important; border: 1px solid #333 !important; color: #fff !important;
            width: 32px; height: 32px; z-index: 999999; pointer-events: auto;
            border-radius: var(--radius-md) !important;
        }
        [data-testid="stSidebarCollapsedControl"] {
            position: fixed !important; bottom: 20px !important; left: 20px !important;
            background-color: #000 !important; border: 1px solid #333 !important; color: #fff !important;
            z-index: 999999;
            border-radius: var(--radius-md) !important;
        }

        /* === 错误提示美化 === */
        .stAlert { display: none; }
        .custom-error {
            background-color: rgba(40, 0, 0, 0.9); border: 1px solid var(--accent-error); color: #ffcccc;
            padding: 15px; font-size: 13px; margin-bottom: 1rem; display: flex; align-items: center; gap: 10px;
            border-radius: var(--radius-md);
        }
        .custom-error::before { content: "[错误]"; color: var(--accent-error); font-weight: bold; }

        /* === 侧边栏 & 表格 === */
        [data-testid="stSidebar"] { background-color: var(--sidebar-bg); border-right: 1px solid var(--border-color); }
        [data-testid="stDataFrame"] { background-color: #000 !important; border: 1px solid #333; border-radius: var(--radius-md); }
        
        .field-tag {
            display: inline-block; background: #111; border: 1px solid #333; 
            color: #888; font-size: 10px; padding: 2px 6px; margin: 2px;
            border-radius: 4px; /* 小圆角 */
        }

        /* === 聊天气泡 & 头像 === */
        [data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 10px 0 !important; }
        
        [data-testid="stChatMessageAvatarBackground"] { 
            background-color: #000000 !important; 
            border: 1px solid #ffffff !important;
            color: #ffffff !important;
            box-shadow: none !important;
            display: flex !important;
        }
        .stChatMessage .stChatMessageAvatarImage {
            width: 100%; height: 100%; object-fit: cover;
            border-radius: 50%; /* 头像强制圆形 */
        }
        
        .msg-prefix { font-weight: bold; margin-right: 8px; font-size: 12px; }
        .p-user { color: #888; }
        .p-ai { color: #00FF00; }

        /* === 底部输入框 === */
        [data-testid="stBottom"] { background: transparent !important; border-top: 1px solid var(--border-color); }
        .stChatInputContainer textarea { 
            background: #050505 !important; color: #fff !important; 
            border: 1px solid #333 !important; 
            border-radius: var(--radius-md) !important;
        }
        
        /* === 思考过程 (Thinking Box) === */
        .thought-box {
            font-family: 'JetBrains Mono', "Microsoft YaHei", monospace;
            font-size: 12px; color: #888;
            border-left: 2px solid #444;
            background: #080808; /* 轻微背景色以突显圆角 */
            padding: 10px;
            margin-bottom: 10px;
            text-align: left !important;
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
        }
        .thought-header { font-weight: bold; color: #AAA; margin-bottom: 4px; display: block; }
        
        /* Streamlit Expander */
        .streamlit-expanderHeader {
            background-color: #0A0A0A !important; color: #888 !important;
            border: 1px solid #222 !important; font-size: 12px !important;
            border-radius: var(--radius-md) !important;
        }
        .streamlit-expanderContent {
            background-color: #050505 !important; border: 1px solid #222 !important;
            border-top: none !important; color: #CCC !important;
            border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
        }

        /* 协议卡片 */
        .protocol-box { 
            background: #0A0A0A; padding: 12px; border: 1px solid #222; 
            margin-bottom: 15px; font-size: 12px; 
            text-align: left !important;
            border-radius: var(--radius-md); /* [修改] 圆角 */
        }
        .protocol-row { display: flex; justify-content: space-between; border-bottom: 1px dashed #222; padding: 4px 0; }
        .protocol-key { color: #555; } .protocol-val { color: #CCC; }
        
        /* 洞察框 */
        .insight-box { 
            background: #0A0A0A; padding: 15px; border-left: 3px solid #FFF; color: #DDD; margin-top: 10px; 
            text-align: left !important;
            border-radius: 0 var(--radius-md) var(--radius-md) 0; /* [修改] 圆角 */
        }
        .mini-insight { color: #666; font-size: 12px; font-style: italic; border-top: 1px solid #222; margin-top: 8px; padding-top: 4px; }
        </style>
    """, unsafe_allow_html=True)

# ================= 3. 核心工具函数 =================

@st.cache_resource
def get_client():
    if not FIXED_API_KEY: return None
    try: return genai.Client(api_key=FIXED_API_KEY, http_options={'api_version': 'v1beta'})
    except Exception as e: st.error(f"SDK Error: {e}"); return None

@st.cache_data
def load_local_data(filename):
    if not os.path.exists(filename): return None
    df = None
    try:
        if filename.endswith('.xlsx'):
            df = pd.read_excel(filename, engine='openpyxl')
        else:
            try: df = pd.read_csv(filename)
            except: df = pd.read_csv(filename, encoding='gbk')
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
            if any(k in str(col).lower() for k in ['日期', 'date', 'time', 'year', 'month']):
                try: df[col] = pd.to_datetime(df[col], errors='coerce').fillna(df[col])
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
        sample = list(df[col].dropna().unique()[:3])
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
    """普通生成（同步）"""
    config = types.GenerateContentConfig(response_mime_type=mime_type)
    try: return client.models.generate_content(model=model, contents=prompt, config=config)
    except Exception as e: return type('obj', (object,), {'text': f"Error: {e}"})

def stream_generate(client, model, prompt):
    """流式生成内容，用于 st.write_stream 实现打字机效果"""
    try:
        # 修正点：使用 generate_content_stream 方法，并移除 stream=True 参数
        response = client.models.generate_content_stream(
            model=model, 
            contents=prompt, 
            config=types.GenerateContentConfig(response_mime_type="text/plain"),
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Stream Error: {e}"

def format_display_df(df):
    if not isinstance(df, pd.DataFrame): return df
    df_fmt = df.copy()
    for col in df_fmt.columns:
        if pd.api.types.is_numeric_dtype(df_fmt[col]):
            if "year" in str(col).lower() or "年" in str(col):
                df_fmt[col] = df_fmt[col].apply(lambda x: str(int(x)) if pd.notnull(x) else "-")
            else:
                df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:,.2f}".rstrip('0').rstrip('.') if pd.notnull(x) else "-")
        elif pd.api.types.is_datetime64_any_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].dt.strftime('%Y-%m-%d')
    return df_fmt

def normalize_result(res):
    if res is None: return pd.DataFrame()
    if isinstance(res, pd.DataFrame): return res
    if isinstance(res, pd.Series): return res.to_frame(name='数值').reset_index()
    if isinstance(res, dict): return pd.DataFrame(list(res.items()), columns=['Key', 'Value'])
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
        <div class="protocol-row"><span class="protocol-key">意图识别</span><span class="protocol-val">{summary.get('intent', '-')}</span></div>
        <div class="protocol-row"><span class="protocol-key">逻辑策略</span><span class="protocol-val">{summary.get('logic', '-')}</span></div>
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
        new_vars = set(context.keys()) - pre_vars
        candidates = []
        for var in new_vars:
            if var not in ["pd", "np", "st", "__builtins__", "result"]:
                val = context[var]
                if isinstance(val, (pd.DataFrame, pd.Series)): candidates.append(val)
        if candidates: return candidates[-1]
        return None
    except Exception as e: raise e

def get_avatar(role):
    """根据角色获取头像，如果图片存在则返回路径，否则返回None"""
    if role == "user":
        return USER_AVATAR if os.path.exists(USER_AVATAR) else None
    else:
        return BOT_AVATAR if os.path.exists(BOT_AVATAR) else None

# ================= 4. 页面渲染 =================

inject_custom_css()
client = get_client()

df_sales = load_local_data(FILE_FACT)
df_product = load_local_data(FILE_DIM)

# --- Top Nav (修改：右上角改为头像) ---
logo_b64 = get_base64_image(LOGO_FILE)
if logo_b64:
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="nav-logo-img">'
else:
    logo_html = """<svg width="24" height="24" viewBox="0 0 24 24" fill="white"><path d="M12 2L2 22h20L12 2zm0 3.5L19 20H5l7-14.5z"/></svg>"""

# 处理右上角用户头像
user_avatar_b64 = get_base64_image(USER_AVATAR)
if user_avatar_b64:
    user_avatar_html = f'<div class="user-avatar-circle"><img src="data:image/png;base64,{user_avatar_b64}"></div>'
else:
    # 默认头像
    user_avatar_html = '<div class="user-avatar-circle" style="color:#FFF; font-size:10px;">User</div>'

st.markdown(f"""
<div class="fixed-header-container">
    <div class="nav-left">
        <div class="nav-logo-icon">{logo_html}</div>
        <div class="nav-logo-text">ChatBI</div>
    </div>
    <div class="nav-right">
        <div class="nav-tag">Peiwen</div>
        {user_avatar_html}
    </div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 系统状态 SYSTEM STATUS")
    
    if df_sales is not None:
        st.markdown(f"<span style='color:#00FF00'>[正常]</span> {FILE_FACT}", unsafe_allow_html=True)
        st.markdown(f"<div style='margin-bottom:5px; color:#666; font-size:10px'>包含字段 ({len(df_sales.columns)}):</div>", unsafe_allow_html=True)
        cols_html = "".join([f"<span class='field-tag'>{c}</span>" for c in df_sales.columns])
        st.markdown(f"<div>{cols_html}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:#FF3333'>[错误]</span> {FILE_FACT} 缺失", unsafe_allow_html=True)
        cwd = os.getcwd()
        try: files_in_dir = os.listdir(cwd)
        except: files_in_dir = []
        st.markdown(f"""
        <div style='font-size:10px; color:#888; background:#111; padding:5px; margin-top:5px; border:1px solid #333'>
        <b>路径诊断:</b><br>
        当前目录: {cwd}<br>
        文件列表: {str(files_in_dir)[:100]}...
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if df_product is not None:
        st.markdown(f"<span style='color:#00FF00'>[正常]</span> {FILE_DIM}", unsafe_allow_html=True)
        cols_html = "".join([f"<span class='field-tag'>{c}</span>" for c in df_product.columns])
        st.markdown(f"<div>{cols_html}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:#FF3333'>[错误]</span> {FILE_DIM} 缺失", unsafe_allow_html=True)

    st.divider()
    if st.button("清除对话记忆", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)

# --- Chat History ---
for msg in st.session_state.messages:
    avatar_file = get_avatar(msg["role"])
    with st.chat_message(msg["role"], avatar=avatar_file):
        if msg["type"] == "text": 
            role_class = "p-ai" if msg["role"] == "assistant" else "p-user"
            prefix = "系统 > " if msg["role"] == "assistant" else "用户 > "
            st.markdown(f"<span class='msg-prefix {role_class}'>{prefix}</span>{msg['content']}", unsafe_allow_html=True)
        elif msg["type"] == "df": 
            st.dataframe(msg["content"], use_container_width=True)
        elif msg["type"] == "error":
            st.markdown(f'<div class="custom-error">{msg["content"]}</div>', unsafe_allow_html=True)

# --- 猜你想问 (左对齐按钮) ---
if not st.session_state.messages:
    st.markdown("### 通过与医药魔方交流，开启对医药市场的探索吧！")
    c1, c2, c3 = st.columns(3)
    def handle_preset(question):
        st.session_state.messages.append({"role": "user", "type": "text", "content": question})
        st.rerun()
    # 按钮内部文字已在 CSS 中强制左对齐
    if c1.button("肿瘤产品的市场表现如何，哪些产品在驱动着市场的增长?"): handle_preset("肿瘤产品的市场表现如何，哪些产品在驱动着市场的增长?")
    if c2.button("查一下K药最近2年的销售额"): handle_preset("查一下K药最近2年的销售额")
    if c3.button("销售额过亿的，独家创新药有哪些，总结一下他们的画像"): handle_preset("销售额过亿的，独家创新药有哪些，总结一下他们的画像")

# --- Input ---
query = st.chat_input("请输入指令...")
if query:
    st.session_state.messages.append({"role": "user", "type": "text", "content": query})
    st.rerun()

# --- Core Logic ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    
    try:
        user_query = st.session_state.messages[-1]["content"]
        history_str = get_history_context(limit=5)

        with st.chat_message("assistant", avatar=get_avatar("assistant")):
            if df_sales is None or df_product is None:
                err_text = f"数据源缺失。请检查侧边栏路径诊断。 (需要文件: {FILE_FACT}, {FILE_DIM})"
                st.markdown(f'<div class="custom-error">{err_text}</div>', unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "type": "error", "content": err_text})
                st.stop()

            context_info = f"""
            {get_dataframe_info(df_sales, "df_sales")}
            {get_dataframe_info(df_product, "df_product")}
            KEY: `{JOIN_KEY}`
            """

            # ================= 1. 意图识别 =================
            intent = "inquiry"
            
            with st.status("正在分析意图...", expanded=False) as status:
                prompt_router = f"""
                Classify intent based on context.
                History: {history_str}
                Query: "{user_query}"
                Rules:
                1. Specific numbers/data -> "inquiry"
                2. Trends/reasons/breakdown -> "analysis"
                3. Unrelated -> "irrelevant"
                Output JSON: {{ "type": "result_value" }} (one of: "inquiry", "analysis", "irrelevant")
                """
                resp = safe_generate(client, MODEL_FAST, prompt_router, "application/json")
                
                if "Error" in resp.text:
                    status.update(label="API 连接错误", state="error")
                    st.stop()
                
                cleaned_data = clean_json_string(resp.text)
                if cleaned_data:
                    intent = str(cleaned_data.get('type', 'inquiry')).lower().strip()
                status.update(label=f"意图: {intent.upper()}", state="complete")

            # ================= 逻辑分流 =================
            
            # 2. 简单查询
            if 'analysis' not in intent and 'irrelevant' not in intent:
                with st.spinner("正在生成查询代码..."):
                    prompt_code = f"""
                    Role: Python Data Expert.
                    History: {history_str}
                    Query: "{user_query}"
                    Context: {context_info}
                    Rules: pd.merge if needed. Define all vars. No print/plot. Final result to `result`.
                    Output JSON: {{ "summary": {{ "intent": "数据查询", "logic": "..." }}, "code": "..." }}
                    """
                    resp_code = safe_generate(client, MODEL_SMART, prompt_code, "application/json")
                    plan = clean_json_string(resp_code.text)
                
                if plan:
                    with st.expander("> 查看思考过程 (THOUGHT PROCESS)", expanded=False):
                        st.markdown(f"""
                        <div class="thought-box">
                            <span class="thought-header">逻辑推演:</span>
                            {plan.get('summary', {}).get('logic', 'No logic provided')}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("**生成代码:**")
                        st.code(plan.get('code'), language='python')

                    render_protocol_card(plan.get('summary', {}))
                    try:
                        exec_ctx = {"df_sales": df_sales, "df_product": df_product}
                        res_raw = safe_exec_code(plan['code'], exec_ctx)
                        res_df = normalize_result(res_raw)
                        
                        if not safe_check_empty(res_df):
                            formatted_df = format_display_df(res_df)
                            st.dataframe(formatted_df, use_container_width=True)
                            st.session_state.messages.append({"role": "assistant", "type": "df", "content": formatted_df})
                        else:
                            st.warning("尝试模糊搜索...")
                            fallback_code = f"result = df_product[df_product.astype(str).apply(lambda x: x.str.contains('{user_query[:2]}', case=False, na=False)).any(axis=1)].head(10)"
                            try:
                                res_fallback = safe_exec_code(fallback_code, exec_ctx)
                                if not safe_check_empty(normalize_result(res_fallback)):
                                    st.dataframe(res_fallback)
                                    st.session_state.messages.append({"role": "assistant", "type": "df", "content": res_fallback})
                                else:
                                    st.markdown(f'<div class="custom-error">未找到相关数据</div>', unsafe_allow_html=True)
                            except: pass
                    except Exception as e:
                        st.markdown(f'<div class="custom-error">代码执行错误: {e}</div>', unsafe_allow_html=True)

            # 3. 深度分析
            elif 'analysis' in intent:
                shared_ctx = {"df_sales": df_sales.copy(), "df_product": df_product.copy()}

                with st.spinner("正在规划分析路径..."):
                    prompt_plan = f"""
                    Role: Senior Analyst.
                    History: {history_str}
                    Query: "{user_query}"
                    Context: {context_info}
                    Task: Create 2-3 analysis angles.
                    Output JSON: {{ "intent_analysis": "...", "angles": [ {{ "title": "...", "desc": "...", "code": "..." }} ] }}
                    """
                    resp_plan = safe_generate(client, MODEL_SMART, prompt_plan, "application/json")
                    plan_json = clean_json_string(resp_plan.text)
                
                if plan_json:
                    intro = f"**分析思路:**\n{plan_json.get('intent_analysis')}"
                    with st.expander("> 查看分析思路 (ANALYSIS THOUGHT)", expanded=True):
                        st.markdown(intro)
                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": intro})
                    
                    angles_data = []
                    
                    for angle in plan_json.get('angles', []):
                        with st.container():
                            st.markdown(f"**> {angle['title']}**")
                            try:
                                res_raw = safe_exec_code(angle['code'], shared_ctx)
                                res_df = normalize_result(res_raw)
                                
                                if not safe_check_empty(res_df):
                                    formatted_df = format_display_df(res_df)
                                    st.dataframe(formatted_df, use_container_width=True)
                                    st.session_state.messages.append({"role": "assistant", "type": "df", "content": formatted_df})
                                    
                                    prompt_mini = f"Interpret data (1 sentence) in Chinese (中文解释):\n{res_df.to_string()}"
                                    resp_mini = safe_generate(client, MODEL_FAST, prompt_mini)
                                    explanation = resp_mini.text
                                    st.markdown(f'<div class="mini-insight">>> {explanation}</div>', unsafe_allow_html=True)
                                    angles_data.append({"title": angle['title'], "explanation": explanation})
                                else:
                                    st.warning(f"{angle['title']} 暂无数据")
                            except Exception as e:
                                st.error(f"分析错误: {e}")

                    if angles_data:
                        # [新功能] 流式输出总结
                        st.markdown("### 分析总结")
                        
                        findings = "\n".join([f"[{a['title']}]: {a['explanation']}" for a in angles_data])
                        prompt_final = f"""Based on findings: {findings}, answer: "{user_query}". Response in Chinese (中文). Professional tone."""
                        
                        # 调用流式生成器并展示打字机效果
                        stream_gen = stream_generate(client, MODEL_SMART, prompt_final)
                        final_response = st.write_stream(stream_gen)
                        
                        # 记录完整回复
                        st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"### 分析总结\n{final_response}"})

                        # === 修复开始：Follow-up questions ===
                        # 1. 优化 Prompt：强制要求返回字符串列表，给出示例
                        prompt_next = f"""
                        Based on the analysis above, suggest 2 follow-up questions in Chinese. 
                        Output ONLY a JSON List of strings. 
                        Example format: ["第一个问题是什么?", "第二个问题是什么?"]
                        """
                        resp_next = safe_generate(client, MODEL_FAST, prompt_next, "application/json")
                        next_questions = clean_json_string(resp_next.text)

                        if isinstance(next_questions, list) and len(next_questions) > 0:
                            st.markdown("### 建议追问")
                            c1, c2 = st.columns(2)
                            
                            # 2. 增加容错逻辑：如果模型还是返回了字典，尝试提取文本
                            def get_q_text(q):
                                if isinstance(q, str): 
                                    return q
                                if isinstance(q, dict): 
                                    # 尝试提取可能的键
                                    return q.get('question_zh', q.get('question', list(q.values())[0]))
                                return str(q)

                            # 渲染按钮
                            if len(next_questions) > 0: 
                                q1_text = get_q_text(next_questions[0])
                                c1.button(f"> {q1_text}", use_container_width=True, on_click=handle_followup, args=(q1_text,))
                            
                            if len(next_questions) > 1: 
                                q2_text = get_q_text(next_questions[1])
                                c2.button(f"> {q2_text}", use_container_width=True, on_click=handle_followup, args=(q2_text,))
                        # === 修复结束 ===
            
            elif 'irrelevant' in intent:
                msg = "该问题似乎与医药数据无关，我是 ChatBI，专注于医药市场分析。"
                def simple_stream():
                    for word in msg:
                        yield word
                        time.sleep(0.02)
                st.write_stream(simple_stream)
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": msg})

    except Exception as e:
        import traceback
        st.markdown(f'<div class="custom-error">系统异常: {str(e)}</div>', unsafe_allow_html=True)
