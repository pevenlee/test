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
    FIXED_API_KEY = "" # 请确保这里有你的 API Key 或者通过 st.secrets 配置

# ================= 2. 视觉体系 (Noir UI - 侧边栏升级版) =================

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
            --border-color: #333333;
            --text-primary: #E0E0E0;
            --accent-error: #FF3333;
            --radius-md: 8px; /* 定义通用圆角变量 */
            --header-height: 60px; /* 统一定义顶导高度 */
        }

        /* 全局字体 */
        .stApp, .element-container, .stMarkdown, .stDataFrame, .stButton, div[data-testid="stDataEditor"] {
            font-family: "Microsoft YaHei", "SimHei", 'JetBrains Mono', monospace !important;
            background-color: var(--bg-color);
        }
        
        /* 全局圆角设置 */
        div, input, select, textarea { border-radius: var(--radius-md) !important; }
        
        /* 按钮样式 */
        .stButton button {
            border-radius: var(--radius-md) !important;
            text-align: left !important;
            justify-content: flex-start !important;
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

        /* === 布局核心修正 === */
        
        /* 1. 顶部导航栏 (最高层级) */
        .fixed-header-container {
            position: fixed; top: 0; left: 0; width: 100%; height: var(--header-height);
            background-color: rgba(5,5,5,0.95);
            border-bottom: 1px solid var(--border-color);
            z-index: 999999 !important; 
            display: flex; align-items: center; justify-content: space-between;
            
            /* --- CHANGE THIS LINE --- */
            /* Old: padding: 0 100px 0 24px; */
            padding: 0 24px; /* Changed right padding from 100px to 24px */
        }

        /* 2. 侧边栏容器 (下沉到顶导下方) */
        section[data-testid="stSidebar"] {
            top: var(--header-height) !important; /* 顶部距离 = 顶导高度 */
            height: calc(100vh - var(--header-height)) !important; /* 高度 = 屏幕 - 顶导 */
            z-index: 999998 !important; /* 层级略低于顶导 */
            background-color: #0A0A0A !important; 
            border-right: 1px solid #333;
            padding-top: 20px !important; 
            box-shadow: 2px 0 10px rgba(0,0,0,0.3);
        }
        
        /* 3. 展开按钮 (Collapsed Control) 位置修正 */
        [data-testid="stSidebarCollapsedControl"] {
            position: fixed !important;
            top: 75px !important; /* 60px(顶导) + 15px(间距) */
            left: 20px !important;
            z-index: 1000000 !important; /* 比顶导还要高，确保能点到 */
            background-color: transparent !important;
            color: #E0E0E0 !important;
            display: block !important; 
        }
        
        [data-testid="stSidebarCollapsedControl"] svg {
            fill: #E0E0E0 !important;
            color: #E0E0E0 !important;
        }

        /* 4. Streamlit 原生 Header (透明化并置顶) */
        header[data-testid="stHeader"] { 
            background: transparent !important; 
            z-index: 999999 !important; 
            height: var(--header-height) !important;
        }
        header[data-testid="stHeader"] > div:first-child {
            background: transparent !important;
        }
        
        /* === 5. [新增] 侧边栏数据字典样式 (Chips) === */
        .dict-category {
            font-size: 13px;
            font-weight: 700;
            color: #888;
            margin-top: 20px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .chip-container {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 10px;
        }
        
        .field-chip {
            display: inline-flex;
            align-items: center;
            background-color: #1A1A1A;
            border: 1px solid #333;
            border-radius: 6px; /* 圆角矩阵 */
            padding: 4px 8px;
            font-size: 11px;
            color: #CCC;
            font-family: 'JetBrains Mono', monospace;
            transition: all 0.2s;
        }
        .field-chip:hover {
            border-color: #555;
            color: #FFF;
            background-color: #222;
        }
        .field-chip.highlight {
            border-color: #444;
            background-color: #181818;
            color: #4CAF50; /* 绿色高亮 */
        }
        
        /* --- 其他样式 --- */
        
        .nav-left { display: flex; align-items: center; gap: 12px; }
        .nav-logo-img { height: 28px; width: auto; }
        .nav-logo-text { font-weight: 700; font-size: 18px; color: #FFF; letter-spacing: -0.5px; }
        
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
            border-radius: 50%;
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
            background: #080808;
            padding: 10px;
            margin-bottom: 10px;
            text-align: left !important;
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
        }
        
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
            background: #0F0F0F; padding: 12px; border: 1px solid #333; 
            margin-bottom: 15px; font-size: 12px; 
            text-align: left !important;
            border-radius: var(--radius-md); 
        }
        .protocol-row { display: flex; justify-content: flex-start; border-bottom: 1px solid #222; padding: 6px 0; }
        .protocol-row:last-child { border-bottom: none; }
        .protocol-key { color: #666; width: 80px; font-weight: bold; flex-shrink: 0; } 
        .protocol-val { color: #DDD; word-break: break-all; }
        
        /* 洞察框 */
        .insight-box { 
            background: #0A0A0A; padding: 15px; border-left: 3px solid #FFF; color: #DDD; margin-top: 10px; 
            text-align: left !important;
            border-radius: 0 var(--radius-md) var(--radius-md) 0; 
        }
        .mini-insight { color: #DDD; font-size: 12px; font-style: italic; border-top: 1px solid #222; margin-top: 8px; padding-top: 4px; }
        
        /* 错误提示 */
        .custom-error {
            background-color: rgba(40, 0, 0, 0.9); border: 1px solid var(--accent-error); color: #ffcccc;
            padding: 15px; font-size: 13px; margin-bottom: 1rem; display: flex; align-items: center; gap: 10px;
            border-radius: var(--radius-md);
        }
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
    info.append("| 字段 | 类型 | 范围/示例 |")
    info.append("|---|---|---|")
    for col in df.columns:
        dtype = str(df[col].dtype)
        if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in str(col).lower() or "日期" in str(col):
            try:
                temp_col = pd.to_datetime(df[col], errors='coerce')
                min_date = temp_col.min()
                max_date = temp_col.max()
                if pd.notnull(min_date) and pd.notnull(max_date):
                    sample = f"{min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}"
                else:
                    sample = list(df[col].dropna().unique()[:3])
            except:
                sample = list(df[col].dropna().unique()[:3])
        else:
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
        match_list = re.search(r'\[.*\]', text, re.DOTALL)
        if match_list:
             try: return json.loads(match_list.group(0))
             except: pass
    return None

# --- [新增] API 重试逻辑核心 ---
def safe_generate(client, model, prompt, mime_type="text/plain", max_retries=3):
    """
    带重试机制的 API 调用
    """
    config = types.GenerateContentConfig(response_mime_type=mime_type)
    
    retry_count = 0
    base_delay = 2 # 初始等待 2 秒
    
    while retry_count <= max_retries:
        try:
            return client.models.generate_content(model=model, contents=prompt, config=config)
        except Exception as e:
            error_str = str(e)
            # 检查是否为 429 (Resource exhausted) 或 503 (Server unavailable)
            if "429" in error_str or "429" in str(getattr(e, 'code', '')) or "Resource exhausted" in error_str:
                if retry_count == max_retries:
                    # 超过最大重试次数，返回错误
                    return type('obj', (object,), {'text': f"Error (Max Retries): {e}"})
                
                wait_time = base_delay * (2 ** retry_count) # 指数退避: 2s, 4s, 8s
                st.toast(f"⏳ API 请求过于频繁，正在重试 ({retry_count + 1}/{max_retries})...等待 {wait_time}秒", icon="⚠️")
                time.sleep(wait_time)
                retry_count += 1
            else:
                # 如果是其他错误 (如 400 Bad Request)，直接返回，不重试
                return type('obj', (object,), {'text': f"Error: {e}"})

def stream_generate(client, model, prompt, max_retries=3):
    """
    带重试机制的流式生成
    注意：流式生成如果中途中断，通常需要重新开始整个请求
    """
    config = types.GenerateContentConfig(response_mime_type="text/plain")
    
    retry_count = 0
    base_delay = 2
    
    while retry_count <= max_retries:
        try:
            response = client.models.generate_content_stream(
                model=model, 
                contents=prompt, 
                config=config
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            return # 成功生成完则退出函数
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "429" in str(getattr(e, 'code', '')) or "Resource exhausted" in error_str:
                if retry_count == max_retries:
                    yield f"Stream Error (Max Retries): {e}"
                    return
                
                wait_time = base_delay * (2 ** retry_count)
                st.toast(f"⏳ 流式生成连接繁忙，正在重试 ({retry_count + 1}/{max_retries})...", icon="⚠️")
                time.sleep(wait_time)
                retry_count += 1
            else:
                yield f"Stream Error: {e}"
                return

def simulated_stream(text, speed=0.01):
    for word in text:
        yield word
        time.sleep(speed)

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
    intent = summary.get('intent', '-')
    scope = summary.get('scope', '-')
    metrics = summary.get('metrics', '-')
    logic = summary.get('logic', '-')
    
    st.markdown(f"""
    <div class="protocol-box">
        <div class="protocol-row"><span class="protocol-key">意图识别</span><span class="protocol-val">{intent}</span></div>
        <div class="protocol-row"><span class="protocol-key">数据范围</span><span class="protocol-val">{scope}</span></div>
        <div class="protocol-row"><span class="protocol-key">计算指标</span><span class="protocol-val">{metrics}</span></div>
        <div class="protocol-row"><span class="protocol-key">计算逻辑</span><span class="protocol-val">{logic}</span></div>
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
    if role == "user":
        return USER_AVATAR if os.path.exists(USER_AVATAR) else None
    else:
        return BOT_AVATAR if os.path.exists(BOT_AVATAR) else None

# ================= 4. 页面渲染 =================

inject_custom_css()
client = get_client()

df_sales = load_local_data(FILE_FACT)
df_product = load_local_data(FILE_DIM)

# --- [重构] Sidebar: 数据字典 & 范围 ---
with st.sidebar:
    st.markdown("### ☷ 可用数据字段范围")
    
    # 获取所有可用列名
    all_cols = set()
    if df_sales is not None: all_cols.update(df_sales.columns)
    if df_product is not None: all_cols.update(df_product.columns)
    
    def render_chips(label, items, is_highlight=False):
        """渲染分类和圆角矩阵标签"""
        st.markdown(f"<div class='dict-category'>{label}</div>", unsafe_allow_html=True)
        html = "<div class='chip-container'>"
        has_item = False
        for item in items:
            # 简单去重和清理
            if item in all_cols or label in ["⚙︎ 渠道范围", "⏱︎ 数据时间"]: 
                extra_class = "highlight" if is_highlight else ""
                html += f"<div class='field-chip {extra_class}'>{item}</div>"
                has_item = True
        html += "</div>"
        if has_item:
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='font-size:11px; color:#555;'>暂无字段</span>", unsafe_allow_html=True)

    # ================= [修改] 1. 时间范围 (已移至最前) =================
    time_range_str = "未加载"
    if df_sales is not None:
        # 尝试寻找时间列
        time_col = None
        for c in df_sales.columns:
            if "年季" in c or "date" in c.lower() or "time" in c.lower():
                time_col = c
                break
        
        if time_col:
            try:
                # 假设是 YearQuarter 格式 (e.g. 20211 or 2021Q1)
                min_val = df_sales[time_col].min()
                max_val = df_sales[time_col].max()
                
                def fmt_q(val):
                    s = str(val)
                    if "Q" in s: return s
                    if len(s) == 5: return f"{s[:4]}Q{s[-1]}" # 20211 -> 2021Q1
                    return s
                
                time_range_str = f"{fmt_q(min_val)} ~ {fmt_q(max_val)}"
            except:
                time_range_str = "格式解析失败"
    
    render_chips("⏱︎ 数据时间", [time_range_str], is_highlight=True)

    # ================= 2. 产品信息 =================
    product_fields = [
        "通用名", "商品名", "药品名称", "成分名", "生产企业", "集团名称", 
        "规格", "剂型", "ATC1Des", "ATC2Des", "ATC3Des", "ATC4Des",
        "药品分类", "药品分类二", "OTC", "零售分类1 描述", "零售分类2 描述", "零售分类3 描述",
        "研究类型", "企业类型"
    ]
    render_chips("🛒 产品信息", product_fields)

    # ================= 3. 政策标签 =================
    policy_fields = ["医保", "最早医保纳入年份", "集采批次", "集采结果", "一致性评价", "首次上市年代"]
    render_chips("◆ 政策标签", policy_fields)

    # ================= 4. 指标类型 =================
    metric_fields = ["销售额", "销售量"]
    render_chips("〽︎ 指标类型", metric_fields)

    # ================= 5. 渠道 =================
    # 尝试从数据中获取渠道值，如果不行则显示字段名
    channel_items = []
    if df_sales is not None and "渠道" in df_sales.columns:
        try:
            unique_channels = df_sales["渠道"].dropna().unique().tolist()
            if len(unique_channels) < 10: # 如果渠道数量少，显示具体值
                channel_items = unique_channels
            else:
                channel_items = ["渠道"]
        except:
            channel_items = ["渠道"]
    else:
        channel_items = ["渠道"]
    
    render_chips("⚙︎ 渠道范围", channel_items)


    st.markdown("---")
    st.markdown(f"<div style='font-size:10px; color:#666; text-align:center;'>Powered by {MODEL_SMART}</div>", unsafe_allow_html=True)
    
# --- Top Nav ---
logo_b64 = get_base64_image(LOGO_FILE)
if logo_b64:
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="nav-logo-img">'
else:
    logo_html = """<svg width="24" height="24" viewBox="0 0 24 24" fill="white"><path d="M12 2L2 22h20L12 2zm0 3.5L19 20H5l7-14.5z"/></svg>"""

user_avatar_b64 = get_base64_image(USER_AVATAR)
if user_avatar_b64:
    user_avatar_html = f'<div class="user-avatar-circle"><img src="data:image/png;base64,{user_avatar_b64}"></div>'
else:
    user_avatar_html = '<div class="user-avatar-circle" style="color:#FFF; font-size:10px;">User</div>'

# 调整后的 HTML 结构
st.markdown(f"""
<div class="fixed-header-container">
    <div class="nav-left">
        {logo_html}
        <div class="nav-logo-text">ChatBI</div>
    </div>
    <div class="nav-right">
        <div class="nav-tag">Peiwen</div>
        {user_avatar_html}
    </div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []

# --- Chat History ---
for msg in st.session_state.messages:
    avatar_file = get_avatar(msg["role"])
    with st.chat_message(msg["role"], avatar=avatar_file):
        if msg["type"] == "text": 
            role_class = "p-ai" if msg["role"] == "assistant" else "p-user"
            prefix = "Doc. > " if msg["role"] == "assistant" else "You > "
            st.markdown(f"<span class='msg-prefix {role_class}'>{prefix}</span>{msg['content']}", unsafe_allow_html=True)
        elif msg["type"] == "df": 
            st.dataframe(msg["content"], use_container_width=True)
        elif msg["type"] == "error":
            st.markdown(f'<div class="custom-error">{msg["content"]}</div>', unsafe_allow_html=True)

# --- 猜你想问 (左对齐按钮) ---
if not st.session_state.messages:
    st.markdown("### 我们正在通过人工智能重塑医药数据，点亮医药行业，有什么要问我们？")
    st.markdown("###  ")
    c1, c2, c3 = st.columns(3)
    def handle_preset(question):
        st.session_state.messages.append({"role": "user", "type": "text", "content": question})
        st.rerun()
    if c1.button("☑︎ 第十一批集采对中国医药市场院内外产生了什么样的影响？"): handle_preset("第十一批集采对中国医药市场院内外产生了什么样的影响？")
    if c2.button("☑︎ K药、O药、拓益、艾瑞卡、达伯舒、百泽安最近2年的销售额、份额、份额变化"): handle_preset("K药、O药、拓益、艾瑞卡、达伯舒、百泽安最近2年的销售额、份额、份额变化")
    if c3.button("☑︎ 销售额过亿的，独家处方药有哪些，总结一下他们的画像"): handle_preset("销售额过亿的，独家处方药有哪些，总结一下他们的画像")

# --- Input ---
query = st.chat_input("了解中国医药市场，从这里开始...")
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
                err_text = f"数据源缺失。请检查路径配置。 (需要文件: {FILE_FACT}, {FILE_DIM})"
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
                # [中文提示词] 意图路由
                prompt_router = f"""
                请根据以下上下文判断用户的意图。
                
                历史记录: {history_str}
                当前提问: "{user_query}"
                
                规则:
                1. 询问具体数值/数据/报表 -> "inquiry"
                2. 询问趋势/原因/细分市场分析 -> "analysis"
                3. 与医药数据无关 -> "irrelevant"
                
                严格输出 JSON: {{ "type": "result_value" }} (必须是 "inquiry", "analysis", "irrelevant" 之一)
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
                with st.spinner("正在生成查询代码，这个过程可能需要1~2分钟，请耐心等待…"):
                    # [中文提示词] 简单查询 & 四要素提取
                    prompt_code = f"""
                    你是一位医药行业的 Python 专家。
                    
                    【历史对话】(用于理解指代)
                    {history_str}
                    
                    【当前用户问题】
                    "{user_query}"
                    
                    【数据上下文】 {context_info}
                    
                    【指令】 
                    1. 严格按用户要求提取字段。
                    2. 使用 `pd.merge` 关联两表 (除非用户只查单表)。
                    3. **重要**: 确保所有使用的变量（如 market_share）都在代码中明确定义。不要使用未定义的变量。
                    4. **绝对禁止**导入 IPython 或使用 display() 函数。
                    5. 禁止使用 df.columns = [...] 强行改名，请使用 df.rename()。
                    6. **避免 'ambiguous' 错误**：如果 index name 与 column name 冲突，请在 reset_index() 前先使用 `df.index.name = None` 或重命名索引。
                    7. 结果必须赋值给变量 `result`。
                    8. **份额计算强制规则**: 
                    - 计算市场份额时，结果**必须乘以 100**，转换为百分数格式 (Percentage)。
                    - 例如：销售额/总额 = 0.1234，应存储为 12.34，而不是 0.1234。
                    - 列名必须包含 "(%)" 以提示用户，例如 "2024份额(%)"。
                    9. **数据类型与精度**:
                    - 份额列、变化率列：必须强制转换为 `float` 类型，保留 1 位小数 (`round(1)`)。
                    - 销售额列：必须转换为 `int` 类型 (无小数)。
                    - **严禁**对份额列使用 `astype(int)`，否则小于 1% 的份额会变成 0
                    10. **市场份额** 当提到计算份额时，优先定义分母是用户提到的所有产品总 > 对应细分领域下所有产品总 > 全产品总和；然后再做计算
                    11. **代码安全 - 严禁 inplace=True 后赋值**: 
                        - 错误写法: `df = df.rename(..., inplace=True)` (这会导致 df 变成 None)
                        - 正确写法: `df = df.rename(...)` 或 `df.rename(..., inplace=True)` (不赋值)
                    
                    
                    【关键指令】
                    1. **数据范围检查**: 查看上下文中的日期范围。最新的日期决定了“当前周期”。
                    2. **同口径对比 (Like-for-Like)**: 当分析跨年增长或趋势时，**必须**筛选前一年的数据以匹配当前年份的月份/季度范围 (YTD逻辑)。
                        - 例如: 如果最大日期是 2025-09-30，那么“2024年数据”用于对比时，只能取 2024-01-01 到 2024-09-30，而不是2024全年的数据。
                    3. 返回时间范围时，需要说明用的原始表中的哪个时间段 如问最近两年的同比，如果为了对齐数据，则返回格式为 2024Q1~Q3 & 2025Q1~Q3
                    
                    【摘要生成规则 (Summary)】
                    - scope (范围): 数据的筛选范围。
                    - metrics (指标): 用户查询的核心指标。
                    - key_match (关键匹配): **必须说明**提取了用户什么词，去匹配了哪个列。例如："提取用户词 'K药' -> 模糊匹配 '商品名' 列"。
                    - logic (加工逻辑): 简述筛选和计算步骤，严禁提及“表关联”、“Merge”等技术术语。
                    
                    输出 JSON: {{ "summary": {{ "intent": "简单取数", "scope": "...", "metrics": "...", "key_match": "...", "logic": "..." }}, "code": "..." }}
                    """
                    resp_code = safe_generate(client, MODEL_SMART, prompt_code, "application/json")
                    plan = clean_json_string(resp_code.text)
                
                if plan:
                    # [新功能] 打印数据调用逻辑
                    summary_obj = plan.get('summary', {})
                    logic_text = summary_obj.get('logic', '暂无逻辑描述')
                    
                    with st.expander("> 查看思考过程 (THOUGHT PROCESS)", expanded=True): 
                        # 使用 placeholder + simulated_stream 实现带样式的打字机效果
                        logic_placeholder = st.empty()
                        streamed_text = ""
                        # 模拟流式输出
                        for chunk in simulated_stream(logic_text):
                            streamed_text += chunk
                            logic_placeholder.markdown(f"""
                            <div class="thought-box">
                                <span class="thought-header">逻辑推演:</span>
                                {streamed_text}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("**生成代码:**")
                        st.code(plan.get('code'), language='python')

                    # 渲染四要素卡片
                    render_protocol_card(summary_obj)
                    
                    try:
                        # 【修正】使用 copy() 确保数据隔离
                        exec_ctx = {"df_sales": df_sales.copy(), "df_product": df_product.copy()}
                        res_raw = safe_exec_code(plan['code'], exec_ctx)
                        res_df = normalize_result(res_raw)
                        
                        if not safe_check_empty(res_df):
                            formatted_df = format_display_df(res_df)
                            st.dataframe(formatted_df, use_container_width=True)
                            st.session_state.messages.append({"role": "assistant", "type": "df", "content": formatted_df})

                            # ==========================================
                            # [新增功能 START] 1. Flash 快速总结表格
                            # ==========================================
                            try:
                                prompt_summary = f"请用精炼的中文一句话总结以下数据的主要发现 (不使用Markdown格式):\n{formatted_df.to_string()}"
                                resp_summary = safe_generate(client, MODEL_FAST, prompt_summary)
                                summary_text = resp_summary.text.strip()
                                
                                # 显示并保存总结
                                st.markdown(f'<div class="mini-insight">>> {summary_text}</div>', unsafe_allow_html=True)
                                st.session_state.messages.append({"role": "assistant", "type": "text", "content": summary_text})
                            except Exception as e:
                                pass # 总结失败不影响数据展示

                            # ==========================================
                            # [新增功能 START] 2. Smart 模型生成追问
                            # ==========================================
                            try:
                                # 1. 获取所有可用字段名
                                all_columns = []
                                if df_sales is not None: all_columns.extend(df_sales.columns.tolist())
                                if df_product is not None: all_columns.extend(df_product.columns.tolist())
                                # 去重并转为字符串
                                cols_str = ", ".join(list(set(all_columns)))

                                # 2. 构建包含字段信息的提示词
                                prompt_next = f"""
                                基于生成的表格数据和洞察。
                                
                                【数据库完整可用字段列表】:
                                {cols_str}
                                
                                【指令】
                                针对用户的问题 "{user_query}"，从上面的“可用字段列表”中寻找灵感，
                                给出客户最可能想深入挖掘的 2 个问题（例如：按[某个具体字段]拆分、看[某个字段]的趋势等）。
                                
                                严格输出 JSON 字符串列表。
                                示例格式: ["查看该产品的分医院排名", "分析不同剂型的份额变化"]
                                """
                                resp_next = safe_generate(client, MODEL_SMART, prompt_next, "application/json")
                                next_questions = clean_json_string(resp_next.text)

                                if isinstance(next_questions, list) and len(next_questions) > 0:
                                    st.markdown("### 是否追问")
                                    c1, c2 = st.columns(2)
                                    
                                    def get_q_text_safe(q):
                                        if isinstance(q, str): return q
                                        if isinstance(q, dict): return q.get('question', list(q.values())[0])
                                        return str(q)

                                    if len(next_questions) > 0: 
                                        q1_text = get_q_text_safe(next_questions[0])
                                        c1.button(f"> {q1_text}", use_container_width=True, on_click=handle_followup, args=(q1_text,))
                                    if len(next_questions) > 1: 
                                        q2_text = get_q_text_safe(next_questions[1])
                                        c2.button(f"> {q2_text}", use_container_width=True, on_click=handle_followup, args=(q2_text,))
                            except Exception as e:
                                pass # 追问生成失败不报错
                            
                            # ==========================================
                            # [新增功能 END]
                            # ==========================================

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
                
                with st.spinner("正在规划分析路径，这个过程可能需要1~2分钟，请耐心等待..."):
                    # [中文提示词] 深度分析 & 四要素提取
                    prompt_plan = f"""
                    角色: 资深医药数据分析师。
                    历史记录: {history_str}
                    当前提问: "{user_query}"
                    数据上下文: {context_info}
                    
                    关键指令:
                    1. **数据范围检查**: 查看上下文中的日期范围。最新的日期决定了“当前周期”。
                    2. **同口径对比 (Like-for-Like)**: 当分析跨年增长或趋势时，**必须**筛选前一年的数据以匹配当前年份的月份/季度范围 (YTD逻辑)。
                        - 例如: 如果最大日期是 2025-09-30，那么“2024年数据”用于对比时，只能取 2024-01-01 到 2024-09-30，而不是2024全年的数据。
                    3. **代码安全**: 绝对禁止 `df = df.func(inplace=True)` 这种写法，这会导致 DataFrame 变成 NoneType 引发合并错误。
                    4. **语言**: 所有的 "title" (标题), "desc" (描述), 和 "intent_analysis" (分析思路) 必须使用**简体中文**。
                    5. **完整性**: 提供 2-5 个不同的分析维度。
                    6. **变量定义检查 (CRITICAL)**: 
                        - **严禁引用未定义的变量**（例如代码中出现了 `df_excl` 或 `df_temp` 但前面没有定义它）。
                        - 凡是使用的变量，必须在当前代码块中显式赋值。
                    
                    严格输出 JSON (不要Markdown, 不要代码块): 
                    {{ 
                        "summary": {{ 
                             "intent": "深度市场分析", 
                             "scope": "产品: [产品名], 时间: [YYYY-MM ~ YYYY-MM]",
                             "metrics": "趋势 / 结构 / 增长驱动力...", 
                             "logic": "1. 总体趋势分析; 2. 产品结构拆解; 3. 增长贡献度计算..." 
                        }},
                        "intent_analysis": "这里用中文详细描述你的分析思路，特别是说明你如何处理了时间周期对齐（例如：'鉴于数据截止至2025Q3，我将提取2024同期数据进行同比分析...'）", 
                        "angles": [ 
                            {{ "title": "中文标题", "desc": "中文描述", "code": "Python code storing result in `result` variable..." }} 
                        ] 
                    }}
                    """
                    resp_plan = safe_generate(client, MODEL_SMART, prompt_plan, "application/json")
                    plan_json = clean_json_string(resp_plan.text)
                
                if not plan_json:
                    st.error("分析规划生成失败，模型未返回有效格式。")
                    st.stop()

                if plan_json:
                    # [新功能] 打印分析思路
                    intro_text = plan_json.get('intent_analysis', '分析思路生成中...')
                    intro = f"**分析思路:**\n{intro_text}"
                    
                    with st.expander("> 查看分析思路 (ANALYSIS THOUGHT)", expanded=True): 
                         st.write_stream(simulated_stream(intro))
                    
                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": intro})
                    
                    # 渲染四要素卡片 (如果有的话)
                    if 'summary' in plan_json:
                        render_protocol_card(plan_json['summary'])

                    angles_data = []
                    
                    for angle in plan_json.get('angles', []):
                        with st.container():
                            st.markdown(f"**> {angle['title']}**")
                            
                            # 【修正】循环内部创建独立 Context，防止污染
                            local_ctx = {
                                "df_sales": df_sales.copy(), 
                                "df_product": df_product.copy(),
                                "pd": pd,
                                "np": np
                            }
                            
                            try:
                                res_raw = safe_exec_code(angle['code'], local_ctx)
                                if isinstance(res_raw, dict) and any(isinstance(v, (pd.DataFrame, pd.Series)) for v in res_raw.values()):
                                    res_df = pd.DataFrame() 
                                    for k, v in res_raw.items():
                                        st.markdown(f"**- {k}**")
                                        sub_df = normalize_result(v)
                                        st.dataframe(format_display_df(sub_df), use_container_width=True)
                                        res_df = sub_df 
                                        st.session_state.messages.append({"role": "assistant", "type": "df", "content": sub_df})
                                else:
                                    res_df = normalize_result(res_raw)
                                    if not safe_check_empty(res_df):
                                        formatted_df = format_display_df(res_df)
                                        st.dataframe(formatted_df, use_container_width=True)
                                        st.session_state.messages.append({"role": "assistant", "type": "df", "content": formatted_df})
                                        
                                        # [中文提示词] 数据解读
                                        prompt_mini = f"用一句话解读以下数据 (中文): \n{res_df.to_string()}"
                                        resp_mini = safe_generate(client, MODEL_FAST, prompt_mini)
                                        explanation = resp_mini.text
                                        st.markdown(f'<div class="mini-insight">>> {explanation}</div>', unsafe_allow_html=True)
                                        angles_data.append({"title": angle['title'], "explanation": explanation})
                                    else:
                                        st.warning(f"{angle['title']} 暂无数据")
                            except Exception as e:
                                st.error(f"分析错误: {e}")

                    if angles_data:
                        st.markdown("### 分析总结")
                        findings = "\n".join([f"[{a['title']}]: {a['explanation']}" for a in angles_data])
                        # [中文提示词] 最终总结
                        prompt_final = f"""基于以下发现: {findings}，回答问题: "{user_query}"。请使用专业、客观的中文口吻。"""
                        
                        stream_gen = stream_generate(client, MODEL_SMART, prompt_final)
                        final_response = st.write_stream(stream_gen)
                        st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"### 分析总结\n{final_response}"})

                        # === Follow-up questions ===
                        # [中文提示词] 追问生成
                        
                        # 1. 获取字段上下文
                        all_columns = []
                        if df_sales is not None: all_columns.extend(df_sales.columns.tolist())
                        if df_product is not None: all_columns.extend(df_product.columns.tolist())
                        cols_str = ", ".join(list(set(all_columns)))

                        prompt_next = f"""
                        基于生成的表格和洞察。
                        
                        【数据库完整可用字段列表】:
                        {cols_str}
                        
                        【指令】
                        结合“可用字段列表”，生成 2 个具有深度的后续分析问题。
                        仅输出一个 JSON 字符串列表。
                        示例格式: ["分析各省份的市场表现差异", "查看Top5企业的竞争格局"]
                        """
                        resp_next = safe_generate(client, MODEL_SMART, prompt_next, "application/json")
                        next_questions = clean_json_string(resp_next.text)

                        if isinstance(next_questions, list) and len(next_questions) > 0:
                            st.markdown("### 是否追问")
                            c1, c2 = st.columns(2)
                            
                            def get_q_text(q):
                                if isinstance(q, str): return q
                                if isinstance(q, dict): return q.get('question_zh', q.get('question', list(q.values())[0]))
                                return str(q)

                            if len(next_questions) > 0: 
                                q1_text = get_q_text(next_questions[0])
                                c1.button(f"> {q1_text}", use_container_width=True, on_click=handle_followup, args=(q1_text,))
                            if len(next_questions) > 1: 
                                q2_text = get_q_text(next_questions[1])
                                c2.button(f"> {q2_text}", use_container_width=True, on_click=handle_followup, args=(q2_text,))
            
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
