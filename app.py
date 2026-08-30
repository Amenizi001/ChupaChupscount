import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
from ultralytics import YOLO
import datetime
import io

# === 新たに追加する Google API 用ライブラリ ===
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# ==========================================
# 0. アクセス制限（パスワード認証）
# ==========================================
EVENT_PASSWORD = st.secrets["app_password"]

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 運営スタッフ用ログイン")
    st.markdown("このアプリはスタッフ専用です。合言葉を入力してください。")
    
    pwd_input = st.text_input("合言葉", type="password")
    if st.button("ログイン"):
        if pwd_input == EVENT_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ 合言葉が間違っています")
            
    st.stop() # 認証失敗時はここでストップ

# ==========================================
# 1. Google API 設定 & キャッシュ
# ==========================================
# ★ここを書き換えてください★
# Streamlit CloudのSecretsから安全に読み込む
SPREADSHEET_ID = st.secrets["SPREADSHEET_ID"]
DRIVE_FOLDER_ID = st.secrets["DRIVE_FOLDER_ID"]

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

@st.cache_resource
def get_google_clients():
    try:
        # StreamlitのSecretsからJSON情報を読み込む
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        
        # スプレッドシート用クライアント
        gc = gspread.authorize(creds)
        # ドライブ用クライアント
        drive_service = build('drive', 'v3', credentials=creds)
        
        return gc, drive_service
    except Exception as e:
        st.error(f"Google APIの認証に失敗しました。Secretsの設定を確認してください。\n{e}")
        st.stop()

gc, drive_service = get_google_clients()

# ==========================================
# 2. ページ設定とモデルの読み込み
# ==========================================
st.set_page_config(page_title="チュッパチャプス周回カウンター", layout="centered")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"モデル(best.pt)の読み込みに失敗しました。\n{e}")
    st.stop()

# ==========================================
# 3. 日時取得 ＆ Drive/Sheets送信関数
# ==========================================
def get_capture_time(image, is_camera):
    if is_camera:
        return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    try:
        exif = image._getexif()
        if exif is not None:
            for tag, value in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == 'DateTimeOriginal':
                    return value.replace(':', '/', 2)
    except Exception:
        pass
    return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

def upload_to_drive_and_sheets(image_array, team, capture_time, count):
    # --- 1. Google Driveへ画像アップロード ---
    safe_time = capture_time.replace("/", "").replace(":", "").replace(" ", "_")
    img_filename = f"{team}_{safe_time}_{count}.jpg"
    
    img = Image.fromarray(image_array)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    file_metadata = {
        'name': img_filename,
        'parents': [DRIVE_FOLDER_ID]
    }
    media = MediaIoBaseUpload(img_byte_arr, mimetype='image/jpeg', resumable=True)
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink', supportsAllDrives=True).execute()
    img_url = file.get('webViewLink')

    # --- 2. スプレッドシートの更新 ---
    sheet = gc.open_by_key(SPREADSHEET_ID)
    system_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    row_data = [team, count, capture_time, img_url, system_time]
    
    # Historyシートへ追記
    history_ws = sheet.worksheet("History")
    history_ws.append_row(row_data)
    
    # Summaryシートの更新
    summary_ws = sheet.worksheet("Summary")
    records = summary_ws.get_all_records()
    
    # 既存のチームがあるか探す
    row_idx = None
    for i, row in enumerate(records):
        if str(row.get("チーム番号", "")) == str(team):
            row_idx = i + 2 # ヘッダーが1行目なので+2
            break
            
    if row_idx:
        # 見つかった場合は上書き更新
        summary_ws.update(f"A{row_idx}:E{row_idx}", [row_data])
    else:
        # 見つからなかった場合は新規追記
        summary_ws.append_row(row_data)

# ==========================================
# 4. メインの画像処理関数（透視変換＋YOLO）
# ==========================================
def process_image(img_array):
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    h, w = img.shape[:2]
    if max(h, w) > 2400:
        resize_scale = 2400 / max(h, w)
        img = cv2.resize(img, (int(w * resize_scale), int(h * resize_scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is None:
        return None, "❌ マーカーが検出されませんでした。明るい場所で、四隅が写るように撮影してください。", None

    marker_w_mm, marker_h_mm = 185, 315
    scale, padding_mm = 2, 50
    dst_w = int((marker_w_mm + padding_mm * 2) * scale)
    dst_h = int((marker_h_mm + padding_mm * 2) * scale)
    offset_px = int(padding_mm * scale)

    dst_pts = np.array([
        [offset_px, offset_px],
        [offset_px + marker_w_mm * scale, offset_px],
        [offset_px + marker_w_mm * scale, offset_px + marker_h_mm * scale],
        [offset_px, offset_px + marker_h_mm * scale]
    ], dtype="float32")
    
    req_ids = [1, 2, 3, 0]
    marker_dict = {}
    team_ids = [] 

    for id_val, c in zip(ids.flatten(), corners):
        id_val = int(id_val)
        if id_val in req_ids:
            cx, cy = float(np.mean(c[0, :, 0])), float(np.mean(c[0, :, 1]))
            marker_dict[id_val] = (cx, cy)
        else:
            team_ids.append(id_val)
            
    team_ids = sorted(list(set(team_ids)))
    if team_ids:
        detected_team_str = "_".join([f"{tid:03d}" for tid in team_ids])
    else:
        detected_team_str = None

    present_ids = [k for k in req_ids if k in marker_dict]
    if len(present_ids) == 3:
        missing = (set(req_ids) - set(present_ids)).pop()
        if missing == 2: marker_dict[2] = (marker_dict[1][0] + marker_dict[3][0] - marker_dict[0][0], marker_dict[1][1] + marker_dict[3][1] - marker_dict[0][1])
        elif missing == 3: marker_dict[3] = (marker_dict[2][0] + marker_dict[0][0] - marker_dict[1][0], marker_dict[2][1] + marker_dict[0][1] - marker_dict[1][1])
        elif missing == 1: marker_dict[1] = (marker_dict[0][0] + marker_dict[2][0] - marker_dict[3][0], marker_dict[0][1] + marker_dict[2][1] - marker_dict[3][1])
        elif missing == 0: marker_dict[0] = (marker_dict[1][0] + marker_dict[3][0] - marker_dict[2][0], marker_dict[1][1] + marker_dict[3][1] - marker_dict[2][1])
                              
    if not all(k in marker_dict for k in req_ids):
        return None, "❌ マーカーが3つ以上確認できませんでした。遮蔽物がないか確認してください。", detected_team_str
        
    src_pts = np.array([marker_dict[1], marker_dict[2], marker_dict[3], marker_dict[0]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (dst_w, dst_h))
    
    results = model.predict(source=warped, conf=0.70, iou=0.45, verbose=False)
    vis_img = warped.copy()
    
    valid_x_min, valid_x_max = offset_px, offset_px + marker_w_mm * scale
    valid_y_min, valid_y_max = offset_px, offset_px + marker_h_mm * scale
    cv2.rectangle(vis_img, (valid_x_min, valid_y_min), (valid_x_max, valid_y_max), (255, 0, 0), 2)
    
    valid_count = 0
    for r in results[0].boxes:
        box = r.xyxy[0].cpu().numpy().astype(int)
        conf = float(r.conf[0].cpu().numpy())
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if valid_x_min <= cx <= valid_x_max and valid_y_min <= cy <= valid_y_max:
            color = (0, 255, 0)
            valid_count += 1
        else:
            color = (0, 165, 255)
            
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_img, f"{conf:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    return vis_img_rgb, valid_count, detected_team_str

# ==========================================
# 5. アプリのUI表示
# ==========================================
st.title("🏃‍♂️ リレー周回カウンター")
st.markdown("有孔ボードを撮影して、現在の周回数をカウントします。")
st.info("💡 カメラ起動時、内側カメラになった場合はUI右上のカメラ切替ボタンで**背面カメラ**に変更してください。")

camera_img = st.camera_input("カメラで撮影")
file_img = st.file_uploader("または画像をアップロード", type=['jpg', 'jpeg', 'png'])

image_source = camera_img if camera_img else file_img

if image_source is not None:
    image = Image.open(image_source)
    img_array = np.array(image)
    
    is_camera = (camera_img is not None)
    capture_time = get_capture_time(image, is_camera)
    
    with st.spinner("AIが周回数とチーム番号を判定中..."):
        result_img, count_or_error, detected_team = process_image(img_array)
        
    if result_img is None:
        st.warning(count_or_error)
        if detected_team:
             st.info(f"💡 (参考) チーム番号マーカー [{detected_team}] は見えています。四隅のマーカーをすべて枠内に収めてください。")
    else:
        st.success("✅ 判定完了！")
        st.metric(label="カウントされた周回数（アメの数）", value=f"{count_or_error} 周")
        st.image(result_img, caption="AI判定結果", use_container_width=True)
        
        st.markdown("### 📝 結果の確認と送信")
        st.markdown("AIが読み取ったチーム番号が表示されています。**間違っている場合は手入力で修正**してください。")
        
        default_team_val = detected_team if detected_team else ""
        final_team_number = st.text_input("チーム番号", value=default_team_val, placeholder="例: 005")
        st.write(f"📷 撮影日時: `{capture_time}`")

        if st.button("この結果を本部に送信する", type="primary"):
            if not final_team_number.strip():
                st.error("⚠️ チーム番号を入力してください！")
            else:
                with st.spinner("Googleクラウドへ安全に送信中..."):
                    try:
                        upload_to_drive_and_sheets(result_img, final_team_number, capture_time, count_or_error)
                        st.success("✅ 本部へのデータ送信が完了しました！")
                        st.code(f"【送信内容】\nチーム番号 : {final_team_number}\n周回数     : {count_or_error} 周\n撮影日時   : {capture_time}")
                    except Exception as e:
                        st.error(f"送信中にエラーが発生しました: {e}")
