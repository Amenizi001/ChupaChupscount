import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
from ultralytics import YOLO
import datetime

# ==========================================
# 1. ページ設定とモデルの読み込み
# ==========================================
st.set_page_config(page_title="チュッパチャプス周回カウンター", layout="centered")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"モデル(best.pt)の読み込みに失敗しました。同じフォルダに配置してください。\n{e}")
    st.stop()

# ==========================================
# 2. 画像撮影日時の取得関数
# ==========================================
def get_capture_time(image, is_camera):
    # カメラ撮影の場合は現在時刻を返す
    if is_camera:
        return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    
    # ファイルアップロードの場合はExifから元の撮影日時を取得
    try:
        exif = image._getexif()
        if exif is not None:
            for tag, value in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == 'DateTimeOriginal':
                    # Exifのフォーマットは 'YYYY:MM:DD HH:MM:SS' なので '/' に変換
                    return value.replace(':', '/', 2)
    except Exception:
        pass
    
    # Exif情報がない場合や取得に失敗した場合は、アップロードした現在時刻を返す
    return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

# ==========================================
# 3. メインの画像処理関数（透視変換＋YOLO）
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
# 4. アプリのUI表示
# ==========================================
st.title("🏃‍♂️ リレー周回カウンター")
st.markdown("有孔ボードを撮影して、現在の周回数をカウントします。")

# Streamlitの仕様上、コードから背面カメラの強制起動はできないため注記を添える
st.info("💡 カメラ起動時、内側カメラになった場合はUI右上のカメラ切替ボタンで**背面カメラ**に変更してください。")

camera_img = st.camera_input("カメラで撮影")
file_img = st.file_uploader("または画像をアップロード", type=['jpg', 'jpeg', 'png'])

image_source = camera_img if camera_img else file_img

if image_source is not None:
    # 画像の読み込みと撮影日時の取得
    image = Image.open(image_source)
    img_array = np.array(image)
    
    # どちらから入力されたかで日時の取得方法を分ける
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
        
        # --- チーム番号の確認と手動修正エリア ---
        st.markdown("### 📝 結果の確認と送信")
        st.markdown("AIが読み取ったチーム番号が表示されています。**間違っている場合は手入力で修正**してください。")
        
        default_team_val = detected_team if detected_team else ""
        final_team_number = st.text_input("チーム番号", value=default_team_val, placeholder="例: 005")
        
        st.write(f"📷 撮影/取得日時: `{capture_time}`")

        # 送信ボタン
        if st.button("この結果を本部に送信する", type="primary"):
            if not final_team_number.strip():
                st.error("⚠️ チーム番号を入力してください！")
            else:
                # フェーズ2用: ここにスプレッドシートへの送信処理を実装
                st.info(f"以下のデータを送信しました！（※現在はテストメッセージです）")
                st.code(f"チーム番号 : {final_team_number}\n周回数     : {count_or_error} 周\n撮影日時   : {capture_time}")
