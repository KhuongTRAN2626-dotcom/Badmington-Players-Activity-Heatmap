import cv2 as cv
import numpy as np

VIDEO_PATH   = 'caulong.mp4'
COURT_IMG    = 'court.png'
M_PATH       = 'M.npy'
OUTPUT_PATH  = 'heatmap_result.png'

COURT_W = 610
COURT_H = 1340

DURATION_MIN    = 10
MIN_AREA        = 600          

DISPLAY_W       = 960
HEATMAP_DISP_W  = 400

BOTTOM_HALF_RATIO = 0.5
CENTER_X_MIN = 0.2
CENTER_X_MAX = 0.8

COLOR_BOX  = (0, 255, 0)
COLOR_FOOT = (0, 0, 255)

# ── Smooth bbox config ──────────────────────────────────────────────
SMOOTH_ALPHA   = 0.30
MIN_BOX_W      = 50
MAX_BOX_W      = 300    
MIN_BOX_H      = 40   
MAX_BOX_H      = 400
MISS_TOLERANCE = 45    
# ────────────────────────────────────────────────────────────────────


class SmoothTracker:
    def __init__(self):
        self.alpha     = SMOOTH_ALPHA
        self.miss_tol  = MISS_TOLERANCE
        self._smooth   = None   # (cx, cy, w, h) float
        self._miss     = 0
        self._last_vel = (0.0, 0.0)   # vận tốc cx, cy để dự đoán khi mất

    def _clamp(self, w, h):
        w = float(np.clip(w, MIN_BOX_W, MAX_BOX_W))
        h = float(np.clip(h, MIN_BOX_H, MAX_BOX_H))
        return w, h

    def update(self, roi_boxes):
        """
        roi_boxes : list (x,y,w,h) đã lọc ROI.
        Chỉ lấy box lớn nhất (người gần camera).
        Trả về (x,y,w,h) đã smooth hoặc None nếu mất quá lâu.
        """
        if roi_boxes:
            # Lấy box có diện tích lớn nhất
            best = max(roi_boxes, key=lambda b: b[2] * b[3])
            bx, by, bw, bh = best
            bw, bh = self._clamp(bw, bh)
            cx = float(bx + bw / 2)
            cy = float(by + bh / 2)

            if self._smooth is None:
                self._smooth = (cx, cy, bw, bh)
                self._last_vel = (0.0, 0.0)
            else:
                a = self.alpha
                scx, scy, sw, sh = self._smooth
                new_cx = a * cx  + (1-a) * scx
                new_cy = a * cy  + (1-a) * scy
                new_w  = a * bw  + (1-a) * sw
                new_h  = a * bh  + (1-a) * sh
                self._last_vel = (new_cx - scx, new_cy - scy)
                self._smooth = (new_cx, new_cy, new_w, new_h)

            self._miss = 0

        else:
            self._miss += 1
            if self._miss > self.miss_tol:
                self._smooth = None
                self._last_vel = (0.0, 0.0)
            elif self._smooth is not None:
                # Dự đoán vị trí bằng vận tốc cuối (coast)
                scx, scy, sw, sh = self._smooth
                vx, vy = self._last_vel
                # Giảm dần vận tốc khi không có detection
                vx *= 0.85
                vy *= 0.85
                self._last_vel = (vx, vy)
                self._smooth = (scx + vx, scy + vy, sw, sh)

        if self._smooth is None:
            return None

        cx, cy, w, h = self._smooth
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        return (x, y, int(w), int(h))


# ═══════════════════════════════════════════════════════════════════════
def detect_players(fg_mask, min_area=MIN_AREA):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN,  kernel)
    mask = cv.morphologyEx(mask,    cv.MORPH_CLOSE, kernel)
    mask = cv.dilate(mask, kernel, iterations=2)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv.boundingRect(cnt)

        aspect = h / (w + 1e-5)
        if aspect < 0.35:          
            continue
        if w * h < min_area:
            continue
        if h < 25 and w < 25:     
            continue
        boxes.append((x, y, w, h))
    return boxes


def get_foot_point(bbox):
    x, y, w, h = bbox
    return x + w // 2, y + h

def transform_point(M, fx, fy):
    pt  = np.float32([[[fx, fy]]])
    out = cv.perspectiveTransform(pt, M)
    return int(out[0][0][0]), int(out[0][0][1])


# ── Heatmap & drawing ───────────────────────────────────────────────
def render_heatmap(acc, court_img):
    HALF_Y   = COURT_H // 2
    acc_half = acc[HALF_Y:, :]
    blur     = cv.GaussianBlur(acc_half, (81, 81), 20)
    if blur.max() == 0:
        print('[WARN] Chua co du lieu heatmap!')
        court = np.ones((HALF_Y, COURT_W, 3), dtype=np.uint8)
        court[:] = (60, 140, 60)
        _draw_half_court_lines(court)
        return court
    blur   = np.power(blur, 0.4)
    norm   = cv.normalize(blur, None, 0, 255, cv.NORM_MINMAX)
    colored = cv.applyColorMap(np.uint8(norm), cv.COLORMAP_JET)
    court  = np.ones((HALF_Y, COURT_W, 3), dtype=np.uint8)
    court[:] = (60, 140, 60)
    result = cv.addWeighted(court, 0.3, colored, 0.7, 0)
    _draw_half_court_lines(result)
    _draw_legend(result)
    return result


def _draw_half_court_lines(img):
    WHITE  = (255, 255, 255)
    YELLOW = (0, 220, 220)
    t  = 3
    H  = img.shape[0]
    R  = COURT_W - 1
    CX = COURT_W // 2
    cv.line(img, (0, 0),   (COURT_W, 0), YELLOW, t + 2)
    cv.line(img, (0, 0),   (0, H-1),     WHITE,  t)
    cv.line(img, (R, 0),   (R, H-1),     WHITE,  t)
    cv.line(img, (0, H-1), (R, H-1),     WHITE,  t + 1)
    cv.line(img, (0, 198), (R, 198),     WHITE,  t)
    cv.line(img, (CX, 0),  (CX, 198),   WHITE,  t)
    cv.circle(img, (CX, 0), 6, YELLOW, -1)
    cv.putText(img, 'NET',       (CX - 15, 18), cv.FONT_HERSHEY_SIMPLEX, 0.5,  YELLOW, 2)
    cv.putText(img, 'BACK LINE', (5, H - 8),    cv.FONT_HERSHEY_SIMPLEX, 0.45, WHITE,  1)


def _draw_legend(img):
    bar_w, bar_h = 20, 150
    x0 = COURT_W - 30
    y0 = img.shape[0] - bar_h - 10
    for i in range(bar_h):
        val   = int(255 * (1 - i / bar_h))
        color = cv.applyColorMap(np.array([[val]], dtype=np.uint8),
                                 cv.COLORMAP_JET)[0][0]
        cv.line(img, (x0, y0+i), (x0+bar_w, y0+i), color.tolist(), 1)
    cv.rectangle(img, (x0, y0), (x0+bar_w, y0+bar_h), (255,255,255), 1)
    cv.putText(img, 'Nhieu', (x0-28, y0+8),     cv.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
    cv.putText(img, 'It',    (x0-10, y0+bar_h), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)


# ═══════════════════════════════════════════════════════════════════════
def main():
    try:
        M = np.load(M_PATH)
        print(f'[OK] Da load ma tran M tu {M_PATH}')
    except FileNotFoundError:
        print(f'[LOI] Khong tim thay {M_PATH}')
        return

    cap = cv.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f'[LOI] Khong mo duoc video: {VIDEO_PATH}')
        return

    fps = cap.get(cv.CAP_PROP_FPS) or 30
    max_frames = int(fps * 60 * DURATION_MIN)
    print(f'[INFO] FPS={fps:.1f} | Se xu ly {max_frames} frames ({DURATION_MIN} phut)')

    bg_sub      = cv.createBackgroundSubtractorMOG2(
                      history=500, varThreshold=40,   # giảm threshold để nhạy hơn
                      detectShadows=True)
    heatmap_acc = np.zeros((COURT_H, COURT_W), dtype=np.float32)
    tracker     = SmoothTracker()

    frame_count   = 0
    positions_log = []

    print('[START] Bat dau xu ly... Nhan Q de dung som.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count >= max_frames:
            break

        fg_mask   = bg_sub.apply(frame)
        all_boxes = detect_players(fg_mask)

        # Lọc ROI: nửa dưới sân + vùng X hợp lệ
        frame_h, frame_w = frame.shape[:2]
        split_y  = int(frame_h * BOTTOM_HALF_RATIO)
        x_min_px = int(frame_w * CENTER_X_MIN)
        x_max_px = int(frame_w * CENTER_X_MAX)

        roi_boxes = []
        for (x, y, w, h) in all_boxes:
            cy_box = y + h // 2
            cx_box = x + w // 2
            if cy_box < split_y:
                continue
            if cx_box < x_min_px or cx_box > x_max_px:
                continue
            roi_boxes.append((x, y, w, h))

        smooth_box = tracker.update(roi_boxes)

        display = frame.copy()

        if smooth_box is not None:
            sx, sy, sw, sh = smooth_box
            # Màu box: xanh lá khi tracking, vàng khi đang coast (mất detection)
            box_color = COLOR_BOX if roi_boxes else (0, 200, 255)
            cv.rectangle(display, (sx, sy), (sx+sw, sy+sh), box_color, 2)

            fx, fy = get_foot_point(smooth_box)
            cv.circle(display, (fx, fy), 5, COLOR_FOOT, -1)

            rx, ry = transform_point(M, fx, fy)
            if 0 <= rx < COURT_W and 0 <= ry < COURT_H:
                heatmap_acc[ry, rx] += 1
                positions_log.append((rx, ry, frame_count))

        # HUD
        elapsed_min = frame_count / (fps * 60)
        if smooth_box is None:
            status = 'Lost'
        elif not roi_boxes:
            status = f'Coasting ({tracker._miss}f)'
        else:
            status = 'Tracking'

        cv.putText(display,
                   f'Frame:{frame_count}/{max_frames} | '
                   f'Time:{elapsed_min:.1f}/{DURATION_MIN}min | {status}',
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.65, (150, 255, 0), 2)

        h_d, w_d = display.shape[:2]
        small = cv.resize(display, (DISPLAY_W, int(h_d * DISPLAY_W / w_d)))
        cv.imshow('Badminton Tracking', small)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv.destroyAllWindows()

    print(f'\n[INFO] Tong so diem vi tri: {len(positions_log)}')
    if not positions_log:
        print('[CANH BAO] Khong co du lieu. Kiem tra video va M.npy')
        return

    result = render_heatmap(heatmap_acc, COURT_IMG)
    cv.imwrite(OUTPUT_PATH, result)
    print(f'[OK] Da luu heatmap: {OUTPUT_PATH}')

    h_r, w_r = result.shape[:2]
    small_r  = cv.resize(result, (HEATMAP_DISP_W, int(h_r * HEATMAP_DISP_W / w_r)))
    cv.imshow('Heatmap Ket qua', small_r)
    print('Nhan phim bat ky de thoat...')
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()