import cv2 as cv
import numpy as np

COURT_W = 610
COURT_H = 1340

VIDEO_PATH = 'caulong.mp4'   
OUTPUT_M   = 'M.npy'
DISPLAY_W  = 960   
PREVIEW_H  = 700   

points     = []
clone      = None   
clone_disp = None   
scale      = 1.0    

def click_event(event, x, y, flags, param):
    global points, clone_disp, scale
    if event == cv.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            ox = int(x / scale)
            oy = int(y / scale)
            points.append((ox, oy))

            cv.circle(clone_disp, (x, y), 6, (0, 0, 255), -1)
            labels = ['Goc trai-tren', 'Goc phai-tren',
                      'Goc phai-duoi', 'Goc trai-duoi']
            cv.putText(clone_disp, labels[len(points)-1], (x+8, y-8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv.imshow('Setup - Click 4 goc san', clone_disp)

            if len(points) == 4:
                compute_and_save()

def compute_and_save():
  
    src = np.float32(points)
    dst = np.float32([
        [0,        0       ],
        [COURT_W,  0       ],
        [COURT_W,  COURT_H ],
        [0,        COURT_H ]
    ])
    M = cv.getPerspectiveTransform(src, dst)
    np.save(OUTPUT_M, M)
    print(f'[OK] Da luu ma tran M vao {OUTPUT_M}')
    print('Nhan phim bat ky de thoat...')

    warped = cv.warpPerspective(clone, M, (COURT_W, COURT_H))

    scale_p      = PREVIEW_H / COURT_H
    preview_w    = int(COURT_W * scale_p)
    warped_small = cv.resize(warped, (preview_w, PREVIEW_H))
    cv.imshow('Preview san sau transform', warped_small)

def main():
    global clone, clone_disp, scale

    cap = cv.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f'[LOI] Khong mo duoc video: {VIDEO_PATH}')
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print('[LOI] Khong doc duoc frame dau tien')
        return
    clone = frame.copy()
    h0, w0 = frame.shape[:2]
    scale       = DISPLAY_W / w0
    disp_h      = int(h0 * scale)
    clone_disp  = cv.resize(frame.copy(), (DISPLAY_W, disp_h))

    cv.namedWindow('Setup - Click 4 goc san')
    cv.setMouseCallback('Setup - Click 4 goc san', click_event)

    print('=== HUONG DAN ===')
    print('Click lan luot vao 4 goc san theo thu tu:')
    print('  1. Goc TRAI-TREN')
    print('  2. Goc PHAI-TREN')
    print('  3. Goc PHAI-DUOI')
    print('  4. Goc TRAI-DUOI')
    print('Nhan R de click lai, nhan Q de thoat.\n')

    cv.imshow('Setup - Click 4 goc san', clone_disp)

    while True:
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):   # reset nếu click nhầm
            points.clear()
            clone_disp = cv.resize(frame.copy(), (DISPLAY_W, disp_h))
            print('[RESET] Click lai tu dau...')
            cv.imshow('Setup - Click 4 goc san', clone_disp)

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()