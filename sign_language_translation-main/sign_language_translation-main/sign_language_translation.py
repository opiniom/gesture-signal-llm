import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image, ImageFont, ImageDraw

# ---------------------------------------------------------
# [설정] 제스처 맵핑
# ---------------------------------------------------------
gesture_map = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ',
    10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ',
    14: 'ㅏ', 15: 'ㅓ', 16: 'ㅗ', 17: 'ㅜ', 18: 'ㅡ', 19: 'ㅣ',
    20: 'ㅑ', 21: 'ㅕ', 22: 'ㅛ', 23: 'ㅠ', 24: 'ㅐ', 25: 'ㅒ', 26: 'ㅔ', 27: 'ㅖ', 28: 'ㅚ', 29: 'ㅟ', 30: 'ㅢ',
    31: "next", 32: "buffer_clear", 33: "clear", 
    34: "emotion", 35: "null"
}
EMOJI_MAP = {
    "ㅎㅌ": "💖",   # 하트
    "ㅋㅋ": "🤣",   # 웃음
    "ㅇㅋ": "👌",   # 오케이
    "ㄴㄴ": "🙅‍",   # 노노
    "ㅂㅇ": "👋"    # 바이
}
DOUBLE_CONS = {'ㄱ':'ㄲ', 'ㄷ':'ㄸ', 'ㅂ':'ㅃ', 'ㅅ':'ㅆ', 'ㅈ':'ㅉ'}

# ---------------------------------------------------------
# [함수] 한글 자모 합치기
# ---------------------------------------------------------
def join_jamos(jamo_str):
    if not jamo_str: return ""
    CHO_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    JONG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    try:
        chars = list(jamo_str)
        if len(chars) >= 2:
            cho = chars[0]
            jung = chars[1]
            jong = chars[2] if len(chars) > 2 else ''
            if cho in CHO_LIST and jung in JUNG_LIST:
                c_idx = CHO_LIST.index(cho)
                j_idx = JUNG_LIST.index(jung)
                k_idx = JONG_LIST.index(jong) if jong in JONG_LIST else 0
                uni_val = 0xAC00 + (c_idx * 21 * 28) + (j_idx * 28) + k_idx
                return chr(uni_val) + "".join(chars[3:])
        return "".join(chars)
    except:
        return "".join(jamo_str)

# ---------------------------------------------------------
# [준비] 모델 로드
# ---------------------------------------------------------
def sign_language_translation():
    file_path = "./gesture_train_data.csv"

    try:
        file = np.genfromtxt(file_path, delimiter=',')
        angle = file[:,:-1].astype(np.float32)
        label = file[:,-1].astype(np.float32)
        knn = cv2.ml.KNearest_create()
        knn.train(angle, cv2.ml.ROW_SAMPLE, label)
        print("✅ 학습 데이터 로드 완료!")
    except Exception as e:
        print(f"❌ 오류: {e}")
        exit()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    
    # 새로 추가한 해상도 올리는 코드
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    start_t = time.time()

    sentence = ''       
    jamo_buffer = []    
    hold = 2

    print("\n=== 🖐️ 심플 수어 번역기 (터미널 출력용) ===")
    print(" 영상 창은 손 뼈대만 보여줍니다. 결과는 여기에 나옵니다! 👇\n")

    screen_w = 1920   # 모니터 가로 해상도
    screen_h = 1080   # 모니터 세로 해상도

    # 전체화면용 창 생성
    cv2.namedWindow("Hand Tracking", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



    font_path = "C:/Windows/Fonts/malgun.ttf"   # PC에 맞게 수정
    font_korean = ImageFont.truetype(font_path, 40)
    
    imoji_path = "C:/Windows/Fonts/seguiemj.ttf"   # PC에 맞게 수정
    font_emoji = ImageFont.truetype(imoji_path, 40)

    while cap.isOpened():
        success, img = cap.read()
        if not success: 
            continue
        img = cv2.flip(img, 1)

        h, w = img.shape[:2]

    # 🔥 전체화면용 검은 캔버스 생성
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8) 
        # 좌측 상단에 카메라 붙이기
        canvas[0:h, 0:w] = img

        # PIL 변환 (OpenCV BGR -> RGB)
        img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        # ★ 포인트: 한글 출력 ★
        draw.text((50, screen_h - 120), f"입력중: {''.join(jamo_buffer)}", font=font_korean, fill=(0,255,0))
        # ★ 문장 출력 (한글 / 이모지 혼합)
        x = 50
        y = screen_h - 60

        draw.text((x, y), "문장: ", font=font_korean, fill=(0,255,0))
        x += font_korean.getbbox("문장: ")[2] - font_korean.getbbox("문장: ")[0]

        for ch in sentence:
            # 한글 범위
            # 글자 너비 계산
            if '\uAC00' <= ch <= '\uD7A3' or '\u1100' <= ch <= '\u11FF' or '\u3130' <= ch <= '\u318F':
                draw.text((x, y), ch, font=font_korean, fill=(0,255,0))
                x += font_korean.getbbox(ch)[2] - font_korean.getbbox(ch)[0]  # getbbox로 폭 계산
            else:
                draw.text((x, y), ch, font=font_emoji, fill=(0,255,0))
                x += font_emoji.getbbox(ch)[2] - font_emoji.getbbox(ch)[0]



        # draw.text((50, screen_h - 120), f"입력중: {''.join(jamo_buffer)}", font=imoji, fill=(0,255,0))
        # draw.text((50, screen_h - 60), f"문장: {sentence}", font=imoji, fill=(0,255,0))


        # PIL → OpenCV로 되돌리기
        canvas = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
                angle = np.degrees(angle)

                data = np.array([angle], dtype=np.float32)
                ret, results, neighbours, dist = knn.findNearest(data, 3)
                idx = int(results[0][0])

                if idx == 0 or idx == 1: 
                    if res.landmark[8].y < res.landmark[5].y: idx = 1 
                    else: idx = 0

                if idx in gesture_map:
                    action = gesture_map[idx]
                    
                    # 일정 시간 유지 시 입력
                    if time.time() - start_t > hold:
                        if action == 'buffer_clear':
                            # sentence += ' '
                            jamo_buffer = []
                            print(" [띄어쓰기] ")
                        elif action == 'clear':
                            sentence = ''
                            jamo_buffer = []
                            print("\n [지우기] \n")
                        elif action == 'next':
                            sentence += join_jamos(jamo_buffer)
                            jamo_buffer = []
                        elif action == 'emotion':
                            jamo_str = "".join(jamo_buffer)  # 지금까지 입력된 자모 문자열
                            if jamo_str in EMOJI_MAP:
                                jamo_buffer = []
                                sentence += EMOJI_MAP[jamo_str]  # 매핑된 이모지 추가
                            else:
                                sentence += join_jamos(jamo_buffer)  # 매핑 없으면 그냥 합치기
                                jamo_buffer = []
                                print(f"💖 이모션 동작 감지! (터미널 출력용) | 전체 문장: {sentence}")
                        elif action == 'null':
                            pass
                        else:
                            # 쌍자음 변환
                            if jamo_buffer and jamo_buffer[-1] == action and action in DOUBLE_CONS:
                                jamo_buffer[-1] = DOUBLE_CONS[action]
                            else:
                                if len(jamo_buffer) >= 3:
                                    sentence += join_jamos(jamo_buffer)
                                    jamo_buffer = [action]
                                else:
                                    jamo_buffer.append(action)
                        
                        start_t = time.time()

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Tracking', canvas)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 32: # Spacebar
            if jamo_buffer:
                sentence += join_jamos(jamo_buffer) + " "
                jamo_buffer = []
            else:
                sentence += " "
            print(f" [키보드 스페이스] 전체 문장: {sentence}")
            
        elif key == ord('c'): # Clear
            sentence = ""
            jamo_buffer = []
            print("\n [키보드 초기화] \n")

        elif key == 8: # Backspace
            if jamo_buffer:
                jamo_buffer.pop()
            elif sentence:
                sentence = sentence[:-1]
            print(f" [지움] 전체 문장: {sentence}")

    cap.release()
    cv2.destroyAllWindows()

sign_language_translation()