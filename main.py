import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import os
import threading
from PIL import Image, ImageFont, ImageDraw

# ==========================================
# 0. 설정 및 매핑
# ==========================================
gesture_map = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ',
    10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ',
    14: 'ㅏ', 15: 'ㅓ', 16: 'ㅗ', 17: 'ㅜ', 18: 'ㅡ', 19: 'ㅣ',
    20: 'ㅑ', 21: 'ㅕ', 22: 'ㅛ', 23: 'ㅠ', 24: 'ㅐ', 25: 'ㅒ', 26: 'ㅔ', 27: 'ㅖ', 28: 'ㅚ', 29: 'ㅟ', 30: 'ㅢ',
    31: "next", 32: "buffer_clear", 33: "clear", 
    34: "emotion", 35: "null"
}

DOUBLE_CONS = {'ㄱ':'ㄲ', 'ㄷ':'ㄸ', 'ㅂ':'ㅃ', 'ㅅ':'ㅆ', 'ㅈ':'ㅉ'}

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

# ==========================================
# 1. 모델 로드 및 초기화
# ==========================================
def load_sign_language_model():
    file_path = "./sign_language_translation-main/sign_language_translation-main/gesture_train_data.csv"
    try:
        file = np.genfromtxt(file_path, delimiter=',')
        angle = file[:,:-1].astype(np.float32)
        label = file[:,-1].astype(np.float32)
        knn = cv2.ml.KNearest_create()
        knn.train(angle, cv2.ml.ROW_SAMPLE, label)
        print("✅ 수어 모델 (KNN) 로드 완료!")
        return knn
    except Exception as e:
        print(f"❌ 수어 모델 오류: {e}")
        return None

# ==========================================
# 2. 메인 파이프라인 클래스
# ==========================================
class GestureAgentPipeline:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.knn_sign = load_sign_language_model()

        self.jamo_buffer = []
        self.sentence = ""
        self.last_sign_time = time.time()
        self.last_pose_time = time.time()
        self.hold_time = 1.5  

    def process_hand_landmarks(self, hand_landmarks):
        if self.knn_sign is None: return -1
        joint = np.zeros((21, 3))
        for j, lm in enumerate(hand_landmarks.landmark):
            joint[j] = [lm.x, lm.y, lm.z]

        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
        v = v2 - v1
        norms = np.linalg.norm(v, axis=1)[:, np.newaxis]
        norms[norms == 0] = 1e-6
        v = v / norms
        
        angle = np.arccos(np.clip(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]), -1.0, 1.0))
        angle = np.degrees(angle)

        data = np.array([angle], dtype=np.float32)
        ret, results, neighbours, dist = self.knn_sign.findNearest(data, 3)
        idx = int(results[0][0])

        if idx == 0 or idx == 1: 
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y: idx = 1 
            else: idx = 0
            
        return idx

    def process_pose_landmarks(self, pose_landmarks):
        l_wrist = pose_landmarks.landmark[15]
        r_wrist = pose_landmarks.landmark[16]
        l_elbow = pose_landmarks.landmark[13]
        r_elbow = pose_landmarks.landmark[14]
        
        if l_wrist.visibility > 0.5 and r_wrist.visibility > 0.5:
            if r_wrist.x > l_wrist.x:
                if l_wrist.y < l_elbow.y and r_wrist.y < r_elbow.y:
                    return "crossed_arms"
        return "none"

    def evaluate_intent_with_llm(self, text_input):
        import requests
        
        prompt = f"""
        당신은 소음이 심한 공장/물류센터에서 작업자의 수신호를 텍스트로 변환한 결과를 해석하여 장비를 제어하는 AI 시스템입니다.
        작업자가 보낸 수어 텍스트 데이터(자음/모음 분리나 오타 가능성 있음)는 아래와 같습니다. 의도를 파악해서 시스템 명령어 메시지로 변환해주세요.
        
        입력 텍스트: "{text_input}"
        
        규칙:
        1. 텍스트가 "불 꺼", "ㅂㅜㄹㄲㅓ", "ㅂㄹㄲ" 등과 비슷한 의미면 -> "불을 끄겠습니다." 출력
        2. 그 외 의미 파악 시 짧고 간결하게 대답 (예: "에어컨을 켭니다.")
        3. 전혀 의미를 알 수 없다면 -> "알 수 없는 명령입니다." 출력
        4. 답변은 불필요한 설명 없이 오직 결과 텍스트만 출력하세요.
        """
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma2:2b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                return f"[로컬 AI 에러] 상태 코드: {response.status_code}"
        except Exception as e:
            return f"[로컬 AI 통신 실패] Ollama 구동 여부를 확인하세요. ({str(e)})"
            
    def _call_llm_async(self, text_input):
        """LLM 호출이 비디오 프레임을 멈추지 않도록 스레드로 분리"""
        def task():
            result = self.evaluate_intent_with_llm(text_input)
            print(f"🤖 [AI 응답]: {result}\n")
        threading.Thread(target=task, daemon=True).start()

    def run(self):
        cap = cv2.VideoCapture(0)
        print("🎥 카메라 파이프라인 시작 (종료: 'q', 스페이스바: 텍스트 전송)")
        
        # 폰트 로드 (윈도우 기본 맑은 고딕)
        font_path = "C:/Windows/Fonts/malgun.ttf"
        try:
            font = ImageFont.truetype(font_path, 30)
        except IOError:
            font = ImageFont.load_default()
        
        while cap.isOpened():
            success, img = cap.read()
            if not success: continue
                
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(img_rgb)
            
            # --- 수어 인식 (오른손 또는 왼손 감지 시) ---
            # 거울 모드이므로 신체의 오른손이 화면에서는 왼손(left_hand)으로 인식될 수 있음. 둘 다 체크
            hand_landmarks_to_process = results.right_hand_landmarks or results.left_hand_landmarks
            if hand_landmarks_to_process:
                idx = self.process_hand_landmarks(hand_landmarks_to_process)
                self.mp_drawing.draw_landmarks(img, hand_landmarks_to_process, self.mp_holistic.HAND_CONNECTIONS)
                
                if idx in gesture_map:
                    action = gesture_map[idx]
                    if time.time() - self.last_sign_time > self.hold_time:
                        if action == 'clear':
                            self.sentence = ''
                            self.jamo_buffer = []
                        elif action == 'next':
                            self.sentence += join_jamos(self.jamo_buffer)
                            self.jamo_buffer = []
                        elif action not in ['emotion', 'null', 'buffer_clear']:
                            if self.jamo_buffer and self.jamo_buffer[-1] == action and action in DOUBLE_CONS:
                                self.jamo_buffer[-1] = DOUBLE_CONS[action]
                            else:
                                if len(self.jamo_buffer) >= 3:
                                    self.sentence += join_jamos(self.jamo_buffer)
                                    self.jamo_buffer = [action]
                                else:
                                    self.jamo_buffer.append(action)
                        self.last_sign_time = time.time()

            # --- 제스처 인식 ---
            pose_action = "none"
            if results.pose_landmarks:
                pose_action = self.process_pose_landmarks(results.pose_landmarks)
                self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
                
                if pose_action == "crossed_arms" and (time.time() - self.last_pose_time > 3.0):
                    print("\n🛑 [긴급 제어] 팔 교차 인식됨 -> 즉시 장비 정지 신호 전송 (LLM 우회)")
                    print("🤖 [시스템 응답]: 모든 작동을 멈춥니다.\n")
                    self.last_pose_time = time.time()
                    
            # --- 한글 텍스트 화면 출력 (PIL 사용) ---
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((10, 20), f"입력중: {''.join(self.jamo_buffer)}", font=font, fill=(0, 255, 0))
            draw.text((10, 60), f"문장: {self.sentence}", font=font, fill=(255, 255, 0))
            draw.text((10, 100), f"동작: {pose_action}", font=font, fill=(255, 100, 100))
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            cv2.imshow('Gesture Agent', img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 8: # 백스페이스 (Backspace)
                if self.jamo_buffer:
                    self.jamo_buffer.pop()
                elif self.sentence:
                    self.sentence = self.sentence[:-1]
            elif key == 32: # 스페이스바로 LLM 전송
                full_text = self.sentence + join_jamos(self.jamo_buffer)
                print(f"\n👉 [수어 텍스트] LLM에 전달: {full_text}")
                self._call_llm_async(full_text)
                self.sentence = ""
                self.jamo_buffer = []
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pipeline = GestureAgentPipeline()
    pipeline.run()
