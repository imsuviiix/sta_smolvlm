import os
from pathlib import Path
import sys
from typing import Dict, List
import json
import time
import requests
import torch
sys.path.append("..")
sys.path.append("./sam")
from sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from aot_tracker import get_aot
import numpy as np
from tool.segmentor import Segmentor
from tool.detector import Detector
from tool.transfer_tools import draw_outline, draw_points
import cv2
from seg_track_anything import draw_mask
from PIL import Image
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from transformers import pipeline, AutoModelForCausalLM, AutoProcessor

class SegTracker():
    def __init__(self,segtracker_args, sam_args, aot_args, min_area_ratio=0.05, max_obj_num=4, max_area_ratio=0.7) -> None:
        """
         Initialize SAM and AOT.
        """
        self.sam = Segmentor(sam_args)
        self.tracker = get_aot(aot_args)
        self.detector = Detector(self.sam.device)
        self.sam_gap = segtracker_args['sam_gap']
        self.min_area_ratio = min_area_ratio #segtracker_args['min_area_ratio']
        self.max_obj_num = max_obj_num #segtracker_args['max_obj_num']
        self.min_new_obj_iou = segtracker_args['min_new_obj_iou']
        self.max_area_ratio = max_area_ratio  # 크롭된 이미지에서 허용되는 배경(흰색)의 최대 비율
        self.reference_objs_list = []
        self.object_idx = 1
        self.curr_idx = 1
        self.origin_merged_mask = None  # init by segment-everything or update
        self.first_frame_mask = None
        
        # 의미 있는 인스턴스 ID 추적 (key: instance_id, value: 이름)
        self.meaningful_instances = {}
        
        # 객체 일관성 추적을 위한 데이터 구조
        self.instance_history = {}  # key: original_id, value: {name, features, last_seen_frame}
        self.last_frame_time = -1.0
        self.frame_count = 0
        
        # LLM 판단 이력 (재판단 없이 이전 결과 재사용)
        self.instance_judgments = {}  # key: (instance_id, instance_name), value: True/False

        # SmolVLM 모델 초기화
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.smolvlm_pipe = None
        self.init_smolvlm_model()
        
        # 결과 저장을 위한 데이터 구조
        self.video_data = {}
        
        # debug
        self.everything_points = []
        self.everything_labels = []
        print("SegTracker has been initialized")

    def init_smolvlm_model(self):
        """SmolVLM 모델 초기화 - test_smolvlm_pipe.py와 동일한 방식"""
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
        self.model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
        
        # GPU 사용 가능시 모델을 GPU로 이동
        if self.device == "cuda":
            self.model = self.model.to(self.device)

        print(f"SmolVLM 모델이 {self.device}에서 초기화되었습니다.")

    def encode_image_to_base64(self, pil_image):
        """PIL 이미지를 base64로 인코딩"""
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    def _postprocess_mask(self, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """마스크 후처리: 작은 노이즈 제거 및 인접 마스크 통합"""
        if mask is None or mask.max() == 0:
            return mask
        
        processed_mask = np.zeros_like(mask)
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids != 0] # 배경 제외

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 1. 객체별 노이즈 제거 및 구멍 채우기
        for obj_id in unique_ids:
            obj_mask = (mask == obj_id).astype(np.uint8)
            
            # 열림 연산: 작은 점 제거
            opened_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel)
            
            # 닫힘 연산: 객체 내 작은 구멍 메우기
            closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
            
            # 면적이 min_area_ratio보다 작은 객체는 제거
            if closed_mask.sum() / (mask.shape[0] * mask.shape[1]) < self.min_area_ratio:
                continue
                
            # 객체 내부 구멍 완전히 채우기 (옵션)
            contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled_mask = np.zeros_like(closed_mask)
            cv2.drawContours(filled_mask, contours, -1, 1, -1)  # -1은 내부를 채우는 플래그

            processed_mask[filled_mask > 0] = obj_id
        
        # 2. 인접한 객체들 간의 관계 분석
        # 사람 객체로 추정되는 부분들(얼굴, 몸통 등)이 분리되어 있으면 통합
        dilation_kernel = np.ones((15, 15), np.uint8)
        ids = np.unique(processed_mask)
        ids = ids[ids != 0]
        
        # 크기 기준으로 정렬 (큰 객체부터)
        id_sizes = [(id, np.sum(processed_mask == id)) for id in ids]
        id_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # 병합 대상 객체 쌍 찾기
        merge_pairs = []
        
        for i in range(len(id_sizes)):
            id1, size1 = id_sizes[i]
            mask1 = (processed_mask == id1)
            dilated_mask1 = cv2.dilate(mask1.astype(np.uint8), dilation_kernel)
            
            for j in range(i+1, len(id_sizes)):
                id2, size2 = id_sizes[j]
                mask2 = (processed_mask == id2)
                
                # 두 객체가 가까이 있는지 확인
                if np.any(dilated_mask1 & mask2):
                    # 크기가 비슷하거나 작은 객체가 큰 객체와 인접해 있으면 병합
                    merge_pairs.append((id1, id2))
        
        # 병합 수행
        for id1, id2 in merge_pairs:
            # 작은 ID를 큰 ID로 병합 (ID 값이 작을수록 중요 객체)
            processed_mask[processed_mask == id2] = id1
        
        # 3. 최종 마스크 정리
        final_mask = np.zeros_like(processed_mask)
        next_id = 1
        
        for id in np.unique(processed_mask):
            if id == 0:
                continue
                
            # ID 재할당하여 연속적인 값 가지도록
            final_mask[processed_mask == id] = next_id
            next_id += 1
            
        return final_mask

    def analyze_instance_name(self, image, instance_id):
        """인스턴스 이름 추출"""
        prompt = """What is this object? 
    Note:Please provide only the name of the main object in 1-3 words. 
    If this is a nature like sky or not a meaningful object or just background, respond with 'background'.
    If you're uncertain or can't clearly identify the object, just say 'unknown'. 
    DO NOT make up information or hallucinate. Accuracy is more important than specificity."""
        
        try:
            # temperature 값을 낮춰서 보수적인 응답 유도
            response = self.analyze_image_with_smolvlm(image, instance_id, prompt, temperature=0.1)
            print(f"      이름 분석 원본 응답: {response[:150]}...")
            
            # 응답에서 실제 객체 이름만 추출
            lines = response.split('\n')
            for line in lines:
                if 'Assistant:' in line:
                    name = line.split('Assistant:')[-1].strip()
                    # 첫 번째 문장만 가져오기
                    name = name.split('.')[0].strip()
                    # 너무 길면 처음 몇 단어만
                    words = name.split()
                    if len(words) > 3:
                        name = ' '.join(words[:3])
                    # 불확실성 표현이 있으면 unknown으로 처리
                    uncertainty_words = ["hard to tell", "difficult to say", "not sure", "can't tell", "cannot tell"]
                    if any(phrase in name.lower() for phrase in uncertainty_words):
                        return "unknown"
                    return name.lower()
        except Exception as e:
            print(f"      이름 분석 중 오류: {e}")
        
        return "unknown"

    def analyze_instance_action(self, image, instance_id):
        """인스턴스 행동 추출"""
        prompt = """What is this object doing? 
    
    Describe the action or state in detail with 2-3 complete sentences.
    Include information about movement, position, interaction with surroundings, and any other relevant details.
    If the object is not doing anything specific or is just static, provide a detailed description of its current state.
    If you're uncertain or can't clearly see what the object is doing, just say 'unknown'. 
    DO NOT make up information or hallucinate. Accuracy is more important than specificity."""
        
        try:
            # temperature 값을 적절히 조정하여 자세한 응답 유도
            response = self.analyze_image_with_smolvlm(image, instance_id, prompt, temperature=0.5)
            print(f"      행동 분석 원본 응답: {response[:150]}...")
            
            # 응답에서 행동 추출 (전체 설명 유지)
            action = "The object is positioned in the frame."  # 기본값 설정
            
            lines = response.split('\n')
            found_assistant = False
            action_lines = []
            
            # Assistant: 이후의 모든 텍스트를 가져옴
            for line in lines:
                if 'Assistant:' in line:
                    found_assistant = True
                    # Assistant: 부분을 제거하고 나머지만 저장
                    if line.strip() != 'Assistant:':
                        text_after_assistant = line.split('Assistant:', 1)[1].strip()
                        if text_after_assistant:
                            action_lines.append(text_after_assistant)
                elif found_assistant and line.strip():
                    action_lines.append(line.strip())
            
            if action_lines:
                action = ' '.join(action_lines)
                
            # 응답이 없거나 비어있으면 기본값 사용
            if not action or action.strip() == "":
                print("      *** 경고: 응답에서 액션을 추출할 수 없습니다. 기본값을 사용합니다. ***")
                return "The object is positioned in the frame."
                
            # 불확실성 표현이 있으면 unknown으로 처리
            uncertainty_words = ["hard to tell", "difficult to say", "not sure", "can't tell", "cannot tell", "unclear"]
            if any(phrase in action.lower() for phrase in uncertainty_words):
                print("      *** 불확실한 응답이 감지되었습니다. 기본값을 사용합니다. ***")
                return "The object appears to be stationary in the image."
            
            print(f"      최종 추출된 액션: {action}")
            return action
            
        except Exception as e:
            print(f"      행동 분석 중 오류: {e}")
            print("      *** 예외 발생으로 기본값을 사용합니다. ***")
            return "The object is visible in the frame."

    def analyze_scene_description(self, image, instance_id):
        """장면 설명 추출"""
        prompt = """Describe this scene in 1-2 sentences.
    Note: Focus on what you can actually see and the context. 
    If this is just background or meaningless, respond with 'background scene'.
    If you're uncertain about what you're seeing, just say 'unclear scene'. 
    DO NOT make up information or hallucinate. Accuracy is more important than specificity."""
        
        try:
            # temperature 값을 낮춰서 보수적인 응답 유도
            response = self.analyze_image_with_smolvlm(image, instance_id, prompt, temperature=0.5)
            print(f"      장면 분석 원본 응답: {response[:150]}...")
            
            # 응답에서 설명 추출
            lines = response.split('\n')
            for line in lines:
                if 'Assistant:' in line:
                    description = line.split('Assistant:')[-1].strip()
                    # 처음 2문장만 가져오기
                    sentences = description.split('.')
                    if len(sentences) > 2:
                        description = '. '.join(sentences[:2]) + '.'
                    
                    # 불확실성 표현이 있으면 불확실한 장면으로 표시
                    uncertainty_words = ["hard to tell", "difficult to say", "not sure", "can't tell", "cannot tell", "unclear"]
                    if any(phrase in description.lower() for phrase in uncertainty_words):
                        return "Unclear scene, insufficient visual information"
                    
                    return description
        except Exception as e:
            print(f"      장면 분석 중 오류: {e}")
        
        return "No description available"

    def is_meaningful_instance(self, instance_id, instance_name, instance_action, scene_description, image):
        """
        LLM을 활용하여 인스턴스가 의미있는지 판단
        
        Args:
            instance_id: 인스턴스 ID
            instance_name: 인스턴스 이름
            instance_action: 인스턴스 행동
            scene_description: 장면 설명
            image: 인스턴스 이미지 (PIL Image)
        
        Returns:
            bool: 의미있는 인스턴스인지 여부
        """
        # 이미 meaningful로 판단된 인스턴스인지 확인
        if instance_id in self.meaningful_instances:
            print(f"    → 이전에 의미있는 인스턴스로 판단됨 (ID: {instance_id}, 이름: '{self.meaningful_instances[instance_id]}')")
            return True
        
        # 이전에 동일한 객체에 대한 판단 이력이 있는지 확인
        judgment_key = (instance_id, instance_name)
        if judgment_key in self.instance_judgments:
            result = self.instance_judgments[judgment_key]
            print(f"    → 이전 판단 결과 재사용: {'의미있음' if result else '의미없음'}")
            if result:
                self.meaningful_instances[instance_id] = instance_name
            return result
        
        # 기본 키워드 필터링 (빠른 필터링을 위해 유지)
        meaningless_keywords = ['background', 'rectangle', 'unknown', 'static', 'blank', 'empty']
        health_keywords = ['health', 'healthy', 'fitness', 'exercise', 'workout', 'diet', 'nutrition', 
                          'food', 'meal', 'ingredient', 'vitamin', 'supplement', 'medicine', 'medical',
                          'doctor', 'nurse', 'patient', 'hospital', 'clinic', 'therapy', 'treatment',
                          'remedy', 'cure', 'symptom', 'disease', 'illness', 'condition', 'pain',
                          'care', 'wellness', 'wellbeing', 'lifestyle', 'body', 'weight', 'muscle',
                          'bone', 'joint', 'heart', 'blood', 'pressure', 'sugar', 'cholesterol',
                          'sleep', 'rest', 'stress', 'mental', 'emotional', 'brain', 'mind']
        human_keywords = ['person', 'people', 'human', 'man', 'woman', 'child', 'baby', 'adult',
                         'face', 'hand', 'arm', 'leg', 'body', 'head', 'eye', 'mouth',
                         'hair', 'skin', 'host', 'presenter', 'instructor', 'expert', 'trainer',
                         'viewer', 'audience', 'patient', 'demonstrator', 'practitioner']
        important_objects = ['food', 'fruit', 'vegetable', 'meat', 'drink', 'water', 'pill', 'capsule', 
                            'medicine', 'device', 'equipment', 'tool', 'machine', 'book', 'chart', 
                            'graph', 'diagram', 'phone', 'app', 'computer', 'screen', 'bottle', 
                            'container', 'package', 'product', 'scale', 'watch', 'wearable']
        
        # 사람 관련 키워드가 있으면 무조건 의미있다고 판단
        if any(keyword in instance_name.lower() for keyword in human_keywords) or \
           any(keyword in instance_action.lower() for keyword in human_keywords) or \
           any(keyword in scene_description.lower() for keyword in human_keywords):
            print(f"    → 사람 관련 객체로 판단됨: 의미있는 인스턴스")
            self.instance_judgments[judgment_key] = True
            self.meaningful_instances[instance_id] = instance_name
            return True
        
        # 건강 관련 키워드가 있으면 의미있다고 판단
        if any(keyword in instance_name.lower() for keyword in health_keywords) or \
           any(keyword in instance_action.lower() for keyword in health_keywords) or \
           any(keyword in scene_description.lower() for keyword in health_keywords):
            print(f"    → 건강 관련 객체로 판단됨: 의미있는 인스턴스")
            self.instance_judgments[judgment_key] = True
            self.meaningful_instances[instance_id] = instance_name
            return True
        
        # 중요한 물체 관련 키워드가 있으면 의미있다고 판단
        if any(keyword in instance_name.lower() for keyword in important_objects) or \
           any(keyword in instance_action.lower() for keyword in important_objects) or \
           any(keyword in scene_description.lower() for keyword in important_objects):
            print(f"    → 중요한 물체로 판단됨: 의미있는 인스턴스")
            self.instance_judgments[judgment_key] = True
            self.meaningful_instances[instance_id] = instance_name
            return True
        
        # 이름이 명확하게 의미없는 키워드에 포함되면 빠르게 판단
        if any(keyword == instance_name.lower() for keyword in meaningless_keywords):
            self.instance_judgments[judgment_key] = False
            return False
        
        # 모든 정보가 비슷하게 의미없으면 빠르게 판단
        if (instance_name == "unknown" or 
            instance_action == "unknown" or 
            "background" in scene_description.lower()):
            self.instance_judgments[judgment_key] = False
            return False
        
        # LLM에게 중요성 판단 요청 (생활 건강 유튜브 컨텍스트 추가)
        prompt = f"""
    Look at this image and analyze the object carefully.
    
    Object name: {instance_name}
    Object action: {instance_action}
    Scene description: {scene_description}

    This is from a health and wellness related YouTube video. I need to determine if this object is significant.
    
    Consider an object SIGNIFICANT if it is:
    1. A person (host, expert, demonstrator) or any part of a human body
    2. Food, drink, or nutrition-related items
    3. Exercise or fitness equipment
    4. Health devices or wellness products
    5. Medicine, vitamins, or supplements
    6. Visual aids like charts, graphs, or demonstration props
    7. Objects being demonstrated or explained in the video
    8. Anything providing important context to the health topic
    
    Consider an object INSIGNIFICANT if it is:
    1. Background elements not related to the health topic
    2. Generic furniture or decorative objects
    3. Blurry, partial or unidentifiable objects
    4. Technical artifacts from video processing
    
    Based on the image and description, is this object significant for understanding the health content?
    Answer with only 'yes' if it is significant, or 'no' if it is insignificant.
    """
        
        try:
            # LLM에 판단 요청 - 건강 유튜브 맥락에 맞게 판단
            response = self.analyze_image_with_smolvlm(image, 0, prompt, 0.2)  # 낮은 온도로 명확한 판단 유도
            
            # 응답에서 yes/no 추출
            lines = response.split('\n')
            for line in lines:
                if 'Assistant:' in line:
                    answer = line.split('Assistant:')[-1].strip().lower()
                    # 응답에서 yes나 no 찾기
                    result = "no" not in answer
                    print(f"    → LLM 판단: {'의미있는' if result else '의미없는'} 인스턴스 ('{answer}')")
                    
                    # 결과 저장
                    self.instance_judgments[judgment_key] = result
                    if result:
                        self.meaningful_instances[instance_id] = instance_name
                    
                    return result
        except Exception as e:
            print(f"    → LLM 판단 중 오류: {e}, 기본 규칙으로 판단")
            # 오류 발생 시 더 관대하게 판단 (유튜브 영상이므로)
            result = True
            self.instance_judgments[judgment_key] = result
            self.meaningful_instances[instance_id] = instance_name
            return result
        
        # 기본적으로 의미있다고 판단 (건강 유튜브 영상이므로 더 관대하게)
        self.instance_judgments[judgment_key] = True
        self.meaningful_instances[instance_id] = instance_name
        return True
    
    def analyze_image_with_smolvlm(self, image, instance_id, prompt="What is it doing?", temperature=0.1):
        """
        이미지 분석 - test_smolvlm_pipe.py와 동일한 방식
        낮은 temperature로 할루시네이션 감소
        """
        if self.processor is None or self.model is None:
            self.init_smolvlm_model()
        
        # PIL Image로 변환 (필요시)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # 이미지 크기 조정 (오류 방지)
        image = image.resize((384, 384))
        
        # 질문과 함께 입력 준비
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # 입력 텍스트 생성
        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt_text, images=[image], return_tensors="pt")
        
        # GPU 사용시 입력을 GPU로 이동
        if self.device == "cuda":
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # 생성 - temperature 적용하여 할루시네이션 감소
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=512,  # 토큰 수 제한
            temperature=temperature,  # 낮은 temperature로 보수적 응답 유도
            do_sample=True,  # 샘플링 활성화 (temperature 적용 위해 필요)
            top_p=0.9  # 낮은 top_p로 보수적 응답 유도
        )
        
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return generated_text
    
    def parse_object_info(self, api_response):
        """SmolVLM 응답에서 객체 정보 파싱"""
        try:
            # 여기서는 간단한 예시만 구현. 실제 파싱은 응답 형식에 맞게 조정 필요
            instance_info = {
                "instance_name": "",
                "instance_action": "",
                "scene_description": ""
            }
            
            # 응답 파싱 로직
            lines = api_response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Name:") or line.startswith("Object:"):
                    instance_info["instance_name"] = line.split(":", 1)[1].strip()
                elif line.startswith("Action:") or line.startswith("Activity:"):
                    instance_info["instance_action"] = line.split(":", 1)[1].strip()
                elif line.startswith("Scene:") or line.startswith("Description:"):
                    instance_info["scene_description"] = line.split(":", 1)[1].strip()
            
            # 파싱된 정보가 없으면 전체 응답을 설명으로 사용
            if not instance_info["instance_name"] and not instance_info["instance_action"] and not instance_info["scene_description"]:
                instance_info["scene_description"] = api_response
            
            return instance_info
        except Exception as e:
            print(f"응답 파싱 중 오류: {e}")
            return {
                "instance_name": "Unknown",
                "instance_action": "Unknown",
                "scene_description": api_response,
                "error": str(e)
            }

    def extract_instance_features(self, image):
        """
        이미지에서 특징 추출 (간단한 히스토그램 기반)
        실제 환경에서는 CNN 기반 특징 추출기를 사용하는 것이 더 좋음
        """
        # 이미지가 PIL 이미지인 경우 numpy 배열로 변환
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # RGB 히스토그램 계산
        hist_r = cv2.calcHist([image], [0], None, [64], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [64], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [64], [0, 256])
        
        # 정규화
        cv2.normalize(hist_r, hist_r, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_g, hist_g, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
        
        # 특징 벡터 연결
        features = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
        return features
        
    def match_instance_with_history(self, instance_id, image, instance_name, frame_time):
        """
        현재 인스턴스를 이전 프레임의 인스턴스와 매칭
        유사도 기반으로 동일한 객체 판단
        """
        # 첫 번째 프레임이거나 시간 차이가 너무 큰 경우 (새 장면)
        if self.frame_count == 0 or (frame_time - self.last_frame_time) > 5.0:
            # 새로운 인스턴스로 등록
            features = self.extract_instance_features(image)
            self.instance_history[instance_id] = {
                'name': instance_name,
                'features': features,
                'last_seen_frame': self.frame_count,
                'consistent_id': instance_id  # 일관된 ID 유지
            }
            return instance_id  # 원래 ID 반환
            
        # 현재 인스턴스의 특징 추출
        features = self.extract_instance_features(image)
        
        # 가장 유사한 이전 인스턴스 찾기
        best_match_id = None
        best_match_score = 0
        threshold = 0.7  # 유사도 임계값 (조정 가능)
        
        for prev_id, info in self.instance_history.items():
            # 지난 3프레임 이내에 본 객체만 고려 (너무 오래된 객체는 제외)
            if self.frame_count - info['last_seen_frame'] > 3:
                continue
                
            # 이름이 완전히 다르면 다른 객체로 간주 (unknown 제외)
            if instance_name != "unknown" and info['name'] != "unknown" and instance_name != info['name']:
                continue
                
            # 히스토그램 상관관계로 유사도 계산
            score = cv2.compareHist(
                features.reshape(-1, 1), 
                info['features'].reshape(-1, 1), 
                cv2.HISTCMP_CORREL
            )
            
            if score > threshold and score > best_match_score:
                best_match_score = score
                best_match_id = prev_id
        
        if best_match_id is not None:
            # 매칭된 이전 인스턴스 정보 업데이트
            consistent_id = self.instance_history[best_match_id]['consistent_id']
            self.instance_history[instance_id] = {
                'name': instance_name if instance_name != "unknown" else self.instance_history[best_match_id]['name'],
                'features': features,  # 새 특징으로 업데이트
                'last_seen_frame': self.frame_count,
                'consistent_id': consistent_id  # 일관된 ID 유지
            }
            print(f"    → 이전 객체({best_match_id})와 매칭됨: 일관된 ID {consistent_id} 유지")
            return consistent_id
        else:
            # 새로운 인스턴스로 등록
            self.instance_history[instance_id] = {
                'name': instance_name,
                'features': features,
                'last_seen_frame': self.frame_count,
                'consistent_id': instance_id  # 새로운 일관된 ID 할당
            }
            return instance_id

    def visualize_masks(self, frame_rgb, mask, output_path=None, alpha=0.5):
        """
        모든 인스턴스 마스크를 시각화하여 하나의 이미지로 저장
        
        Args:
            frame_rgb: 원본 RGB 프레임
            mask: 인스턴스 마스크 (각 픽셀 값이 인스턴스 ID)
            output_path: 저장할 경로 (None이면 저장 안 함)
            alpha: 마스크 투명도 (0.0 ~ 1.0)
            
        Returns:
            시각화된 이미지 (numpy array, RGB)
        """
        # 원본 이미지 복사
        vis_image = frame_rgb.copy()
        
        # 유효한 인스턴스 ID 추출
        ids = np.unique(mask)
        ids = ids[ids != 0]  # 배경 제외
        
        if len(ids) == 0:
            if output_path:
                cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            return vis_image
            
        # 각 인스턴스에 대해 랜덤 색상 생성
        colors = {}
        for inst_id in ids:
            # 일관된 색상을 위해 ID 기반 색상 생성
            np.random.seed(int(inst_id * 100))
            colors[inst_id] = np.random.randint(0, 255, size=3).tolist()
            
        # 마스크 이미지 생성
        mask_image = np.zeros_like(frame_rgb)
        for inst_id in ids:
            color = colors[inst_id]
            mask_image[mask == inst_id] = color
            
        # 마스크와 원본 이미지 블렌딩
        vis_image = cv2.addWeighted(vis_image, 1-alpha, mask_image, alpha, 0)
        
        # 각 인스턴스 경계선 그리기
        for inst_id in ids:
            binary_mask = (mask == inst_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, colors[inst_id], 2)
            
            # 인스턴스 ID 표시 (큰 객체만)
            if np.sum(binary_mask) > 1000:  # 작은 객체는 ID 생략
                moments = cv2.moments(binary_mask)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    # 흰색 배경에 검은색 텍스트로 ID 표시
                    cv2.putText(vis_image, str(inst_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 5)
                    cv2.putText(vis_image, str(inst_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 0, 0), 2)
        
        # 저장
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
        return vis_image

    def save_instance_crops(self, frame_rgb: np.ndarray,
                        mask: np.ndarray,
                        save_dir: str,
                        video_id: str = "unknown",
                        frame_time: float = 0.0) -> Dict[int, Image.Image]:
        """
        인스턴스 크롭을 저장하고 SmolVLM으로 분석한 후 JSON으로 결과 저장
        """
        os.makedirs(save_dir, exist_ok=True)
        json_save_path = os.path.join(save_dir, f"{video_id}_analysis.json")

        # 새 프레임 처리 시작
        if frame_time != self.last_frame_time:
            self.frame_count += 1
            self.last_frame_time = frame_time

        # 비디오 ID가 데이터에 없으면 초기화
        if video_id not in self.video_data:
            self.video_data[video_id] = []

        # 현재 프레임 데이터 초기화
        frame_data = {
            "timestamp": frame_time,
            "instances": []
        }

        # 전체 마스킹 이미지 저장
        masked_frame_path = os.path.join(save_dir, f"{video_id}_time_{frame_time:.2f}_masked.png")
        self.visualize_masks(frame_rgb, mask, masked_frame_path)
        
        # 마스크 맵 이미지 저장 (각 인스턴스를 다른 색상으로)
        mask_colored = np.zeros_like(frame_rgb)
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids != 0]  # 배경 제외
        
        for idx, id in enumerate(unique_ids):
            # 각 ID마다 다른 색상 할당
            np.random.seed(int(id * 100))
            color = np.random.randint(0, 255, size=3).tolist()
            mask_colored[mask == id] = color
            
        mask_path = os.path.join(save_dir, f"{video_id}_time_{frame_time:.2f}_mask_map.png")
        cv2.imwrite(mask_path, cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR))

        crops: Dict[int, Image.Image] = {}
        ids = [int(i) for i in np.unique(mask) if i != 0]
        
        # 마스크 후처리 적용
        processed_mask = self._postprocess_mask(mask)
        if processed_mask is not None and processed_mask.max() > 0:
            mask = processed_mask
            # 후처리된 마스크도 저장
            processed_mask_path = os.path.join(save_dir, f"{video_id}_time_{frame_time:.2f}_processed_mask.png")
            self.visualize_masks(frame_rgb, mask, processed_mask_path)
        
        print(f"프레임 {frame_time:.2f}초: {len(ids)}개 인스턴스 발견")
        
        for inst_id in ids:
            bin_m = (mask == inst_id)
            if not bin_m.any():
                continue

            ys, xs = np.where(bin_m)
            if len(ys) == 0 or len(xs) == 0:
                continue
                
            # 객체 주변에 마진 추가 (최소 10픽셀, 또는 객체 크기의 10%)
            h, w = frame_rgb.shape[:2]
            margin_y = max(10, int((ys.max() - ys.min()) * 0.1))
            margin_x = max(10, int((xs.max() - xs.min()) * 0.1))
            
            y0 = max(0, ys.min() - margin_y)
            y1 = min(h - 1, ys.max() + margin_y)
            x0 = max(0, xs.min() - margin_x)
            x1 = min(w - 1, xs.max() + margin_x)
            
            # 크롭 크기가 너무 작으면 건너뜀
            if (y1 - y0) < 20 or (x1 - x0) < 20:
                print(f"  인스턴스 {inst_id} 무시: 크기가 너무 작음 ({y1-y0}x{x1-x0})")
                continue

            crop = frame_rgb[y0:y1+1, x0:x1+1].copy()
            crop_mask = bin_m[y0:y1+1, x0:x1+1]
            
            # 배경(흰색) 비율 계산을 위해 원본 크롭 저장
            original_crop = crop.copy()
            
            # 경계를 부드럽게 처리하기 위한 마스크 블러
            crop_mask_float = crop_mask.astype(np.float32)
            blurred_mask = cv2.GaussianBlur(crop_mask_float, (5, 5), 0)
            
            # 마스크 외부를 흰색으로 설정 (부드러운 전환)
            alpha = np.stack([blurred_mask] * 3, axis=2)
            white_bg = np.ones_like(crop) * 255
            crop = crop * alpha + white_bg * (1 - alpha)
            crop = crop.astype(np.uint8)
            
            # 배경(흰색) 비율 계산
            # RGB 값이 모두 240 이상인 픽셀 수 카운트 (거의 흰색)
            white_pixels = np.sum(np.all(crop >= 240, axis=2))
            total_pixels = crop.shape[0] * crop.shape[1]
            white_ratio = white_pixels / total_pixels
            
            # 흰색 비율이 너무 높으면 건너뜀
            if white_ratio > self.max_area_ratio:
                print(f"  인스턴스 {inst_id} 무시: 흰색 배경 비율({white_ratio:.2f})이 max_area_ratio({self.max_area_ratio:.2f})보다 높음")
                continue
                
            print(f"  인스턴스 {inst_id} 분석 중... (흰색 배경 비율: {white_ratio:.2f})")

            pil_img = Image.fromarray(crop)
            
            # SmolVLM으로 각각 분석 - 디버깅 출력 추가
            print(f"    이름 분석 중...")
            instance_name = self.analyze_instance_name(pil_img, inst_id)
            print(f"    이름 결과: '{instance_name}'")

            print(f"    행동 분석 중...")
            instance_action = self.analyze_instance_action(pil_img, inst_id)
            print(f"    행동 결과: '{instance_action}'")
            
            # 행동 결과가 비어있거나 None인 경우에 대한 디버깅
            if not instance_action or instance_action == "unknown":
                print(f"    *** 경고: 행동 분석 결과가 비어있거나 unknown입니다 ***")
                # 간단한 기본값 설정 (옵션)
                if not instance_action:
                    instance_action = "The object is positioned in the frame."

            print(f"    장면 분석 중...")
            scene_description = self.analyze_scene_description(pil_img, inst_id)
            print(f"    장면 결과: '{scene_description[:100]}...'")

            # 객체 일관성 유지를 위해 이전 프레임과 매칭
            consistent_id = self.match_instance_with_history(inst_id, crop, instance_name, frame_time)

            # 이미지를 포함하여 의미있는 인스턴스인지 확인 (LLM에 판단 요청)
            print(f"    중요도 판단 중...")
            is_meaningful = self.is_meaningful_instance(consistent_id, instance_name, instance_action, scene_description, pil_img)
                        
            if is_meaningful:
            # 인스턴스 정보 저장
                instance_data = {
                    "instance_id": consistent_id,  # 일관된 ID 사용
                    "original_id": inst_id,        # 디버깅용 원본 ID 저장
                    "instance_name": instance_name,
                    "instance_action": instance_action,
                    "scene_description": scene_description,
                    "is_meaningful": True,
                    "crop_path": f"{video_id}_time_{frame_time:.2f}_inst_{consistent_id:03d}.png",
                    "white_ratio": float(f"{white_ratio:.2f}")  # 흰색 비율 저장 (디버깅용)
                }
                
                # 행동 결과 저장 디버깅
                print(f"    저장되는 액션: '{instance_action}'")
                
                frame_data["instances"].append(instance_data)
                
                # 이미지 파일 저장
                pil_img.save(os.path.join(save_dir, f"{video_id}_time_{frame_time:.2f}_inst_{consistent_id:03d}.png"))
                crops[consistent_id] = pil_img
                print(f"    → 의미있는 인스턴스: Name='{instance_name}', Action='{instance_action}'")
                # meaningful_instances_count += 1
            else:
                print(f"    → 의미없는 인스턴스 (건너뜀)")
        # 의미 있는 인스턴스가 있는 경우에만 프레임 데이터 추가 및 JSON 저장
        if len(frame_data["instances"]) > 0:
            # 프레임 데이터를 비디오 데이터에 추가
            self.video_data[video_id].append(frame_data)
            
            # JSON 파일로 저장
            with open(json_save_path, 'w', encoding='utf-8') as f:
                json.dump(self.video_data, f, ensure_ascii=False, indent=2)
            
            print(f"분석 결과가 {json_save_path}에 저장되었습니다.")
        else:
            print(f"의미있는 인스턴스가 없어 저장을 건너뜁니다.")
        
        return crops


    def seg(self, frame):
        '''
        Arguments:
            frame: numpy array (h,w,3)
        Return:
            origin_merged_mask: numpy array (h,w)
        '''
        frame = frame[:, :, ::-1]
        
        # SAM 전체 객체 세그멘테이션
        anns = self.sam.everything_generator.generate(frame)

        # anns is a list recording all predictions in an image
        if len(anns) == 0:
            return
            
        # 프레임 크기 가져오기
        h, w = frame.shape[:2]
        
        # 넓은 영역을 차지하는 객체 먼저 처리 (사람일 가능성이 높은 객체)
        anns_sorted = sorted(anns, key=lambda x: x['area'], reverse=True)
        large_objects = [ann for ann in anns_sorted if ann['area'] > 0.05 * h * w]
        
        # merge all predictions into one mask (h,w)
        # note that the merged mask may lost some objects due to the overlapping
        self.origin_merged_mask = np.zeros(anns[0]['segmentation'].shape, dtype=np.uint8)
        idx = 1
        
        # 먼저 큰 객체들 처리 (주요 객체)
        for ann in large_objects:
            if ann['area'] > self.min_area_ratio * frame.shape[0] * frame.shape[1]:
                m = ann['segmentation']
                self.origin_merged_mask[m==1] = idx
                idx += 1
                self.everything_points.append(ann["point_coords"][0])
                self.everything_labels.append(1)
        
        # 나머지 작은 객체들 처리
        for ann in anns:
            # 이미 처리된 큰 객체는 건너뜀
            if ann in large_objects:
                continue
                
            if ann['area'] > self.min_area_ratio * frame.shape[0] * frame.shape[1]:
                m = ann['segmentation']
                
                # 이미 마스킹된 영역과 겹치는지 확인
                overlap_ratio = np.sum(m & (self.origin_merged_mask > 0)) / np.sum(m)
                
                # 50% 이상 겹치면 기존 객체의 일부로 간주하고 건너뜀
                if overlap_ratio > 0.5:
                    continue
                    
                self.origin_merged_mask[m==1] = idx
                idx += 1
                self.everything_points.append(ann["point_coords"][0])
                self.everything_labels.append(1)

        # 객체 ID 필터링 및 정리
        obj_ids = np.unique(self.origin_merged_mask)
        obj_ids = obj_ids[obj_ids!=0]

        self.object_idx = 1
        for id in obj_ids:
            if np.sum(self.origin_merged_mask==id) < self.min_area_ratio * frame.shape[0] * frame.shape[1] or self.object_idx > self.max_obj_num:
                self.origin_merged_mask[self.origin_merged_mask==id] = 0
            else:
                # 큰 객체 ID 유지, 작은 객체들의 ID 재할당
                old_id = id
                self.origin_merged_mask[self.origin_merged_mask==old_id] = self.object_idx
                self.object_idx += 1

        # 마스크 후처리로 노이즈 제거 및 경계 개선
        self.origin_merged_mask = self._postprocess_mask(self.origin_merged_mask)
        
        self.first_frame_mask = self.origin_merged_mask
        return self.origin_merged_mask
    
    def update_origin_merged_mask(self, updated_merged_mask):
        self.origin_merged_mask = updated_merged_mask
        # obj_ids = np.unique(updated_merged_mask)
        # obj_ids = obj_ids[obj_ids!=0]
        # self.object_idx = int(max(obj_ids)) + 1

    def reset_origin_merged_mask(self, mask, id):
        self.origin_merged_mask = mask
        self.curr_idx = id

    def add_reference(self,frame,mask,frame_step=0):
        '''
        Add objects in a mask for tracking.
        Arguments:
            frame: numpy array (h,w,3)
            mask: numpy array (h,w)
        '''
        self.reference_objs_list.append(np.unique(mask))
        self.curr_idx = self.get_obj_num()
        self.tracker.add_reference_frame(frame,mask, self.curr_idx, frame_step)

    def track(self,frame,update_memory=True):
        '''
        Track all known objects.
        Arguments:
            frame: numpy array (h,w,3)
        Return:
            origin_merged_mask: numpy array (h,w)
        '''
        pred_mask = self.tracker.track(frame)
        if update_memory:
            self.tracker.update_memory(pred_mask)
        return pred_mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    
    def get_tracking_objs(self):
        objs = set()
        for ref in self.reference_objs_list:
            objs.update(set(ref))
        objs = list(sorted(list(objs)))
        objs = [i for i in objs if i!=0]
        return objs
    
    def get_obj_num(self):
        objs = self.get_tracking_objs()
        if len(objs) == 0: return 0
        return int(max(objs))

    def find_new_objs(self, track_mask, seg_mask):
        '''
        Compare tracked results from AOT with segmented results from SAM. Select objects from background if they are not tracked.
        Arguments:
            track_mask: numpy array (h,w)
            seg_mask: numpy array (h,w)
        Return:
            new_obj_mask: numpy array (h,w)
        '''
        new_obj_mask = (track_mask==0) * seg_mask
        new_obj_ids = np.unique(new_obj_mask)
        new_obj_ids = new_obj_ids[new_obj_ids!=0]
        # 프레임 크기 기준 min_area 비율 적용
        min_area_ratio = 0.01  # 1% (필요시 조정)
        frame_area = seg_mask.shape[0] * seg_mask.shape[1]
        min_area = frame_area * min_area_ratio
        obj_num = self.curr_idx
        for idx in new_obj_ids:
            new_obj_area = np.sum(new_obj_mask==idx)
            obj_area = np.sum(seg_mask==idx)
            if new_obj_area/obj_area < self.min_new_obj_iou or new_obj_area < min_area\
                or obj_num > self.max_obj_num:
                new_obj_mask[new_obj_mask==idx] = 0
            else:
                new_obj_mask[new_obj_mask==idx] = obj_num
                obj_num += 1
        return new_obj_mask
        
    def restart_tracker(self):
        self.tracker.restart()

    def seg_acc_bbox(self, origin_frame: np.ndarray, bbox: np.ndarray,):
        ''''
        Use bbox-prompt to get mask
        Parameters:
            origin_frame: H, W, C
            bbox: [[x0, y0], [x1, y1]]
        Return:
            refined_merged_mask: numpy array (h, w)
            masked_frame: numpy array (h, w, c)
        '''
        # get interactive_mask
        interactive_mask = self.sam.segment_with_box(origin_frame, bbox)[0]
        refined_merged_mask = self.add_mask(interactive_mask)

        # draw mask
        masked_frame = draw_mask(origin_frame.copy(), refined_merged_mask)

        # draw bbox
        masked_frame = cv2.rectangle(masked_frame, bbox[0], bbox[1], (0, 0, 255))

        return refined_merged_mask, masked_frame

    def seg_acc_click(self, origin_frame: np.ndarray, coords: np.ndarray, modes: np.ndarray, multimask=True):
        '''
        Use point-prompt to get mask
        Parameters:
            origin_frame: H, W, C
            coords: nd.array [[x, y]]
            modes: nd.array [[1]]
        Return:
            refined_merged_mask: numpy array (h, w)
            masked_frame: numpy array (h, w, c)
        '''
        # get interactive_mask
        interactive_mask = self.sam.segment_with_click(origin_frame, coords, modes, multimask)

        refined_merged_mask = self.add_mask(interactive_mask)

        # draw mask
        masked_frame = draw_mask(origin_frame.copy(), refined_merged_mask)

        # draw points
        # self.everything_labels = np.array(self.everything_labels).astype(np.int64)
        # self.everything_points = np.array(self.everything_points).astype(np.int64)

        masked_frame = draw_points(coords, modes, masked_frame)

        # draw outline
        masked_frame = draw_outline(interactive_mask, masked_frame)

        return refined_merged_mask, masked_frame

    def add_mask(self, interactive_mask: np.ndarray):
        '''
        Merge interactive mask with self.origin_merged_mask
        Parameters:
            interactive_mask: numpy array (h, w)
        Return:
            refined_merged_mask: numpy array (h, w)
        '''
        if self.origin_merged_mask is None:
            self.origin_merged_mask = np.zeros(interactive_mask.shape,dtype=np.uint8)
 
        refined_merged_mask = self.origin_merged_mask.copy()
        refined_merged_mask[interactive_mask > 0] = self.curr_idx

        return refined_merged_mask
    
    def detect_and_seg(self, origin_frame: np.ndarray, grounding_caption, box_threshold, text_threshold, box_size_threshold=1, reset_image=False):
        '''
        Using Grounding-DINO to detect object acc Text-prompts
        Retrun:
            refined_merged_mask: numpy array (h, w)
            annotated_frame: numpy array (h, w, 3)
        '''
        # backup id and origin-merged-mask
        bc_id = self.curr_idx
        bc_mask = self.origin_merged_mask

        # get annotated_frame and boxes
        annotated_frame, boxes = self.detector.run_grounding(origin_frame, grounding_caption, box_threshold, text_threshold)
        for i in range(len(boxes)):
            bbox = boxes[i]
            if (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1]) > annotated_frame.shape[0] * annotated_frame.shape[1] * box_size_threshold:
                continue
            interactive_mask = self.sam.segment_with_box(origin_frame, bbox, reset_image)[0]
            refined_merged_mask = self.add_mask(interactive_mask)
            self.update_origin_merged_mask(refined_merged_mask)
            self.curr_idx += 1

        # reset origin_mask
        self.reset_origin_merged_mask(bc_mask, bc_id)

        return refined_merged_mask, annotated_frame

if __name__ == '__main__':
    from model_args import segtracker_args,sam_args,aot_args

    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    
    # ------------------ detect test ----------------------
    
    origin_frame = cv2.imread('/data2/cym/Seg_Tra_any/Segment-and-Track-Anything/debug/point.png')
    origin_frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
    grounding_caption = "swan.water"
    box_threshold = 0.25
    text_threshold = 0.25

    predicted_mask, annotated_frame = Seg_Tracker.detect_and_seg(origin_frame, grounding_caption, box_threshold, text_threshold)
    masked_frame = draw_mask(annotated_frame, predicted_mask)
    origin_frame = cv2.cvtColor(origin_frame, cv2.COLOR_RGB2BGR)

    cv2.imwrite('./debug/masked_frame.png', masked_frame)
    cv2.imwrite('./debug/x.png', annotated_frame)