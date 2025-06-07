#!/usr/bin/env python3
import os
import sys
import cv2
import json
import time
import argparse
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool


class SceneAnalyzer:
    """VLM을 사용하여 비디오 장면을 직접 분석하는 클래스"""
    
    def __init__(self, model_name="HuggingFaceTB/SmolVLM-500M-Instruct", device=None):
        """
        모델 초기화
        
        Args:
            model_name: 사용할 VLM 모델 이름
            device: 사용할 장치 (None이면 자동 감지)
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"디바이스 {self.device}에서 모델 {model_name} 로드 중...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name)
        
        if self.device == "cuda":
            self.model = self.model.to(self.device)
            
        print(f"모델 로드 완료. {self.device} 사용")
        
        # 결과 저장을 위한 데이터 구조
        self.analysis_results = {}
        
    def analyze_frame(self, frame: np.ndarray, timestamp: float, scene_id: str) -> Dict:
        """
        비디오 프레임 분석
        
        Args:
            frame: RGB 이미지 (numpy array)
            timestamp: 프레임 타임스탬프 (초)
            scene_id: 비디오/장면 ID
            
        Returns:
            프레임 분석 결과 딕셔너리
        """
        # 결과 저장용 데이터 구조
        frame_data = {
            "timestamp": timestamp,
            "objects": [],
            "scene_description": "",
            "activities": []
        }
        
        # PIL 이미지로 변환
        if not isinstance(frame, Image.Image):
            pil_img = Image.fromarray(frame)
        else:
            pil_img = frame
            
        # 1. 장면 설명 추출
        scene_desc = self.get_scene_description(pil_img)
        frame_data["scene_description"] = scene_desc
        print(f"장면 설명: {scene_desc}")
        
        # 2. 객체 인식
        objects = self.identify_objects(pil_img)
        frame_data["objects"] = objects
        print(f"인식된 객체: {objects}")
        
        # 3. 액션/활동 인식
        activities = self.identify_activities(pil_img)
        frame_data["activities"] = activities
        print(f"인식된 활동: {activities}")
        
        # 결과 저장
        if scene_id not in self.analysis_results:
            self.analysis_results[scene_id] = []
            
        self.analysis_results[scene_id].append(frame_data)
        
        return frame_data
    
    def get_scene_description(self, image: Image.Image) -> str:
        """전체 장면 설명 추출"""
        prompt = """Describe this scene in 1-2 detailed sentences. 
        Focus on the main elements and activities in the scene."""
        
        response = self._get_vlm_response(image, prompt)
        
        # 응답에서 장면 설명 추출
        for line in response.split('\n'):
            if 'Assistant:' in line:
                description = line.split('Assistant:')[-1].strip()
                return description
                
        return response  # 폴백
    
    def identify_objects(self, image: Image.Image) -> List[Dict]:
        """장면 내 주요 객체 식별"""
        prompt = """List all important objects and people in this scene.
        For each object or person, provide:
        1. A short name or label
        2. A brief description of its appearance
        
        Format your response as a list:
        - Object1: description
        - Object2: description
        
        Focus on the most significant 3-5 objects/people that are relevant to understanding the scene."""
        
        response = self._get_vlm_response(image, prompt)
        
        # 응답에서 객체 목록 추출
        objects = []
        capturing = False
        object_text = ""
        
        for line in response.split('\n'):
            if 'Assistant:' in line:
                capturing = True
                line = line.split('Assistant:')[-1].strip()
                
            if capturing and line.strip():
                if line.strip().startswith('-'):
                    if object_text:  # 이전 객체가 있으면 저장
                        objects.append(self._parse_object(object_text))
                    object_text = line.strip()[1:].strip()  # '-' 제거
                else:
                    object_text += " " + line.strip()
        
        # 마지막 객체 처리
        if object_text:
            objects.append(self._parse_object(object_text))
            
        return objects
    
    def _parse_object(self, text: str) -> Dict:
        """객체 텍스트 파싱"""
        if ':' in text:
            name, description = text.split(':', 1)
            return {
                "name": name.strip(),
                "description": description.strip()
            }
        else:
            return {
                "name": "unknown",
                "description": text.strip()
            }
    
    def identify_activities(self, image: Image.Image) -> List[str]:
        """장면 내 주요 활동/액션 식별"""
        prompt = """List the main activities or actions happening in this scene.
        Be specific about what each person or main object is doing.
        Format as a simple list of 2-4 short action phrases."""
        
        response = self._get_vlm_response(image, prompt)
        
        # 응답에서 활동 목록 추출
        activities = []
        capturing = False
        
        for line in response.split('\n'):
            if 'Assistant:' in line:
                capturing = True
                line = line.split('Assistant:')[-1].strip()
                
            if capturing and line.strip():
                if line.strip().startswith('-'):
                    activities.append(line.strip()[1:].strip())
                elif line.strip()[0].isdigit() and '. ' in line:
                    # 숫자 형식 목록 (1. 2. 등)
                    activity = line.strip().split('. ', 1)[1].strip()
                    activities.append(activity)
        
        # 목록 형식이 아니면 전체 텍스트를 분할
        if not activities and capturing:
            text = response.split('Assistant:')[-1].strip()
            # 문장 단위로 분할
            activities = [s.strip() for s in text.split('.') if s.strip()]
            
        return activities
    
    def _get_vlm_response(self, image: Image.Image, prompt: str, temperature: float = 0.2) -> str:
        """VLM에 이미지와 프롬프트를 전송하고 응답을 받음"""
        try:
            # 이미지 크기 조정 (필요한 경우)
            image = image.resize((384, 384))
            
            # 입력 준비
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 프롬프트 생성
            prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt_text, images=[image], return_tensors="pt")
            
            # GPU로 이동 (필요시)
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # 응답 생성
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=256,  # 응답 길이 제한
                temperature=temperature,  # 응답 다양성 조절
                do_sample=True,
                top_p=0.9
            )
            
            # 디코딩
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return generated_text
            
        except Exception as e:
            print(f"VLM 응답 중 오류: {e}")
            return f"Error: {str(e)}"
    
    def save_results(self, output_path: str) -> None:
        """분석 결과를 JSON 파일로 저장"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, ensure_ascii=False, indent=2)
            print(f"분석 결과가 {output_path}에 저장되었습니다.")
        except Exception as e:
            print(f"결과 저장 중 오류: {e}")
    
    def visualize_frame_analysis(self, 
                               frame: np.ndarray, 
                               frame_data: Dict, 
                               output_path: Optional[str] = None) -> np.ndarray:
        """
        프레임 분석 결과를 시각화
        
        Args:
            frame: 원본 RGB 프레임
            frame_data: 프레임 분석 결과
            output_path: 저장할 경로 (None이면 저장 안 함)
            
        Returns:
            시각화된 이미지
        """
        # 이미지 복사
        vis_image = frame.copy()
        h, w = vis_image.shape[:2]
        
        # 반투명한 오버레이 추가
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (0, 0), (w, int(h*0.3)), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, int(h*0.7)), (w, h), (0, 0, 0), -1)
        
        # 알파 블렌딩
        alpha = 0.7
        vis_image = cv2.addWeighted(overlay, alpha, vis_image, 1-alpha, 0)
        
        # 장면 설명 텍스트 추가
        scene_desc = frame_data.get("scene_description", "")
        cv2.putText(vis_image, "Scene:", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 여러 줄로 텍스트 렌더링
        y_offset = 60
        for i, line in enumerate(self._wrap_text(scene_desc, 50)):
            if i < 3:  # 최대 3줄까지
                cv2.putText(vis_image, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 1)
                y_offset += 30
        
        # 객체 목록 추가
        objects = frame_data.get("objects", [])
        y_offset = int(h*0.75)
        cv2.putText(vis_image, "Objects:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        y_offset += 30
        
        for i, obj in enumerate(objects):
            if i < 4:  # 최대 4개 객체까지
                obj_text = f"- {obj.get('name', '')}: {obj.get('description', '')[:30]}"
                cv2.putText(vis_image, obj_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
                y_offset += 30
        
        # 액션 목록 추가 (오른쪽 상단)
        activities = frame_data.get("activities", [])
        y_offset = 30
        cv2.putText(vis_image, "Activities:", (w-250, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        y_offset += 30
        
        for i, activity in enumerate(activities):
            if i < 4:  # 최대 4개 활동까지
                cv2.putText(vis_image, f"- {activity[:30]}", (w-250, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 30
        
        # 타임스탬프 추가
        timestamp = frame_data.get("timestamp", 0)
        timestamp_text = f"Time: {timestamp:.2f}s"
        cv2.putText(vis_image, timestamp_text, (w-150, h-20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 1)
        
        # 저장
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
        return vis_image
    
    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        """텍스트를 지정된 최대 폭에 맞게 여러 줄로 분할"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            if len(' '.join(current_line + [word])) <= max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                
        if current_line:
            lines.append(' '.join(current_line))
            
        return lines


def process_video(video_path: str, output_dir: str, sample_rate: int = 1, gpu_id: int = 0) -> None:
    """
    비디오 처리 함수
    
    Args:
        video_path: 처리할 비디오 경로
        output_dir: 결과 저장 디렉토리
        sample_rate: 샘플링 비율 (프레임당 초)
        gpu_id: 사용할 GPU ID
    """
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 비디오 ID 추출
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    # 결과 디렉토리 생성
    video_output_dir = os.path.join(output_dir, video_id)
    Path(video_output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"비디오 처리 시작: {video_id}")
    
    # 장면 분석기 초기화
    analyzer = SceneAnalyzer()
    
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {video_path}")
        return
    
    # 비디오 정보
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"비디오 정보: {video_id}")
    print(f"  - 원본 FPS: {fps}")
    print(f"  - 총 프레임 수: {frame_count}")
    print(f"  - 해상도: {width}x{height}")
    print(f"  - 처리 FPS: {sample_rate} (매 {int(fps/sample_rate)}번째 프레임 처리)")
    
    # 프로그레스 바 초기화
    total_frames = int(frame_count / (fps/sample_rate))
    pbar = tqdm(total=total_frames, desc=f"처리 중: {video_id}")
    
    # 메타데이터 초기화
    metadata = {
        "video_id": video_id,
        "video_path": video_path,
        "processing_start_time": time.time(),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps,
        "sample_rate": sample_rate
    }
    
    frame_idx = 0
    processed_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 샘플링: 지정된 비율에 따라 프레임 처리
            if frame_idx % int(fps/sample_rate) != 0:
                frame_idx += 1
                continue
            
            # 현재 시간(초) 계산
            current_time = frame_idx / fps
            
            # BGR에서 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            print(f"\n프레임 {processed_count+1} ({current_time:.2f}초) 분석 중...")
            
            # 프레임 분석
            frame_data = analyzer.analyze_frame(frame_rgb, current_time, video_id)
            
            # 분석 결과 시각화 및 저장
            # vis_image = analyzer.visualize_frame_analysis(
            #     frame_rgb, 
            #     frame_data, 
            #     os.path.join(video_output_dir, f"{video_id}_time_{current_time:.2f}_analysis.jpg")
            # )
            
            # 원본 프레임도 저장
            cv2.imwrite(
                os.path.join(video_output_dir, f"{video_id}_time_{current_time:.2f}_frame.jpg"),
                frame
            )
            
            processed_count += 1
            frame_idx += 1
            pbar.update(1)
            
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        pbar.close()
    
    # 메타데이터 업데이트 및 저장
    metadata["processing_end_time"] = time.time()
    metadata["processing_duration"] = metadata["processing_end_time"] - metadata["processing_start_time"]
    metadata["processed_frames"] = processed_count
    
    with open(os.path.join(video_output_dir, f"{video_id}_metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # JSON 파일 저장 경로 설정
    json_save_dir = os.path.join(video_output_dir, 'json_results')
    Path(json_save_dir).mkdir(parents=True, exist_ok=True)

    # 분석 결과 저장
    analyzer.save_results(os.path.join(json_save_dir, f'{video_id}_analysis.json'))
    
    print(f"\n총 {processed_count}개 프레임 처리 완료")
    print(f"처리 시간: {metadata['processing_duration']:.2f}초")


def process_image_sequence(image_dir: str, output_dir: str, ext: str = '.jpg', gpu_id: int = 0) -> None:
    """
    이미지 시퀀스 처리 함수
    
    Args:
        image_dir: 이미지 시퀀스 디렉토리
        output_dir: 결과 저장 디렉토리
        ext: 이미지 파일 확장자
        gpu_id: 사용할 GPU ID
    """
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 시퀀스 ID 추출
    seq_id = os.path.basename(image_dir)
    
    # 결과 디렉토리 생성
    seq_output_dir = os.path.join(output_dir, seq_id)
    Path(seq_output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"이미지 시퀀스 처리 시작: {seq_id}")
    
    # 이미지 파일 목록 가져오기
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(ext)])
    
    if not image_files:
        print(f"이미지를 찾을 수 없습니다: {image_dir} (확장자: {ext})")
        return
    
    print(f"이미지 시퀀스 정보: {seq_id}")
    print(f"  - 총 이미지 수: {len(image_files)}")
    
    # 장면 분석기 초기화
    analyzer = SceneAnalyzer()
    
    # 프로그레스 바 초기화
    pbar = tqdm(total=len(image_files), desc=f"처리 중: {seq_id}")
    
    # 메타데이터 초기화
    metadata = {
        "sequence_id": seq_id,
        "image_dir": image_dir,
        "processing_start_time": time.time(),
        "total_images": len(image_files)
    }
    
    processed_count = 0
    
    try:
        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(image_dir, img_file)
            
            # 이미지 로드
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"이미지를 열 수 없습니다: {img_path}")
                continue
            
            # 현재 시간(초) 계산 - 인덱스를 시간으로 사용
            current_time = idx / 1.0  # 1fps로 가정
            
            # BGR에서 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            print(f"\n이미지 {idx+1}/{len(image_files)} ({img_file}) 분석 중...")
            
            # 프레임 분석
            frame_data = analyzer.analyze_frame(frame_rgb, current_time, seq_id)
            
            # 분석 결과 시각화 및 저장
            # vis_image = analyzer.visualize_frame_analysis(
            #     frame_rgb, 
            #     frame_data, 
            #     os.path.join(seq_output_dir, f"{seq_id}_time_{current_time:.2f}_analysis.jpg")
            # )
            
            processed_count += 1
            pbar.update(1)
            
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pbar.close()
    
    # 메타데이터 업데이트 및 저장
    metadata["processing_end_time"] = time.time()
    metadata["processing_duration"] = metadata["processing_end_time"] - metadata["processing_start_time"]
    metadata["processed_images"] = processed_count
    
    with open(os.path.join(seq_output_dir, f"{seq_id}_metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # 분석 결과 저장
    analyzer.save_results(os.path.join(seq_output_dir, f"{seq_id}_analysis.json"))
    
    print(f"\n총 {processed_count}개 이미지 처리 완료")
    print(f"처리 시간: {metadata['processing_duration']:.2f}초")


def process_video_directory(video_dir: str, output_dir: str, sample_rate: int = 1, gpu_ids: List[int] = [0]) -> None:
    """
    비디오 디렉토리 처리 함수
    
    Args:
        video_dir: 비디오 파일들이 있는 디렉토리 경로
        output_dir: 결과 저장 디렉토리
        sample_rate: 샘플링 비율 (프레임당 초)
        gpu_ids: 사용할 GPU ID들
    """
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))
    
    # 비디오 파일 목록 가져오기
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print(f"비디오 파일을 찾을 수 없습니다: {video_dir}")
        return
    
    # 멀티프로세싱 풀 생성
    with Pool(processes=len(gpu_ids)) as pool:
        pool.starmap(process_video, [(os.path.join(video_dir, video_file), output_dir, sample_rate, gpu_ids[i % len(gpu_ids)]) for i, video_file in enumerate(video_files)])


def main():
    parser = argparse.ArgumentParser(description="VLM을 사용한 비디오/이미지 장면 분석")
    parser.add_argument("--input", required=True, help="입력 비디오 파일 또는 이미지 디렉토리 경로")
    parser.add_argument("--output_dir", default="./vlm_results", help="결과를 저장할 디렉토리 경로")
    parser.add_argument("--mode", choices=["video", "images"], default="video", help="입력 타입 (video 또는 images)")
    parser.add_argument("--gpu_id", type=str, default="0", help="사용할 GPU ID들 (쉼표로 구분)")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="비디오 샘플링 비율 (초당)")
    parser.add_argument("--image_ext", default=".jpg", help="이미지 파일 확장자 (이미지 모드에서만 사용)")
    
    args = parser.parse_args()
    
    # 입력 유효성 검사
    if not os.path.exists(args.input):
        print(f"입력 경로가 존재하지 않습니다: {args.input}")
        return
    
    # 출력 디렉토리 생성
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # GPU ID 파싱
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_id.split(',')]
        
        # 모드에 따라 처리
        if args.mode == "video":
            if os.path.isdir(args.input):
                process_video_directory(args.input, args.output_dir, args.sample_rate, gpu_ids)
            else:
                process_video(args.input, args.output_dir, args.sample_rate, gpu_ids[0])
        else:
            process_image_sequence(args.input, args.output_dir, args.image_ext, gpu_ids[0])
        
        total_time = time.time() - start_time
        print(f"총 처리 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 