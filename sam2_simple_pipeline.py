import os
import cv2
import json
from pathlib import Path
from typing import List, Dict

try:
    from sam2 import SamAutomaticMaskGenerator, sam_model_registry
except Exception as e:
    raise ImportError("SAM2 library is required. Install with `pip install git+https://github.com/facebookresearch/sam2.git`")

from scene_analyzer_vlm import SceneAnalyzer
from PIL import Image
import numpy as np

class NaiveTracker:
    """Simple IoU based tracker"""
    def __init__(self, iou_threshold: float = 0.5):
        self.tracks = []  # list of {'id': int, 'box': [x, y, w, h]}
        self.next_id = 1
        self.iou_threshold = iou_threshold

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
        yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        union = boxAArea + boxBArea - interArea
        return interArea / union if union > 0 else 0

    def update(self, boxes: List[List[int]]) -> List[int]:
        assignments = []
        new_tracks = []
        for box in boxes:
            matched_id = None
            best_iou = 0
            for track in self.tracks:
                iou = self._iou(box, track['box'])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    matched_id = track['id']
            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1
            new_tracks.append({'id': matched_id, 'box': box})
            assignments.append(matched_id)
        self.tracks = new_tracks
        return assignments

def process_video(video_path: str, output_dir: str, mask_generator, analyzer: SceneAnalyzer, tracker: NaiveTracker, sample_rate: int = 1) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_id = Path(video_path).stem
    results: Dict[int, List[Dict]] = {}
    frame_idx = 0
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % max(int(fps/sample_rate),1) != 0:
            frame_idx += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(rgb)
        boxes = [m['bbox'] for m in masks]
        ids = tracker.update(boxes)
        for m, obj_id in zip(masks, ids):
            x, y, w, h = [int(v) for v in m['bbox']]
            obj_crop = rgb[y:y+h, x:x+w].copy()
            mask_crop = m['segmentation'][y:y+h, x:x+w]
            obj_crop[mask_crop == 0] = 0
            pil_img = Image.fromarray(obj_crop)
            name = analyzer.identify_objects(pil_img)
            action = analyzer.identify_activities(pil_img)
            scene = analyzer.get_scene_description(pil_img)
            info = {
                'frame': frame_idx,
                'name': name,
                'action': action,
                'scene': scene
            }
            results.setdefault(obj_id, []).append(info)
        frame_idx += 1
    cap.release()
    with open(os.path.join(output_dir, f"{video_id}_analysis.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SAM2 segmentation + VLM analysis pipeline")
    parser.add_argument('video', help='input video path')
    parser.add_argument('--output_dir', default='sam2_results', help='directory to save results')
    parser.add_argument('--model_type', default='vit_b', help='SAM2 model type')
    parser.add_argument('--sample_rate', type=int, default=1, help='frames per second to process')
    args = parser.parse_args()

    sam_model = sam_model_registry[args.model_type](checkpoint=None)
    mask_generator = SamAutomaticMaskGenerator(sam_model)
    analyzer = SceneAnalyzer()
    tracker = NaiveTracker()
    process_video(args.video, args.output_dir, mask_generator, analyzer, tracker, args.sample_rate)
