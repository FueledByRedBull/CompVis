"""Emotion Analyzer - 3-Model Ensemble with dlib 68-point landmarks."""

from collections import Counter
from typing import Optional, Dict, List, Tuple

import numpy as np
import cv2
import dlib
import os
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

# Emotion descriptions (simplified)
EMOTION_DESCRIPTIONS = {
    'angry': 'anger or frustration',
    'disgust': 'disgust or displeasure',
    'fear': 'fear or anxiety',
    'happy': 'happiness or joy',
    'sad': 'sadness or melancholy',
    'surprise': 'surprise or astonishment',
    'neutral': 'a neutral state',
    'contempt': 'contempt or disdain'
}

EMOTION_LABELS = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Per-emotion weights based on model strengths
# enet_b2: Best overall, especially Happy/Neutral
# vgaf: Better at natural expressions (Sad, Surprise)
# afew: Better at acted expressions (Fear, Angry, Disgust)
EMOTION_WEIGHTS = {
    'angry':    {'enet_b2': 0.3, 'vgaf': 0.2, 'afew': 0.5},
    'contempt': {'enet_b2': 0.4, 'vgaf': 0.3, 'afew': 0.3},
    'disgust':  {'enet_b2': 0.3, 'vgaf': 0.2, 'afew': 0.5},
    'fear':     {'enet_b2': 0.2, 'vgaf': 0.3, 'afew': 0.5},
    'happy':    {'enet_b2': 0.5, 'vgaf': 0.3, 'afew': 0.2},
    'neutral':  {'enet_b2': 0.5, 'vgaf': 0.3, 'afew': 0.2},
    'sad':      {'enet_b2': 0.3, 'vgaf': 0.5, 'afew': 0.2},
    'surprise': {'enet_b2': 0.3, 'vgaf': 0.5, 'afew': 0.2},
}

HIGH_CONFIDENCE = 60.0
MEDIUM_CONFIDENCE = 40.0


class EmotionAnalyzer:
    """3-model ensemble emotion analyzer with dlib landmarks."""

    def __init__(self, use_ensemble: bool = True):
        self.use_ensemble = use_ensemble

        # dlib face detector and 68-point landmark predictor
        print("Loading dlib face detector...")
        self.face_detector = dlib.get_frontal_face_detector()

        # Find shape predictor file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        predictor_path = os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat")

        if not os.path.exists(predictor_path):
            raise FileNotFoundError(
                f"Shape predictor not found at {predictor_path}. "
                "Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )

        print("Loading 68-point landmark predictor...")
        self.landmark_predictor = dlib.shape_predictor(predictor_path)

        # Load multiple models for ensemble (3-model voting)
        print("Loading emotion models...")
        self.models = {
            'enet_b2': HSEmotionRecognizer(model_name='enet_b2_8'),
            'vgaf': HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf'),
            'afew': HSEmotionRecognizer(model_name='enet_b0_8_best_afew'),
        }
        print(f"Loaded {len(self.models)} models: {', '.join(self.models.keys())}")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE preprocessing."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _get_emotion_scores(self, face_rgb: np.ndarray, model_name: str) -> dict:
        """Get emotion scores from a single model."""
        try:
            model = self.models[model_name]
            emotion, scores = model.predict_emotions(face_rgb, logits=True)

            # Softmax
            exp_scores = np.exp(scores - np.max(scores))
            probs = (exp_scores / exp_scores.sum()) * 100

            return {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}
        except Exception:
            return None

    def _ensemble_predict(self, face_rgb: np.ndarray) -> Tuple[Optional[Dict], float, Dict]:
        """
        Weighted ensemble prediction from multiple models with majority voting override.
        Returns: (ensemble_scores, agreement_percentage, model_votes)
        """
        model_scores = {}
        model_votes = {}  # What each model predicted as top emotion

        for model_name in self.models:
            scores = self._get_emotion_scores(face_rgb, model_name)
            if scores:
                model_scores[model_name] = scores
                # Track what each model thinks is the top emotion
                top_emo = max(scores.items(), key=lambda x: x[1])[0]
                top_score = scores[top_emo]
                model_votes[model_name] = {'emotion': top_emo, 'confidence': round(top_score, 1)}

        if not model_scores:
            return None, 0.0, {}

        # Calculate weighted ensemble scores
        ensemble_scores = {}
        for emotion in EMOTION_LABELS:
            weighted_sum = 0.0
            weight_total = 0.0
            for model_name, scores in model_scores.items():
                weight = EMOTION_WEIGHTS[emotion].get(model_name, 0.33)
                weighted_sum += scores[emotion] * weight
                weight_total += weight
            ensemble_scores[emotion] = weighted_sum / weight_total if weight_total > 0 else 0

        # Get ensemble's top emotion
        ensemble_top = max(ensemble_scores.items(), key=lambda x: x[1])[0]

        # Calculate agreement and check for majority override
        top_emotions = [v['emotion'] for v in model_votes.values()]
        counts = Counter(top_emotions)
        most_common_emotion, most_common_count = counts.most_common(1)[0]

        # Agreement percentage
        agreement = (most_common_count / len(top_emotions)) * 100 if top_emotions else 100.0

        # MAJORITY OVERRIDE: If 2/3+ models agree on something different from ensemble
        if most_common_count >= 2 and most_common_emotion != ensemble_top:
            # Check if the majority emotion has reasonable scores
            majority_avg_score = np.mean([
                model_scores[m][most_common_emotion]
                for m in model_scores if model_votes[m]['emotion'] == most_common_emotion
            ])

            # Only override if majority has decent confidence
            if majority_avg_score > 30:
                # Boost the majority emotion
                boost = min(20, ensemble_scores[ensemble_top] * 0.4)
                ensemble_scores[most_common_emotion] += boost
                ensemble_scores[ensemble_top] -= boost * 0.5

        return ensemble_scores, agreement, model_votes

    def _analyze_mouth_opening(self, landmarks: np.ndarray) -> float:
        """Analyze mouth opening ratio from lip landmarks (62, 66, 48, 54)."""
        if landmarks is None or len(landmarks) < 68:
            return 0.5

        # Mouth landmarks
        top_lip_center = landmarks[62]      # Inner top lip center
        bottom_lip_center = landmarks[66]   # Inner bottom lip center
        mouth_left = landmarks[48]          # Left corner
        mouth_right = landmarks[54]         # Right corner

        # Calculate mouth opening (vertical distance between lips)
        mouth_open_dist = np.linalg.norm(bottom_lip_center - top_lip_center)

        # Calculate mouth width for normalization
        mouth_width = np.linalg.norm(mouth_right - mouth_left)

        if mouth_width == 0:
            return 0.5

        # Ratio: open distance / width (0.0 = closed, 0.5+ = very open)
        ratio = mouth_open_dist / mouth_width
        return ratio

    def _estimate_head_pose(self, landmarks: np.ndarray) -> str:
        """Estimate head yaw from eye centers and nose tip."""
        if landmarks is None or len(landmarks) < 68:
            return "facing camera"

        # Eye centers (average of eye points)
        left_eye = np.mean(landmarks[42:48], axis=0)   # Left eye: 42-47
        right_eye = np.mean(landmarks[36:42], axis=0)  # Right eye: 36-41
        nose_tip = landmarks[30]                        # Nose tip

        eye_center = (left_eye + right_eye) / 2
        eye_dist = np.linalg.norm(right_eye - left_eye)

        if eye_dist == 0:
            return "facing camera"

        # Nose offset from eye center (normalized)
        nose_offset = (nose_tip[0] - eye_center[0]) / eye_dist

        # Thresholds for pose detection
        if nose_offset < -0.15:
            return "turned left"
        elif nose_offset > 0.15:
            return "turned right"
        return "facing camera"

    def _analyze_mouth_corners(self, landmarks: np.ndarray) -> str:
        """Analyze if mouth corners are upturned, downturned, or neutral."""
        if landmarks is None or len(landmarks) < 68:
            return "neutral"

        # Key landmarks
        left_eye = np.mean(landmarks[42:48], axis=0)
        right_eye = np.mean(landmarks[36:42], axis=0)
        mouth_left = landmarks[48]         # Left mouth corner
        mouth_right = landmarks[54]        # Right mouth corner
        mouth_top = landmarks[51]          # Top center of outer lip
        mouth_bottom = landmarks[57]       # Bottom center of outer lip

        # Eye distance for normalization
        eye_dist = np.linalg.norm(right_eye - left_eye)
        if eye_dist == 0:
            return "neutral"

        # Calculate mouth center y-position (average of corners)
        corners_avg_y = (mouth_left[1] + mouth_right[1]) / 2

        # Calculate lip center y-position (center of lip)
        lip_center_y = (mouth_top[1] + mouth_bottom[1]) / 2

        # Compare corners to lip center
        # In a smile: corners are ABOVE (lower y value) the lip center
        # In sad/frown: corners are BELOW (higher y value) the lip center
        corner_vs_center = corners_avg_y - lip_center_y
        normalized_offset = corner_vs_center / eye_dist

        # Thresholds based on 68-point landmark geometry
        # Negative offset = corners above center = smile
        # Positive offset = corners below center = frown
        if normalized_offset < -0.03:
            return "upturned"
        elif normalized_offset > 0.05:
            return "downturned"
        else:
            return "neutral"

    def _refine_emotions(self, emotions: dict, landmarks: np.ndarray = None) -> dict:
        """
        Apply targeted refinements based on known confusion patterns.
        """
        refined = emotions.copy()

        sorted_emo = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        top_emotion = sorted_emo[0][0]
        top_score = sorted_emo[0][1]

        # Get second emotion
        second_emotion = sorted_emo[1][0] if len(sorted_emo) > 1 else None
        second_score = sorted_emo[1][1] if len(sorted_emo) > 1 else 0

        # FEAR vs SURPRISE refinement
        # Surprise: typically open mouth, raised eyebrows, relaxed
        # Fear: similar but more tension, eyes wider relative to mouth
        # BIAS: Surprise is MUCH more common in photos than genuine fear
        # NOTE: mouth_ratio is lip_gap/mouth_width (0=closed, 0.2+=slightly open, 0.4+=wide open)
        if top_emotion == 'fear' and emotions.get('surprise', 0) > 8:
            mouth_ratio = self._analyze_mouth_opening(landmarks) if landmarks is not None else 0.1

            # ANY mouth opening suggests surprise over fear
            if mouth_ratio > 0.12:  # Even slightly open mouth → surprise
                boost = min(40, emotions['fear'] * 0.8)
                refined['surprise'] = emotions['surprise'] + boost
                refined['fear'] = emotions['fear'] - boost * 0.9

            # Strong bias toward surprise - fear is rare in posed photos
            fear_surprise_diff = emotions['fear'] - emotions['surprise']
            if fear_surprise_diff < 35:  # Widened from 25
                # Always bias toward surprise when close
                bias = 20 if fear_surprise_diff < 15 else 15
                refined['surprise'] = refined.get('surprise', emotions['surprise']) + bias
                refined['fear'] = refined.get('fear', emotions['fear']) - bias * 0.7

        # SURPRISE detected but might be fear - be VERY conservative (fear is rare)
        if top_emotion == 'surprise' and emotions.get('fear', 0) > 35:
            mouth_ratio = self._analyze_mouth_opening(landmarks) if landmarks is not None else 0.1

            # Only flip to fear if mouth is CLEARLY closed AND fear score is very high
            if mouth_ratio < 0.05 and emotions['fear'] > 45:
                boost = min(10, emotions['surprise'] * 0.2)
                refined['fear'] = emotions['fear'] + boost
                refined['surprise'] = emotions['surprise'] - boost * 0.3

        # HAPPY with low confidence - boost if clearly smiling
        if top_emotion == 'happy' and top_score < 50:
            # If happy is top and second is far behind, boost confidence
            if second_score < top_score * 0.7:
                refined['happy'] = min(75, emotions['happy'] * 1.3)

        # ANGRY vs DISGUST refinement
        # Disgust: upper lip raised, mouth typically compressed
        # Angry: normal mouth position, more symmetric face
        # NOTE: mouth_ratio is now lip_gap/mouth_width (0=closed, 0.3+=open)
        if top_emotion in ['angry', 'disgust'] and landmarks is not None:
            mouth_ratio = self._analyze_mouth_opening(landmarks)

            # Disgust typically has closed/compressed mouth (upper lip raised)
            if top_emotion == 'angry' and emotions.get('disgust', 0) > 15:
                if mouth_ratio < 0.12:  # Compressed/closed mouth suggests disgust
                    boost = min(15, emotions['angry'] * 0.3)
                    refined['disgust'] = emotions['disgust'] + boost
                    refined['angry'] = emotions['angry'] - boost * 0.5

            if top_emotion == 'disgust' and emotions.get('angry', 0) > 15:
                if mouth_ratio > 0.18:  # More open mouth suggests angry over disgust
                    boost = min(15, emotions['disgust'] * 0.3)
                    refined['angry'] = emotions['angry'] + boost
                    refined['disgust'] = emotions['disgust'] - boost * 0.5

        # SAD vs ANGRY refinement (enhanced with intensity + mouth corners)
        # Sad is often confused with Angry because both have furrowed brows
        # Key differences:
        # - Angry has tension/intensity, tighter face muscles
        # - Angry has neutral/tense mouth, Sad has clearly downturned mouth
        # BIAS: In posed photos, angry is more common than genuine sadness
        if top_emotion == 'sad' and emotions.get('angry', 0) > 12:
            sad_angry_diff = emotions['sad'] - emotions['angry']
            mouth_corners = self._analyze_mouth_corners(landmarks) if landmarks is not None else "neutral"

            # INTENSITY CHECK: Low-confidence sad → probably angry
            # Angry expressions are typically more intense/confident looking
            if top_score < 60 and sad_angry_diff < 20:
                boost = min(20, emotions['sad'] * 0.4)
                refined['angry'] = emotions['angry'] + boost
                refined['sad'] = emotions['sad'] - boost * 0.7

            # MOUTH CORNER CHECK: If mouth is NOT clearly downturned → probably angry
            # Only keep as sad if mouth is clearly drooping
            if mouth_corners != "downturned" and sad_angry_diff < 25:
                boost = min(18, emotions['sad'] * 0.35)
                refined['angry'] = emotions['angry'] + boost
                refined['sad'] = emotions['sad'] - boost * 0.6

            # ANGRY CLUSTER CHECK: If disgust is also present, suggests anger family
            if emotions.get('disgust', 0) > 10 and sad_angry_diff < 20:
                boost = min(12, emotions['sad'] * 0.25)
                refined['angry'] = emotions['angry'] + boost
                refined['sad'] = emotions['sad'] - boost * 0.5

        # ANGRY mistaken as SAD - only flip if mouth is CLEARLY downturned
        if top_emotion == 'angry' and emotions.get('sad', 0) > 25:
            mouth_corners = self._analyze_mouth_corners(landmarks) if landmarks is not None else "neutral"

            # Only flip to sad if mouth is clearly downturned AND scores are close
            if mouth_corners == "downturned":
                angry_sad_diff = emotions['angry'] - emotions['sad']
                if angry_sad_diff < 10:  # Tightened from 15
                    boost = min(8, emotions['angry'] * 0.15)
                    refined['sad'] = emotions['sad'] + boost
                    refined['angry'] = emotions['angry'] - boost * 0.3

        # Normalize to 100%
        total = sum(refined.values())
        if total > 0:
            refined = {k: (v / total) * 100 for k, v in refined.items()}

        return refined

    def analyze_image(self, image: np.ndarray) -> List[Dict]:
        """Analyze emotions in all faces using dlib."""
        processed = self._preprocess_image(image)
        rgb_image = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # Detect faces with dlib (1 = upsample once for better detection)
        face_rects = self.face_detector(gray, 1)

        if len(face_rects) == 0:
            return []

        # Collect valid faces with landmarks
        valid_faces = []
        h, w = image.shape[:2]

        for face_rect in face_rects:
            x1 = max(0, face_rect.left())
            y1 = max(0, face_rect.top())
            x2 = min(w, face_rect.right())
            y2 = min(h, face_rect.bottom())

            if x2 <= x1 or y2 <= y1:
                continue

            # Get 68-point landmarks
            shape = self.landmark_predictor(gray, face_rect)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])

            valid_faces.append({
                'box': (x1, y1, x2, y2),
                'landmarks': landmarks
            })

        # Sort faces by position: top-to-bottom, then left-to-right
        # Use row-based sorting: group by y-position (with tolerance), then sort by x
        def get_sort_key(face):
            x1, y1, x2, y2 = face['box']
            # Round y to nearest 50px to group faces in same "row"
            row = y1 // 80
            return (row, x1)

        valid_faces.sort(key=get_sort_key)

        analyzed_faces = []

        for idx, face_info in enumerate(valid_faces):
            x1, y1, x2, y2 = face_info['box']
            face_landmarks = face_info['landmarks']

            face_rgb = rgb_image[y1:y2, x1:x2]
            if face_rgb.size == 0:
                continue

            try:
                # Ensemble prediction
                if self.use_ensemble:
                    emotions, agreement, model_votes = self._ensemble_predict(face_rgb)
                else:
                    emotions = self._get_emotion_scores(face_rgb, 'enet_b2')
                    agreement = 100.0
                    model_votes = {'enet_b2': {'emotion': max(emotions.items(), key=lambda x: x[1])[0],
                                               'confidence': max(emotions.values())}}

                if emotions is None:
                    continue

                # Apply refinements
                emotions = self._refine_emotions(emotions, face_landmarks)

                # Estimate head pose
                head_pose = self._estimate_head_pose(face_landmarks)

                face_data = self._process_result(
                    emotions=emotions,
                    region={'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1},
                    face_number=len(analyzed_faces) + 1,
                    agreement=agreement,
                    head_pose=head_pose,
                    model_votes=model_votes
                )
                analyzed_faces.append(face_data)

            except Exception as e:
                print(f"Error analyzing face {idx + 1}: {e}")
                continue

        return analyzed_faces

    def _process_result(self, emotions: Dict, region: Dict, face_number: int,
                        agreement: float = 100.0, head_pose: str = "facing camera",
                        model_votes: Optional[Dict] = None) -> Dict:
        """Process emotion results into standard format."""
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

        dominant_emotion = sorted_emotions[0][0]
        dominant_score = sorted_emotions[0][1]
        secondary_emotion = sorted_emotions[1][0] if len(sorted_emotions) > 1 else None
        secondary_score = sorted_emotions[1][1] if len(sorted_emotions) > 1 else 0

        if dominant_score >= HIGH_CONFIDENCE:
            confidence_level = 'high'
        elif dominant_score >= MEDIUM_CONFIDENCE:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'

        experience = EMOTION_DESCRIPTIONS.get(dominant_emotion, dominant_emotion)
        is_ambiguous = (dominant_score - secondary_score) < 12 and secondary_score > 20

        return {
            'face_number': face_number,
            'bounding_box': {
                'x': region['x'],
                'y': region['y'],
                'width': region['w'],
                'height': region['h']
            },
            'dominant_emotion': dominant_emotion,
            'confidence': round(dominant_score, 1),
            'confidence_level': confidence_level,
            'secondary_emotion': secondary_emotion,
            'secondary_confidence': round(secondary_score, 1),
            'is_ambiguous': is_ambiguous,
            'experience': experience,
            'description': f'Experiencing {experience}.',
            'all_emotions': {k: round(v, 1) for k, v in emotions.items()},
            'backends_used': len(self.models) if self.use_ensemble else 1,
            'backend_agreement': round(agreement, 1),
            'head_pose': head_pose,
            'model_votes': model_votes or {}
        }

    def draw_emotions(self, image: np.ndarray, analyses: List[Dict],
                      font_scale: float = 0.7, thickness: int = 2) -> np.ndarray:
        result = image.copy()
        for analysis in analyses:
            box = analysis['bounding_box']
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), thickness)
            label = f"{analysis['dominant_emotion']}: {analysis['confidence']:.1f}%"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(result, (x, y - text_h - 10), (x + text_w + 5, y), (0, 255, 0), -1)
            cv2.putText(result, label, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        return result


def format_analysis_report(analyses: List[Dict], image_name: str) -> str:
    lines = ["=" * 60, f"FACIAL EXPRESSION ANALYSIS: {image_name}", "=" * 60]
    if not analyses:
        lines.append("\nNo faces detected in this image.")
        return "\n".join(lines)
    lines.append(f"\nDetected {len(analyses)} face(s) in the image.\n")
    for analysis in analyses:
        lines.extend(["-" * 40, f"FACE #{analysis['face_number']}", "-" * 40])
        lines.append(f"\nPrimary Emotion: {analysis['dominant_emotion'].upper()}")
        lines.append(f"Confidence: {analysis['confidence']}% ({analysis['confidence_level']})")
        if analysis['is_ambiguous']:
            lines.append(f"AMBIGUOUS - Also likely: {analysis['secondary_emotion']} ({analysis['secondary_confidence']}%)")
        lines.append(f"\nExperience: This person is experiencing {analysis['experience']}.")
        lines.append("\nFull emotion breakdown:")
        for emotion, score in sorted(analysis['all_emotions'].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
            lines.append(f"  {emotion:8}: [{bar}] {score}%")
        lines.append("")
    return "\n".join(lines)
