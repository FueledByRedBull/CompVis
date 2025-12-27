"""
Facial Expression Analyzer - Modern GUI Application
Built with CustomTkinter for a sleek, modern look.
Features face enumeration, emotion analysis, and webcam support.
"""

from typing import List, Dict, Optional

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
from pathlib import Path
import threading
import time
import tkinter as tk

from emotion_analyzer import EmotionAnalyzer

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

# Color scheme for emotions (RGB)
EMOTION_COLORS = {
    'angry': (255, 82, 82),
    'disgust': (156, 39, 176),
    'fear': (103, 58, 183),
    'happy': (76, 175, 80),
    'sad': (33, 150, 243),
    'surprise': (255, 193, 7),
    'neutral': (158, 158, 158),
    'contempt': (255, 112, 67)
}

# Confidence level colors
CONFIDENCE_COLORS = {
    'high': (76, 175, 80),
    'medium': (255, 193, 7),
    'low': (255, 82, 82)
}

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class FacialExpressionGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Facial Expression Analyzer")
        self.root.geometry("1600x950")

        # State
        self.image_files = []
        self.current_index = 0
        self.current_image = None
        self.current_analyses = []
        self.analyzer = None
        self.photo = None

        # Webcam state
        self.webcam_active = False
        self.webcam_capture = None
        self.webcam_job = None

        # Build UI
        self.create_widgets()

        # Initialize analyzer
        self.status_var.set("Initializing 3-model ensemble...")
        threading.Thread(target=self.init_analyzer, daemon=True).start()

    def create_widgets(self):
        # Main container
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # ===== TOP TOOLBAR =====
        toolbar = ctk.CTkFrame(self.root, height=60, corner_radius=0)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 5))
        toolbar.grid_columnconfigure(3, weight=1)

        # Buttons
        self.folder_btn = ctk.CTkButton(
            toolbar, text="📁 Select Folder", command=self.select_folder,
            width=140, height=36, font=ctk.CTkFont(size=13, weight="bold")
        )
        self.folder_btn.grid(row=0, column=0, padx=(10, 5), pady=12)

        self.webcam_btn = ctk.CTkButton(
            toolbar, text="📷 Start Webcam", command=self.toggle_webcam,
            width=140, height=36, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2d7d46", hover_color="#236b38"
        )
        self.webcam_btn.grid(row=0, column=1, padx=5, pady=12)

        # Folder label
        self.folder_label = ctk.CTkLabel(
            toolbar, text="No folder selected",
            font=ctk.CTkFont(size=12), text_color="#888888"
        )
        self.folder_label.grid(row=0, column=2, padx=15, pady=12)

        # Model info
        model_label = ctk.CTkLabel(
            toolbar, text="3-Model Ensemble + dlib 68-Point Landmarks",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#4CAF50"
        )
        model_label.grid(row=0, column=3, padx=10, pady=12)

        # Progress
        self.progress_label = ctk.CTkLabel(
            toolbar, text="", font=ctk.CTkFont(size=12), text_color="#888888"
        )
        self.progress_label.grid(row=0, column=4, padx=10, pady=12)

        # ===== LEFT PANEL - IMAGE =====
        left_frame = ctk.CTkFrame(self.root, corner_radius=10)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        # Canvas for image
        self.canvas = tk.Canvas(left_frame, bg='#1a1a1a', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Navigation frame
        nav_frame = ctk.CTkFrame(left_frame, height=50, fg_color="transparent")
        nav_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        nav_frame.grid_columnconfigure(1, weight=1)

        self.prev_btn = ctk.CTkButton(
            nav_frame, text="◀ Previous", command=self.prev_image,
            width=120, height=35, state="disabled",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.prev_btn.grid(row=0, column=0, padx=5)

        self.image_counter = ctk.CTkLabel(
            nav_frame, text="0 / 0", font=ctk.CTkFont(size=14, weight="bold")
        )
        self.image_counter.grid(row=0, column=1)

        self.next_btn = ctk.CTkButton(
            nav_frame, text="Next ▶", command=self.next_image,
            width=120, height=35, state="disabled",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.next_btn.grid(row=0, column=2, padx=5)

        # ===== RIGHT PANEL - RESULTS =====
        right_frame = ctk.CTkFrame(self.root, width=450, corner_radius=10)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=5)
        right_frame.grid_rowconfigure(2, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_propagate(False)

        # Title
        title_label = ctk.CTkLabel(
            right_frame, text="📊 ANALYSIS RESULTS",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, pady=(15, 5))

        # Legend
        legend_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        legend_frame.grid(row=1, column=0, pady=(0, 10))

        ctk.CTkLabel(legend_frame, text="Confidence:", font=ctk.CTkFont(size=11)).pack(side="left", padx=5)

        for level, color in [('HIGH', '#4CAF50'), ('MEDIUM', '#FFC107'), ('LOW', '#FF5252')]:
            badge = ctk.CTkLabel(
                legend_frame, text=f" {level} ",
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color=color, corner_radius=4, text_color="white"
            )
            badge.pack(side="left", padx=2)

        # Scrollable results
        self.results_scroll = ctk.CTkScrollableFrame(
            right_frame, fg_color="#1a1a1a", corner_radius=8
        )
        self.results_scroll.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Placeholder
        self.placeholder = ctk.CTkLabel(
            self.results_scroll, text="Select a folder or start webcam\nto begin analysis",
            font=ctk.CTkFont(size=13), text_color="#666666"
        )
        self.placeholder.pack(pady=50)

        # ===== STATUS BAR =====
        status_frame = ctk.CTkFrame(self.root, height=35, corner_radius=0, fg_color="#1a1a1a")
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(5, 10))

        self.status_var = ctk.StringVar(value="Ready")
        self.status_label = ctk.CTkLabel(
            status_frame, textvariable=self.status_var,
            font=ctk.CTkFont(size=11), text_color="#888888"
        )
        self.status_label.pack(side="left", padx=10, pady=5)

        self.processing_label = ctk.CTkLabel(
            status_frame, text="",
            font=ctk.CTkFont(size=11), text_color="#4CAF50"
        )
        self.processing_label.pack(side="right", padx=10, pady=5)

    def init_analyzer(self):
        """Initialize emotion analyzer in background."""
        try:
            self.analyzer = EmotionAnalyzer()
            self.root.after(0, lambda: self.status_var.set("Ready [3 Models + dlib 68-Point Landmarks]"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to initialize: {e}"))

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            return

        folder_path = Path(folder)
        self.image_files = sorted([
            f for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

        if not self.image_files:
            messagebox.showwarning("No Images", "No supported images found")
            return

        self.folder_label.configure(text=f"📁 {folder_path.name}")
        self.current_index = 0
        self.update_navigation()
        self.load_and_analyze_current()

    def update_navigation(self):
        total = len(self.image_files)
        current = self.current_index + 1 if total > 0 else 0

        self.image_counter.configure(text=f"{current} / {total}")
        self.prev_btn.configure(state="normal" if self.current_index > 0 else "disabled")
        self.next_btn.configure(state="normal" if self.current_index < total - 1 else "disabled")
        self.progress_label.configure(text=f"{current}/{total} images")

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_navigation()
            self.load_and_analyze_current()

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.update_navigation()
            self.load_and_analyze_current()

    def load_and_analyze_current(self):
        if not self.image_files or self.analyzer is None:
            return

        image_path = self.image_files[self.current_index]
        self.status_var.set(f"Processing: {image_path.name}...")
        self.processing_label.configure(text="⏳ Analyzing...")

        self.current_image = cv2.imread(str(image_path))
        if self.current_image is None:
            self.status_var.set(f"Error loading: {image_path.name}")
            return

        self.display_image(self.current_image, [])
        self.root.update()

        threading.Thread(target=self.analyze_current, daemon=True).start()

    def analyze_current(self):
        if self.current_image is None or self.analyzer is None:
            return

        try:
            analyses = self.analyzer.analyze_image(self.current_image)
            self.root.after(0, lambda: self.show_results(analyses))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {e}"))

    def show_results(self, analyses):
        self.current_analyses = analyses
        self.display_image(self.current_image, analyses)
        self.display_results(analyses)

        face_count = len(analyses)
        image_name = self.image_files[self.current_index].name
        self.status_var.set(f"{image_name} - {face_count} face(s) detected")
        self.processing_label.configure(text="✓ Complete")

    def display_image(self, cv_image, analyses):
        """Display image with face overlays and numbers."""
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        if analyses:
            pil_image = self.draw_overlays(pil_image, analyses)

        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            img_width, img_height = pil_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")

        x = (self.canvas.winfo_width() - pil_image.width) // 2
        y = (self.canvas.winfo_height() - pil_image.height) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)

    def draw_overlays(self, image: Image.Image, analyses: List[Dict]) -> Image.Image:
        """Draw face boxes with modern effects: Ghost, Glassmorphism, Bracket Corners."""
        # Create overlay layer for transparency effects
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, 'RGBA')

        try:
            font_large = ImageFont.truetype("segoeui.ttf", 18)
            font_small = ImageFont.truetype("segoeui.ttf", 12)
            font_number = ImageFont.truetype("segoeuib.ttf", 24)
        except OSError:
            try:
                font_large = ImageFont.truetype("arial.ttf", 18)
                font_small = ImageFont.truetype("arial.ttf", 12)
                font_number = ImageFont.truetype("arialbd.ttf", 24)
            except OSError:
                font_large = ImageFont.load_default()
                font_small = font_large
                font_number = font_large

        for analysis in analyses:
            box = analysis['bounding_box']
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            emotion = analysis['dominant_emotion']
            confidence = analysis['confidence']
            confidence_level = analysis.get('confidence_level', 'medium')
            face_number = analysis['face_number']
            is_ambiguous = analysis.get('is_ambiguous', False)

            # ===== GHOST EFFECT: Transparency based on confidence =====
            # High confidence = fully visible, Low confidence = more transparent
            if confidence >= 60:
                alpha = 255  # Fully opaque
            elif confidence >= 40:
                alpha = 200  # Slightly transparent
            else:
                alpha = 140  # Ghost effect for low confidence

            # Color based on emotion (high confidence) or warning colors
            if confidence_level == 'high':
                box_color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            elif confidence_level == 'medium':
                box_color = (255, 193, 7)
            else:
                box_color = (255, 82, 82)

            # ===== BRACKET CORNERS (Sci-Fi/Tech Look) =====
            corner_length = 20
            bracket_thickness = 3
            gap = 8  # Gap between bracket and face box

            # Top-left bracket
            draw.line(
                [(x - gap, y - gap + corner_length), (x - gap, y - gap), (x - gap + corner_length, y - gap)],
                fill=(*box_color, alpha), width=bracket_thickness
            )
            # Top-right bracket
            draw.line(
                [(x + w + gap - corner_length, y - gap), (x + w + gap, y - gap),
                 (x + w + gap, y - gap + corner_length)],
                fill=(*box_color, alpha), width=bracket_thickness
            )
            # Bottom-left bracket
            draw.line(
                [(x - gap, y + h + gap - corner_length), (x - gap, y + h + gap),
                 (x - gap + corner_length, y + h + gap)],
                fill=(*box_color, alpha), width=bracket_thickness
            )
            # Bottom-right bracket
            draw.line(
                [(x + w + gap - corner_length, y + h + gap), (x + w + gap, y + h + gap),
                 (x + w + gap, y + h + gap - corner_length)],
                fill=(*box_color, alpha), width=bracket_thickness
            )

            # Subtle inner box outline
            draw.rectangle([x, y, x + w, y + h], outline=(*box_color, alpha // 2), width=1)

            # ===== FACE NUMBER BADGE =====
            number_text = str(face_number)
            number_bbox = draw.textbbox((0, 0), number_text, font=font_number)
            number_width = number_bbox[2] - number_bbox[0] + 16
            number_height = number_bbox[3] - number_bbox[1] + 8

            circle_x = x - gap - 5
            circle_y = y - gap - 5
            # Glassmorphism badge background
            draw.ellipse(
                [circle_x, circle_y, circle_x + number_width, circle_y + number_height],
                fill=(30, 30, 30, 200), outline=(*box_color, alpha), width=2
            )
            draw.text(
                (circle_x + 8, circle_y + 2), number_text,
                fill=(255, 255, 255, alpha), font=font_number
            )

            # Build label
            label = f"{emotion.upper()} {confidence:.1f}%"
            if is_ambiguous and analysis.get('secondary_emotion'):
                secondary_label = f"or {analysis['secondary_emotion']} {analysis['secondary_confidence']:.1f}%"
            else:
                secondary_label = None

            # Text size
            text_bbox = draw.textbbox((0, 0), label, font=font_large)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            padding = 10
            label_height = text_height + padding * 2

            if secondary_label:
                sec_bbox = draw.textbbox((0, 0), secondary_label, font=font_small)
                text_width = max(text_width, sec_bbox[2] - sec_bbox[0])
                label_height += 18

            label_width = text_width + padding * 2

            # ===== GLASSMORPHISM LABEL (Modern Frosted Glass Look) =====
            label_x = x
            label_y = y - label_height - gap - 5

            # Dark frosted glass background with border
            glass_bg = (20, 20, 20, int(180 * alpha / 255))  # Dark semi-transparent
            border_color = (*box_color, alpha)

            # Rounded rectangle effect (draw multiple overlapping shapes)
            radius = 8
            # Main glass panel
            draw.rounded_rectangle(
                [label_x, label_y, label_x + label_width, label_y + label_height],
                radius=radius,
                fill=glass_bg,
                outline=border_color,
                width=2
            )

            # Subtle highlight line at top (glass reflection effect)
            draw.line(
                [(label_x + radius, label_y + 2), (label_x + label_width - radius, label_y + 2)],
                fill=(255, 255, 255, 40),
                width=1
            )

            # Color accent bar on left side
            draw.rectangle(
                [label_x, label_y + radius, label_x + 4, label_y + label_height - radius],
                fill=(*box_color, alpha)
            )

            # Main text with slight shadow
            text_x = label_x + padding + 4
            text_y = label_y + padding - 2
            # Shadow
            draw.text((text_x + 1, text_y + 1), label,
                     fill=(0, 0, 0, 100), font=font_large)
            # Main text
            draw.text((text_x, text_y), label,
                     fill=(255, 255, 255, alpha), font=font_large)

            # Secondary text
            if secondary_label:
                draw.text((text_x, text_y + text_height + 4), secondary_label,
                         fill=(255, 255, 200, int(alpha * 0.8)), font=font_small)

            # ===== CONFIDENCE INDICATOR (Small dot) =====
            indicator_color = CONFIDENCE_COLORS.get(confidence_level, (158, 158, 158))
            dot_x = label_x + label_width - 14
            dot_y = label_y + 8
            draw.ellipse([dot_x, dot_y, dot_x + 8, dot_y + 8],
                        fill=(*indicator_color, alpha))

            # Scanning line effect for high-tech look (optional decorative element)
            if confidence_level == 'high':
                scan_y = y + h // 2
                draw.line([(x + 5, scan_y), (x + w - 5, scan_y)],
                         fill=(*box_color, 60), width=1)

        # Composite overlay onto original image
        image = image.convert('RGBA')
        image = Image.alpha_composite(image, overlay)

        return image

    def display_results(self, analyses: List[Dict]) -> None:
        """Display results in the side panel."""
        # Clear previous
        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        if not analyses:
            placeholder = ctk.CTkLabel(
                self.results_scroll, text="No faces detected",
                font=ctk.CTkFont(size=13), text_color="#666666"
            )
            placeholder.pack(pady=50)
            return

        for analysis in analyses:
            self.create_face_card(analysis)

    def create_face_card(self, analysis):
        """Create a modern card for face results."""
        emotion = analysis['dominant_emotion']
        confidence_level = analysis.get('confidence_level', 'medium')
        face_number = analysis['face_number']
        is_ambiguous = analysis.get('is_ambiguous', False)

        # Color
        if confidence_level == 'high':
            color = EMOTION_COLORS.get(emotion, (158, 158, 158))
        elif confidence_level == 'medium':
            color = (255, 193, 7)
        else:
            color = (255, 82, 82)

        hex_color = '#{:02x}{:02x}{:02x}'.format(*color)

        # Card frame
        card = ctk.CTkFrame(self.results_scroll, corner_radius=10, border_width=2, border_color=hex_color)
        card.pack(fill="x", pady=5, padx=5)

        # Header
        header = ctk.CTkFrame(card, fg_color=hex_color, corner_radius=8)
        header.pack(fill="x", padx=3, pady=3)

        # Face number badge
        number_badge = ctk.CTkLabel(
            header, text=f"  #{face_number}  ",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#000000", corner_radius=6, text_color="white"
        )
        number_badge.pack(side="left", padx=8, pady=8)

        # Emotion and confidence
        ctk.CTkLabel(
            header, text=f"{emotion.upper()} {analysis['confidence']:.1f}%",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="white"
        ).pack(side="left", padx=5, pady=8)

        # Confidence badge
        conf_colors = {'high': '#2d7d46', 'medium': '#b8860b', 'low': '#a83232'}
        ctk.CTkLabel(
            header, text=f" {confidence_level.upper()} ",
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=conf_colors[confidence_level], corner_radius=4, text_color="white"
        ).pack(side="right", padx=8, pady=8)

        # Content
        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(fill="x", padx=10, pady=10)

        # Ambiguous warning
        if is_ambiguous:
            warning = ctk.CTkLabel(
                content,
                text=f"⚠️ Could also be {analysis['secondary_emotion'].upper()} ({analysis['secondary_confidence']:.1f}%)",
                font=ctk.CTkFont(size=11), text_color="#FFD700",
                fg_color="#3d3d00", corner_radius=4
            )
            warning.pack(fill="x", pady=(0, 8))

        # Experience
        ctk.CTkLabel(
            content, text=f"Experiencing: {analysis['experience']}",
            font=ctk.CTkFont(size=11), text_color="#cccccc", wraplength=350,
            justify="left", anchor="w"
        ).pack(anchor="w")

        # Backend info
        backends_used = analysis.get('backends_used', 1)
        agreement = analysis.get('backend_agreement', 100)
        head_pose = analysis.get('head_pose', 'facing camera')
        model_votes = analysis.get('model_votes', {})

        if backends_used > 1:
            info_text = f"Models: {backends_used} | Agreement: {agreement:.0f}% | Pose: {head_pose}"
            ctk.CTkLabel(
                content, text=info_text,
                font=ctk.CTkFont(size=10), text_color="#888888"
            ).pack(anchor="w", pady=(5, 0))

            # Model votes
            if model_votes:
                votes_frame = ctk.CTkFrame(content, fg_color="#252525", corner_radius=6)
                votes_frame.pack(fill="x", pady=(5, 0))

                ctk.CTkLabel(
                    votes_frame, text="Model Votes:",
                    font=ctk.CTkFont(size=10, weight="bold"), text_color="#aaaaaa"
                ).pack(anchor="w", padx=8, pady=(5, 2))

                for model_name, vote in model_votes.items():
                    vote_emo = vote.get('emotion', '?')
                    vote_conf = vote.get('confidence', 0)
                    vote_color = EMOTION_COLORS.get(vote_emo, (158, 158, 158))
                    vote_hex = '#{:02x}{:02x}{:02x}'.format(*vote_color)

                    ctk.CTkLabel(
                        votes_frame, text=f"  {model_name}: {vote_emo.upper()} {vote_conf:.0f}%",
                        font=ctk.CTkFont(family="Consolas", size=10), text_color=vote_hex
                    ).pack(anchor="w", padx=8)

                # Add bottom padding
                ctk.CTkLabel(votes_frame, text="", height=5).pack()

        # Emotion bars section
        bars_label = ctk.CTkLabel(
            content, text="Emotion Breakdown:",
            font=ctk.CTkFont(size=10, weight="bold"), text_color="#aaaaaa"
        )
        bars_label.pack(anchor="w", pady=(10, 5))

        for emo, score in sorted(analysis['all_emotions'].items(), key=lambda x: x[1], reverse=True):
            self.create_emotion_bar(content, emo, score)

    def create_emotion_bar(self, parent, emotion, score):
        """Create a modern emotion bar."""
        bar_frame = ctk.CTkFrame(parent, fg_color="transparent", height=22)
        bar_frame.pack(fill="x", pady=1)
        bar_frame.pack_propagate(False)

        color = EMOTION_COLORS.get(emotion, (158, 158, 158))
        hex_color = '#{:02x}{:02x}{:02x}'.format(*color)

        # Label
        ctk.CTkLabel(
            bar_frame, text=f"{emotion:8}",
            font=ctk.CTkFont(family="Consolas", size=10), text_color="#aaaaaa",
            width=70, anchor="w"
        ).pack(side="left")

        # Bar background
        bar_bg = ctk.CTkFrame(bar_frame, fg_color="#404040", height=14, width=150, corner_radius=3)
        bar_bg.pack(side="left", padx=(5, 5))
        bar_bg.pack_propagate(False)

        # Bar fill
        fill_width = max(1, int(score * 1.5))
        if fill_width > 0:
            bar_fill = ctk.CTkFrame(bar_bg, fg_color=hex_color, height=14, width=fill_width, corner_radius=3)
            bar_fill.place(x=0, y=0)

        # Score
        ctk.CTkLabel(
            bar_frame, text=f"{score:5.1f}%",
            font=ctk.CTkFont(family="Consolas", size=10), text_color="#ffffff"
        ).pack(side="left")

    def toggle_webcam(self):
        if self.webcam_active:
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        if self.analyzer is None:
            messagebox.showwarning("Not Ready", "Please wait for analyzer to initialize.")
            return

        try:
            self.webcam_capture = cv2.VideoCapture(0)
            if not self.webcam_capture.isOpened():
                messagebox.showerror("Error", "Could not open webcam.")
                return

            self.webcam_active = True
            self.webcam_btn.configure(text="⏹ Stop Webcam", fg_color="#a83232", hover_color="#8b2929")
            self.folder_btn.configure(state="disabled")
            self.prev_btn.configure(state="disabled")
            self.next_btn.configure(state="disabled")
            self.status_var.set("Webcam active - analyzing live feed...")
            self.folder_label.configure(text="📷 LIVE WEBCAM")

            self.process_webcam_frame()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start webcam: {e}")
            self.stop_webcam()

    def stop_webcam(self):
        self.webcam_active = False

        if self.webcam_job:
            self.root.after_cancel(self.webcam_job)
            self.webcam_job = None

        if self.webcam_capture:
            self.webcam_capture.release()
            self.webcam_capture = None

        self.webcam_btn.configure(text="📷 Start Webcam", fg_color="#2d7d46", hover_color="#236b38")
        self.folder_btn.configure(state="normal")
        self.status_var.set("Ready [3 Models + dlib 68-Point Landmarks]")
        self.folder_label.configure(text="No folder selected")

        self.canvas.delete("all")

        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        ctk.CTkLabel(
            self.results_scroll, text="Select a folder or start webcam\nto begin analysis",
            font=ctk.CTkFont(size=13), text_color="#666666"
        ).pack(pady=50)

        if self.image_files:
            self.update_navigation()

    def process_webcam_frame(self):
        if not self.webcam_active or self.webcam_capture is None:
            return

        start_time = time.time()

        ret, frame = self.webcam_capture.read()
        if not ret:
            self.status_var.set("Webcam read error")
            self.webcam_job = self.root.after(200, self.process_webcam_frame)
            return

        self.current_image = frame

        try:
            analyses = self.analyzer.analyze_image(frame)
            self.current_analyses = analyses
            self.display_image(frame, analyses)
            self.display_results(analyses)
            self.status_var.set(f"Webcam: {len(analyses)} face(s) detected")
        except Exception as e:
            self.status_var.set(f"Error: {e}")

        # Adaptive delay: maintain ~200ms target, minimum 50ms
        elapsed_ms = (time.time() - start_time) * 1000
        delay = max(50, int(200 - elapsed_ms))
        self.webcam_job = self.root.after(delay, self.process_webcam_frame)

    def run(self):
        self.root.mainloop()


def main():
    app = FacialExpressionGUI()
    app.run()


if __name__ == '__main__':
    main()
