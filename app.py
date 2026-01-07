import os
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import time
import sys

# =========================================================================
# 模块 1: AI 智能引擎 (通用版：支持检测、分割、姿态)
# =========================================================================

class YoloDetector:
    def __init__(self):
        self.models = {} 
        self.device = 'cpu'
        self.gpu_info = "正在初始化..."
        
        self.model_dir = "models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self._check_environment_immediate()

    def _check_environment_immediate(self):
        try:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                self.device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                self.gpu_info = f"🚀 {gpu_name} | CUDA Ready"
            else:
                self.device = 'cpu'
                self.gpu_info = "🐢 CPU 模式 (未检测到NVIDIA显卡)"
        except Exception as e:
            self.gpu_info = f"⚠️ 环境异常: {str(e)}"

    def load_models(self, target_model_names):
        try:
            from ultralytics import YOLO
            current_keys = set(self.models.keys())
            target_keys = set(target_model_names)
            
            for name in (current_keys - target_keys):
                del self.models[name]
                
            for name in (target_keys - current_keys):
                path = os.path.join(self.model_dir, name)
                if not os.path.exists(path): path = name
                if os.path.exists(path):
                    model = YOLO(path)
                    self.models[name] = model
                else:
                    print(f"❌ 找不到模型: {name}")

            if self.models:
                return True, f"{self.gpu_info} | 加载: {len(self.models)}"
            else:
                return False, "未加载任何模型"
        except Exception as e:
            return False, str(e)

    def process_frame(self, frame, conf_threshold=0.15, draw=True):
        if not self.models or frame is None:
            return False, frame

        has_target = False
        annotated_frame = frame.copy()

        for name, model in self.models.items():
            # 运行推理
            results = model(frame, device=self.device, verbose=False, conf=conf_threshold)
            
            if results:
                r = results[0]
                # [关键逻辑更新] 兼容性判断：
                # 1. boxes: 目标检测 (如手套、安全帽)
                # 2. keypoints: 姿态识别 (骨架)
                # 3. masks: 实例分割
                if (len(r.boxes) > 0 or 
                   (r.keypoints is not None and len(r.keypoints.conf) > 0) or
                   (r.masks is not None)):
                    
                    has_target = True
                    
                    if draw:
                        # plot() 会自动根据模型类型画框或画骨架
                        # img=... 参数实现多模型叠加绘制
                        annotated_frame = r.plot(img=annotated_frame)

        return has_target, annotated_frame

# =========================================================================
# 模块 2: 核心逻辑层
# =========================================================================

class FileManager:
    @staticmethod
    def scan_directory(target_dir, file_extension=".mp4"):
        if not target_dir: return
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.lower().endswith(file_extension):
                    yield root, file

    @staticmethod
    def delete_files(file_paths):
        success_count = 0
        errors = []
        for filepath in file_paths:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    success_count += 1
            except Exception as e:
                errors.append(f"{os.path.basename(filepath)}: {str(e)}")
        return success_count, errors

    @staticmethod
    def delete_folders(folder_paths):
        success_count = 0
        errors = []
        for folder_path in folder_paths:
            try:
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                    success_count += 1
            except Exception as e:
                errors.append(f"{os.path.basename(folder_path)}: {str(e)}")
        return success_count, errors

class VideoProcessor:
    def __init__(self, detector):
        self.detector = detector

    def extract_preview_data(self, filepath, count, target_width, ai_conf, draw_labels):
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened(): return [], 0.0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        indices = []
        for i in range(count):
            if count > 1: idx = int(total_frames * i / (count - 1))
            else: idx = total_frames // 2
            indices.append(min(idx, total_frames - 1))

        frames_data = []
        target_detected_count = 0

        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue

            # 调用处理
            has_target, annotated_frame = self.detector.process_frame(frame, conf_threshold=ai_conf, draw=draw_labels)
            if has_target: target_detected_count += 1
            
            # 缩放处理
            h, w = annotated_frame.shape[:2]
            scale = 800 / w if w > 800 else 1
            if scale != 1:
                annotated_frame = cv2.resize(annotated_frame, (int(w*scale), int(h*scale)))
            
            img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            time_sec = idx / fps if fps else 0
            time_str = f"{int(time_sec//60):02d}:{int(time_sec%60):02d}"
            
            frames_data.append({
                "label": f"第{i+1}帧",
                "time": time_str,
                "pil_img": img_pil,
                "has_target": has_target
            })
        cap.release()
        ratio = (target_detected_count / len(frames_data)) * 100 if frames_data else 0.0
        return frames_data, ratio

# =========================================================================
# 模块 3: 全功能 UI
# =========================================================================

class UnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 智能视频筛选器 v2.5 - 劳保手套检测版")
        self.root.geometry("1400x950")
        
        self.current_filepath = None
        self.checkbox_vars = {}
        self.model_select_vars = {} 
        self.cached_preview_data = [] 
        self.cached_ratio = 0.0
        
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.stop_flag = False
        self.is_running = False
        
        self.detector = YoloDetector()
        self.file_manager = FileManager()
        self.video_processor = VideoProcessor(self.detector)
        
        self._init_ui()
        self._configure_styles()
        self._scan_models()
        self.preview_canvas.bind("<Configure>", self._on_window_resize)

    def _scan_models(self):
        if not os.path.exists("models"): os.makedirs("models")
        model_files = [f for f in os.listdir("models") if f.endswith(".pt")]
        
        self.menu_models.delete(0, tk.END)
        self.model_select_vars.clear()
        
        if not model_files:
            self.menu_models.add_command(label="请放入 .pt 模型文件", state="disabled")
            self.mb_models.config(text="无模型")
            return

        for f in model_files:
            var = tk.BooleanVar(value=False)
            self.model_select_vars[f] = var
            self.menu_models.add_checkbutton(label=f, variable=var, onvalue=True, offvalue=False, command=self._on_model_check)
            
        if model_files:
            self.model_select_vars[model_files[0]].set(True)
            self._update_model_btn_text()

    def _on_model_check(self):
        selected = [name for name, var in self.model_select_vars.items() if var.get()]
        if len(selected) > 3:
            messagebox.showwarning("限制", "最多只能同时勾选 3 个模型！")
            count = 0
            for name, var in self.model_select_vars.items():
                if var.get():
                    count += 1
                    if count > 3: var.set(False)
        self._update_model_btn_text()

    def _update_model_btn_text(self):
        count = sum(1 for v in self.model_select_vars.values() if v.get())
        if count == 0: self.mb_models.config(text="选择模型 (0)")
        else: self.mb_models.config(text=f"已选模型 ({count}/3)")

    def _configure_styles(self):
        style = ttk.Style()
        style.map("Treeview", background=[("selected", "#3a7ebf")], foreground=[("selected", "white")])
        self.tree.tag_configure('checked_item', background='#E3F2FD', foreground='black')
        self.tree.tag_configure('normal_item', background='white', foreground='black')

    def _init_ui(self):
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(fill=tk.X)
        
        # 1. 扫描
        path_group = tk.LabelFrame(top_frame, text="1. 扫描设置", padx=10, pady=5)
        path_group.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        self.path_var = tk.StringVar()
        self.entry_path = tk.Entry(path_group, textvariable=self.path_var, width=18)
        self.entry_path.pack(side=tk.LEFT, padx=2)
        self.btn_select = tk.Button(path_group, text="📂", command=self.select_folder)
        self.btn_select.pack(side=tk.LEFT, padx=2)
        self.btn_scan = tk.Button(path_group, text="🔍 扫描", command=self.search_files, bg="#4CAF50", fg="white", font=("Arial", 9, "bold"))
        self.btn_scan.pack(side=tk.LEFT, padx=5)

        # 2. AI
        ai_group = tk.LabelFrame(top_frame, text="2-4. AI 智能参数", padx=10, pady=5)
        ai_group.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        f_row1 = tk.Frame(ai_group)
        f_row1.pack(side=tk.TOP, fill=tk.X, pady=2)
        tk.Label(f_row1, text="模型:").pack(side=tk.LEFT)
        self.mb_models = tk.Menubutton(f_row1, text="选择模型", relief=tk.RAISED, bg="#f0f0f0", width=12)
        self.menu_models = tk.Menu(self.mb_models, tearoff=0)
        self.mb_models.config(menu=self.menu_models)
        self.mb_models.pack(side=tk.LEFT, padx=5)
        tk.Label(f_row1, text="帧数:").pack(side=tk.LEFT, padx=(10,0))
        self.preview_count_var = tk.StringVar(value="3")
        self.combo_frames = ttk.Combobox(f_row1, textvariable=self.preview_count_var, values=[str(i) for i in range(1, 31)], width=3)
        self.combo_frames.pack(side=tk.LEFT, padx=5)

        f_row2 = tk.Frame(ai_group)
        f_row2.pack(side=tk.TOP, fill=tk.X, pady=5)
        tk.Label(f_row2, text="灵敏度:").pack(side=tk.LEFT)
        self.conf_var = tk.DoubleVar(value=0.15)
        self.conf_scale = tk.Scale(f_row2, variable=self.conf_var, from_=0.01, to=0.95, resolution=0.01, orient=tk.HORIZONTAL, length=120, width=15, showvalue=0)
        self.conf_scale.pack(side=tk.LEFT, padx=5)
        self.spin_conf = tk.Spinbox(f_row2, textvariable=self.conf_var, from_=0.01, to=0.95, increment=0.01, width=4, format="%.2f")
        self.spin_conf.pack(side=tk.LEFT)

        f_row3 = tk.Frame(ai_group)
        f_row3.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.draw_labels_var = tk.BooleanVar(value=True)
        # [修改文案] 适应通用模型
        self.chk_draw = tk.Checkbutton(f_row3, text="显示识别框", variable=self.draw_labels_var)
        self.chk_draw.pack(side=tk.LEFT)
        tk.Frame(f_row3, width=20).pack(side=tk.LEFT)
        self.btn_start_ai = tk.Button(f_row3, text="▶ 运行", command=self.start_batch_ai_scan, bg="#2196F3", fg="white", font=("Arial", 9, "bold"), width=8)
        self.btn_start_ai.pack(side=tk.LEFT, padx=2)
        self.btn_pause = tk.Button(f_row3, text="⏸", command=self.toggle_pause, state=tk.DISABLED, width=3)
        self.btn_pause.pack(side=tk.LEFT, padx=2)
        self.btn_stop = tk.Button(f_row3, text="⏹", command=self.stop_task, state=tk.DISABLED, bg="#ffcccb", width=3)
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        # 3. 筛选
        del_group = tk.LabelFrame(top_frame, text="5. 结果处理", padx=10, pady=5, fg="red")
        del_group.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        f_del1 = tk.Frame(del_group)
        f_del1.pack(side=tk.TOP, pady=5)
        tk.Label(f_del1, text="出现率 <").pack(side=tk.LEFT)
        self.threshold_var = tk.IntVar(value=20)
        self.entry_thresh = tk.Entry(f_del1, textvariable=self.threshold_var, width=4)
        self.entry_thresh.pack(side=tk.LEFT, padx=2)
        tk.Label(f_del1, text="%").pack(side=tk.LEFT)
        self.btn_reselect = tk.Button(f_del1, text="⚡一键勾选", command=self.apply_threshold_selection, bg="#FF9800", fg="white")
        self.btn_reselect.pack(side=tk.LEFT, padx=5)
        f_del2 = tk.Frame(del_group)
        f_del2.pack(side=tk.TOP, pady=5)
        self.btn_del_files = tk.Button(f_del2, text="🗑 删文件", command=self.delete_selected_files, bg="#f44336", fg="white")
        self.btn_del_files.pack(side=tk.LEFT, padx=5)
        self.btn_del_folders = tk.Button(f_del2, text="📂 删文件夹", command=self.delete_selected_folders, bg="#D32F2F", fg="white")
        self.btn_del_folders.pack(side=tk.LEFT, padx=5)

        # 列表
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        list_frame = tk.Frame(paned)
        paned.add(list_frame, width=600)
        cols = ("checkbox", "filename", "ai_score", "folder", "full_path")
        self.tree = ttk.Treeview(list_frame, columns=cols, show='headings')
        headers = [("✓", 40), ("文件名", 200), ("出现率", 80), ("父文件夹", 120), ("完整路径", 150)]
        for col, (txt, w) in zip(cols, headers):
            self.tree.heading(col, text=txt)
            self.tree.column(col, width=w)
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<Button-1>", self.on_tree_click)
        self.tree.bind("<Button-1>", self.on_header_click, add="+")

        # 预览
        self.preview_frame = tk.Frame(paned, bg="#eeeeee")
        paned.add(self.preview_frame)
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="#eeeeee")
        self.preview_scroll = tk.Scrollbar(self.preview_frame, orient="vertical", command=self.preview_canvas.yview)
        self.preview_content = tk.Frame(self.preview_canvas, bg="#eeeeee")
        self.preview_win = self.preview_canvas.create_window((0,0), anchor="nw", window=self.preview_content)
        self.preview_content.bind("<Configure>", lambda e: self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all")))
        self.preview_canvas.pack(side="left", fill="both", expand=True)
        self.preview_scroll.pack(side="right", fill="y")
        self.preview_canvas.configure(yscrollcommand=self.preview_scroll.set)

        # 状态栏
        bottom_bar = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.gpu_status_var = tk.StringVar(value=self.detector.gpu_info)
        status_color = "#2E7D32" if "🚀" in self.detector.gpu_info else "black"
        tk.Label(bottom_bar, textvariable=self.gpu_status_var, fg=status_color, font=("Segoe UI", 9, "bold"), padx=10).pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="准备就绪")
        tk.Label(bottom_bar, textvariable=self.status_var, padx=10).pack(side=tk.RIGHT)
        self.progress = ttk.Progressbar(bottom_bar, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)

    # ----------------- 响应式布局 -----------------

    def _on_window_resize(self, event):
        self.preview_canvas.itemconfig(self.preview_win, width=event.width)
        self._reflow_grid(event.width)

    def _reflow_grid(self, container_width):
        if not self.cached_preview_data: return
        for widget in self.preview_content.winfo_children(): widget.destroy()

        tk.Label(self.preview_content, text=f"出现率: {self.cached_ratio:.1f}%", font=("bold",12), bg="#eeeeee").pack(pady=(10,5))
        f_container = tk.Frame(self.preview_content, bg="#eeeeee")
        f_container.pack(fill=tk.X, padx=5)

        count = len(self.cached_preview_data)
        if count == 0: return

        if container_width < 400: cols = 1
        elif count == 1: cols = 1
        elif count <= 4: cols = 2
        elif count <= 9: cols = 3
        else: cols = 4

        padding = 10
        item_w = int((container_width - (cols + 1) * padding) / cols)
        if item_w < 100: 
            item_w = 100
            cols = max(1, int((container_width - padding) / (item_w + padding)))

        for i, d in enumerate(self.cached_preview_data):
            f = tk.Frame(f_container, bd=1, relief="solid", padx=2, pady=2, bg="white")
            f.grid(row=i//cols, column=i%cols, padx=5, pady=5, sticky="nsew")
            
            pil_img = d['pil_img']
            w, h = pil_img.size
            item_h = int(item_w * h / w)
            resized_pil = pil_img.resize((item_w, item_h), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(resized_pil)
            
            l = tk.Label(f, image=tk_img, bg="white")
            l.image = tk_img 
            l.pack()
            tk.Label(f, text=f"{d['label']} ({d['time']})", bg="white").pack()
            
        for c in range(cols): f_container.grid_columnconfigure(c, weight=1)

    # ----------------- 逻辑控制 -----------------

    def _toggle_inputs(self, enable):
        state = tk.NORMAL if enable else tk.DISABLED
        self.btn_select.config(state=state)
        self.btn_scan.config(state=state)
        self.entry_path.config(state=state)
        self.mb_models.config(state=state) 
        self.combo_frames.config(state="readonly" if enable else tk.DISABLED)
        self.conf_scale.config(state=state)
        self.spin_conf.config(state=state)
        self.chk_draw.config(state=state)
        self.btn_start_ai.config(state=state)
        self.btn_reselect.config(state=state)
        self.entry_thresh.config(state=state)
        self.btn_del_files.config(state=state)
        self.btn_del_folders.config(state=state)

    def _set_ui_state_busy(self, is_ai_running=False):
        self.is_running = True
        self._toggle_inputs(False)
        if is_ai_running:
            self.btn_pause.config(state=tk.NORMAL, text="⏸", bg="SystemButtonFace")
            self.btn_stop.config(state=tk.NORMAL)
        else:
            self.btn_pause.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.DISABLED)

    def _set_ui_state_idle(self):
        self.is_running = False
        self.stop_flag = False
        self.pause_event.set()
        self._toggle_inputs(True)
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.DISABLED)

    def stop_task(self):
        if not self.is_running: return
        if messagebox.askyesno("确认终止", "确定要停止当前的 AI 扫描任务吗？"):
            self.stop_flag = True
            self.pause_event.set() 
            self.status_var.set("正在停止任务...")

    def select_folder(self):
        path = filedialog.askdirectory()
        if path: self.path_var.set(path)

    def search_files(self):
        self.status_var.set("正在扫描文件...")
        self.progress['mode'] = 'indeterminate'
        self.progress.start()
        self._set_ui_state_busy(is_ai_running=False)
        threading.Thread(target=self._search_thread, daemon=True).start()

    def _search_thread(self):
        target = self.path_var.get()
        if not target: 
            self.root.after(0, self._set_ui_state_idle)
            return

        self.root.after(0, lambda: [self.tree.delete(i) for i in self.tree.get_children()])
        self.checkbox_vars.clear()
        
        count = 0
        for root, file in self.file_manager.scan_directory(target):
            self.root.after(0, self._add_item, root, file)
            count += 1
            
        self.root.after(0, lambda: [self.progress.stop(), self.status_var.set(f"扫描完成，共 {count} 个文件")])
        self.root.after(0, self._set_ui_state_idle)

    def _add_item(self, root, file):
        item_id = self.tree.insert('', 'end', values=("", file, "--", os.path.basename(root), os.path.join(root, file)))
        self.checkbox_vars[item_id] = tk.BooleanVar(value=False)
        self.update_checkbox_display(item_id)

    def start_batch_ai_scan(self):
        selected_models = [name for name, var in self.model_select_vars.items() if var.get()]
        if not selected_models:
            return messagebox.showerror("错误", "请至少勾选一个模型！")

        items = self.tree.get_children()
        if not items: return
        
        try: scan_frames = int(self.preview_count_var.get())
        except: scan_frames = 3
        draw_labels = self.draw_labels_var.get()
        
        model_str = "\n  - ".join(selected_models)
        if not messagebox.askyesno("确认运行", f"将使用以下模型检测：\n  - {model_str}\n\n标注: {'开启' if draw_labels else '关闭'}"): return

        self.stop_flag = False
        self.pause_event.set()
        self._set_ui_state_busy(is_ai_running=True)
        self.progress['mode'] = 'determinate'
        self.progress['maximum'] = len(items)
        
        threading.Thread(target=self._ai_scan_thread, args=(items, scan_frames, draw_labels, selected_models), daemon=True).start()

    def _ai_scan_thread(self, items, scan_frames, draw_labels, selected_models):
        self.root.after(0, lambda: self.status_var.set("正在加载/切换模型..."))
        success, msg = self.detector.load_models(selected_models)
        self.root.after(0, lambda: self.gpu_status_var.set(msg))
        
        if not success:
            self.root.after(0, lambda: messagebox.showerror("加载失败", msg))
            self.root.after(0, self._set_ui_state_idle)
            return

        thresh = self.threshold_var.get()
        ai_conf = self.conf_var.get()
        
        for i, iid in enumerate(items):
            if self.stop_flag:
                self.root.after(0, lambda: self.status_var.set("任务已终止"))
                break
            self.pause_event.wait()

            path = self.tree.item(iid, 'values')[4]
            try:
                _, ratio = self.video_processor.extract_preview_data(path, scan_frames, 100, ai_conf, draw_labels)
                is_waste = ratio < thresh
                self.root.after(0, lambda id=iid, r=ratio, chk=is_waste: self._update_ai_result(id, r, chk))
            except Exception as e:
                print(e)
            
            self.root.after(0, lambda v=i+1: self.progress.configure(value=v))

        self.root.after(0, self._set_ui_state_idle)
        self.root.after(0, lambda: messagebox.showinfo("结果", "AI 分析完成"))

    def _update_ai_result(self, iid, ratio, check):
        if not self.tree.exists(iid): return 
        vals = self.tree.item(iid, 'values')
        self.tree.item(iid, values=(vals[0], vals[1], f"{ratio:.1f}%", vals[3], vals[4]))
        self.checkbox_vars[iid].set(check)
        self.update_checkbox_display(iid)

    def apply_threshold_selection(self):
        try: thresh = self.threshold_var.get()
        except: return
        count = 0
        for iid in self.tree.get_children():
            score_str = self.tree.item(iid, 'values')[2]
            if "%" in score_str:
                score = float(score_str.replace("%", ""))
                should = score < thresh
                self.checkbox_vars[iid].set(should)
                self.update_checkbox_display(iid)
                if should: count += 1
        self.status_var.set(f"已勾选 {count} 个出现率 < {thresh}% 的文件")

    def _get_checked_items(self):
        return [(i, self.tree.item(i, 'values')[4]) for i, v in self.checkbox_vars.items() if v.get()]

    def delete_selected_files(self):
        checked = self._get_checked_items()
        if not checked: return messagebox.showwarning("提示", "未勾选文件")
        
        if not messagebox.askyesno("确认", f"删除 {len(checked)} 个文件？\n(列表将实时更新，不重新扫描)"): return
        
        self._set_ui_state_busy(is_ai_running=False)
        self.status_var.set("正在删除文件...")
        threading.Thread(target=self._delete_files_thread, args=(checked,), daemon=True).start()

    def _delete_files_thread(self, checked_items):
        paths = [p for _, p in checked_items]
        success_count, errors = self.file_manager.delete_files(paths)
        self.root.after(0, lambda: self._post_delete_cleanup(checked_items, success_count, errors))

    def _post_delete_cleanup(self, checked_items, count, errors):
        for iid, path in checked_items:
            if not any(os.path.basename(path) in e for e in errors):
                if self.tree.exists(iid):
                    self.tree.delete(iid)
                    del self.checkbox_vars[iid]

        self._set_ui_state_idle()
        msg = f"成功删除 {count} 个文件。"
        if errors: msg += f"\n失败 {len(errors)} 个。"
        messagebox.showinfo("删除结果", msg)
        self.status_var.set(msg.split('\n')[0])

    def delete_selected_folders(self):
        checked = self._get_checked_items()
        if not checked: return messagebox.showwarning("提示", "未勾选任何文件")
        
        folders = set()
        for _, path in checked:
            folders.add(os.path.dirname(path))
            
        if not messagebox.askyesno("警告", f"将删除 {len(folders)} 个文件夹及其内容！\n确认继续？"): return

        self._set_ui_state_busy(is_ai_running=False)
        self.status_var.set("正在删除文件夹...")
        threading.Thread(target=self._delete_folders_thread, args=(list(folders),), daemon=True).start()

    def _delete_folders_thread(self, folders):
        count, errors = self.file_manager.delete_folders(folders)
        self.root.after(0, lambda: self._post_folder_delete_cleanup(folders, count, errors))

    def _post_folder_delete_cleanup(self, deleted_folders, count, errors):
        norm_deleted = [os.path.normpath(f) for f in deleted_folders]
        items_to_remove = []
        for iid in self.tree.get_children():
            path = self.tree.item(iid, 'values')[4]
            folder = os.path.dirname(path)
            if os.path.normpath(folder) in norm_deleted:
                items_to_remove.append(iid)
        
        for iid in items_to_remove:
            self.tree.delete(iid)
            if iid in self.checkbox_vars: del self.checkbox_vars[iid]

        self._set_ui_state_idle()
        messagebox.showinfo("删除结果", f"已删除 {count} 个文件夹。")

    def toggle_pause(self):
        if not self.is_running: return
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.btn_pause.config(text="▶", bg="#FFEB3B")
            self.status_var.set("任务已暂停...")
        else:
            self.pause_event.set()
            self.btn_pause.config(text="⏸", bg="SystemButtonFace")
            self.status_var.set("任务继续执行中...")

    def on_tree_click(self, event):
        if self.tree.identify_region(event.x, event.y) == "cell":
            if self.tree.identify_column(event.x) == "#1":
                iid = self.tree.identify_row(event.y)
                self.checkbox_vars[iid].set(not self.checkbox_vars[iid].get())
                self.update_checkbox_display(iid)

    def on_header_click(self, event):
        if self.tree.identify_column(event.x) == "#1":
            state = not any(v.get() for v in self.checkbox_vars.values())
            for v in self.checkbox_vars.values(): v.set(state)
            for iid in self.checkbox_vars: self.update_checkbox_display(iid)

    def update_checkbox_display(self, iid):
        v = self.checkbox_vars[iid].get()
        vals = self.tree.item(iid, 'values')
        self.tree.item(iid, values=("✓" if v else "",) + vals[1:], tags=('checked_item' if v else 'normal_item',))

    def on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel: return
        path = self.tree.item(sel[-1], 'values')[4]
        if self.current_filepath == path: return
        self.current_filepath = path
        threading.Thread(target=self._preview_thread, args=(path,), daemon=True).start()

    def _preview_thread(self, path):
        try:
            cnt = int(self.preview_count_var.get())
            ai_conf = self.conf_var.get()
            draw = self.draw_labels_var.get()
            # 400 是占位宽，这里不重要，因为我们现在用动态缩放
            data, ratio = self.video_processor.extract_preview_data(path, cnt, 400, ai_conf, draw)
            
            self.cached_preview_data = data
            self.cached_ratio = ratio
            
            # 回到主线程渲染
            self.root.after(0, lambda: self._render_preview_init())
        except: pass

    def _render_preview_init(self):
        """预览数据的初始化渲染，并触发一次排版"""
        self._reflow_grid(self.preview_frame.winfo_width())

if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    root = tk.Tk()
    app = UnifiedApp(root)
    root.mainloop()