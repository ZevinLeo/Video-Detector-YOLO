import os
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import time

# =========================================================================
# 模块 1: AI 智能引擎 (YOLOv8 GPU)
# =========================================================================

class YoloDetector:
    def __init__(self):
        self.available = False
        self.model = None
        self.device = 'cpu'
        self._try_load_model()

    def _try_load_model(self):
        try:
            from ultralytics import YOLO
            if torch.cuda.is_available():
                self.device = 'cuda'
            self.model = YOLO('yolov8n-pose.pt') 
            self.available = True
        except Exception as e:
            print(f"YOLO 加载失败: {e}")
            self.available = False

    def process_frame(self, frame, conf_threshold=0.15):
        if not self.available or frame is None:
            return False, frame

        results = self.model(frame, device=self.device, verbose=False, conf=conf_threshold)
        has_hand = False
        annotated_frame = frame.copy()

        if results:
            r = results[0]
            if r.keypoints is not None and r.keypoints.conf is not None:
                keypoints_xy = r.keypoints.xy.cpu().numpy()
                keypoints_conf = r.keypoints.conf.cpu().numpy()

                for i in range(len(keypoints_xy)):
                    kpts = keypoints_conf[i]
                    if len(kpts) < 11: continue 

                    # 只要有一个点的置信度 > 传入的阈值，就算有手
                    if (kpts[9] > conf_threshold or kpts[10] > conf_threshold or 
                        kpts[7] > conf_threshold or kpts[8] > conf_threshold):
                        
                        has_hand = True
                        
                        if kpts[9] > conf_threshold: self._draw_marker(annotated_frame, *keypoints_xy[i][9], "L-Wrist")
                        elif kpts[7] > conf_threshold: self._draw_marker(annotated_frame, *keypoints_xy[i][7], "L-Arm")
                        
                        if kpts[10] > conf_threshold: self._draw_marker(annotated_frame, *keypoints_xy[i][10], "R-Wrist")
                        elif kpts[8] > conf_threshold: self._draw_marker(annotated_frame, *keypoints_xy[i][8], "R-Arm")

        return has_hand, annotated_frame

    def _draw_marker(self, img, x, y, label):
        cv2.circle(img, (int(x), int(y)), 20, (0, 255, 255), 2) 
        cv2.circle(img, (int(x), int(y)), 6, (0, 255, 0), -1)
        cv2.putText(img, label, (int(x)-20, int(y)-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# =========================================================================
# 模块 2: 核心逻辑层 (保持不变)
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

    def extract_preview_data(self, filepath, count, target_width, ai_conf):
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
        hand_detected_count = 0

        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue

            has_hand, annotated_frame = self.detector.process_frame(frame, conf_threshold=ai_conf)
            if has_hand: hand_detected_count += 1
            
            img_tk = self._resize_for_tk(annotated_frame, target_width, count)
            time_sec = idx / fps if fps else 0
            time_str = f"{int(time_sec//60):02d}:{int(time_sec%60):02d}"
            
            frames_data.append({
                "label": f"第{i+1}帧",
                "time": time_str,
                "img_tk": img_tk,
                "has_hand": has_hand
            })
        cap.release()
        ratio = (hand_detected_count / len(frames_data)) * 100 if frames_data else 0.0
        return frames_data, ratio

    def _resize_for_tk(self, frame_bgr, target_width, grid_count):
        """[升级] 适配 1-30 帧的自动缩放逻辑"""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        
        is_narrow_mode = target_width < 380 
        
        # 动态计算最大宽度，保证30张图也能放下
        if is_narrow_mode or grid_count <= 1: max_w = int(target_width * 0.95)
        elif grid_count <= 4:  max_w = int(target_width * 0.46) # 2列
        elif grid_count <= 9:  max_w = int(target_width * 0.30) # 3列
        elif grid_count <= 16: max_w = int(target_width * 0.22) # 4列
        elif grid_count <= 25: max_w = int(target_width * 0.18) # 5列
        else:                  max_w = int(target_width * 0.15) # 6列 (Max 30)

        max_w = min(max_w, 500) 
        max_h = int(max_w * 0.75)
        
        scale = min(max_w/width, max_h/height)
        new_w, new_h = int(width * scale), int(height * scale)
        img = Image.fromarray(frame_rgb)
        return ImageTk.PhotoImage(img.resize((new_w, new_h), Image.Resampling.LANCZOS))

# =========================================================================
# 模块 3: 全功能 UI (升级: 支持30帧选项)
# =========================================================================

class UnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("全能版 MP4 AI智能筛选器    作者：倪梓纹")
        self.root.geometry("1400x900")
        
        self.current_filepath = None
        self.checkbox_vars = {}
        
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.stop_flag = False
        self.is_running = False
        
        self.detector = YoloDetector()
        self.file_manager = FileManager()
        self.video_processor = VideoProcessor(self.detector)
        
        self._init_ui()
        self._configure_styles()
        
        if not self.detector.available:
            messagebox.showwarning("提示", "未检测到 ultralytics")

    def _configure_styles(self):
        style = ttk.Style()
        style.map("Treeview", background=[("selected", "#3a7ebf")], foreground=[("selected", "white")])
        self.tree.tag_configure('checked_item', background='#E3F2FD', foreground='black')
        self.tree.tag_configure('normal_item', background='white', foreground='black')

    def _init_ui(self):
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(fill=tk.X)
        
        # 流程 1: 获取文件列表
        path_group = tk.LabelFrame(top_frame, text="1. 获取文件列表", padx=5, pady=5)
        path_group.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        self.path_var = tk.StringVar()
        tk.Entry(path_group, textvariable=self.path_var, width=20).pack(side=tk.LEFT)
        tk.Button(path_group, text="📂", command=self.select_folder).pack(side=tk.LEFT)
        tk.Button(path_group, text="扫描", command=self.search_files, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)

        # 流程 2 & 3 & 4: 视图 + AI 设置 + 运行
        ai_group = tk.LabelFrame(top_frame, text="2-4. 视图/灵敏度/AI初筛", padx=5, pady=5)
        ai_group.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # 流程 2: 视图范围 (1-30帧)
        tk.Label(ai_group, text="帧数:").pack(side=tk.LEFT)
        self.preview_count_var = tk.StringVar(value="3")
        # [升级] 范围扩大到 1-30
        ttk.Combobox(ai_group, textvariable=self.preview_count_var, values=[str(i) for i in range(1, 31)], width=3).pack(side=tk.LEFT, padx=(0,10))

        # 流程 3: AI灵敏度
        tk.Label(ai_group, text="AI灵敏度:").pack(side=tk.LEFT)
        self.conf_scale = tk.Scale(ai_group, from_=0.01, to=0.9, resolution=0.01, orient=tk.HORIZONTAL, length=80)
        self.conf_scale.set(0.15) 
        self.conf_scale.pack(side=tk.LEFT, padx=(0,10))

        # 流程 4: AI初筛 (含暂停/终止)
        self.btn_start_ai = tk.Button(ai_group, text="▶ 运行AI初筛", command=self.start_batch_ai_scan, bg="#2196F3", fg="white")
        self.btn_start_ai.pack(side=tk.LEFT, padx=2)
        self.btn_pause = tk.Button(ai_group, text="⏸", command=self.toggle_pause, state=tk.DISABLED, width=3)
        self.btn_pause.pack(side=tk.LEFT, padx=1)
        self.btn_stop = tk.Button(ai_group, text="⏹", command=self.stop_task, state=tk.DISABLED, bg="#ffcccb", width=3)
        self.btn_stop.pack(side=tk.LEFT, padx=1)

        # 流程 5 & 6: 阈值筛选 + 删除
        del_group = tk.LabelFrame(top_frame, text="5-6. 筛选阈值/删除", padx=5, pady=5, fg="red")
        del_group.pack(side=tk.LEFT, padx=5, fill=tk.Y)

        # 流程 5: 选择含手率阈值
        filter_frame = tk.Frame(del_group)
        filter_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        tk.Label(filter_frame, text="含手率 <").pack(side=tk.LEFT)
        self.threshold_var = tk.IntVar(value=20)
        tk.Entry(filter_frame, textvariable=self.threshold_var, width=3).pack(side=tk.LEFT)
        tk.Label(filter_frame, text="%").pack(side=tk.LEFT)
        tk.Button(filter_frame, text="⚡重选", command=self.apply_threshold_selection, bg="#FF9800", fg="white", font=("Arial", 8)).pack(side=tk.LEFT, padx=5)

        # 流程 6: 删除
        action_frame = tk.Frame(del_group)
        action_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        tk.Button(action_frame, text="🗑 删文件", command=self.delete_selected_files, bg="#f44336", fg="white").pack(side=tk.LEFT, padx=2)
        tk.Button(action_frame, text="📂 删文件夹", command=self.delete_selected_folders, bg="#D32F2F", fg="white").pack(side=tk.LEFT, padx=2)

        # 主区域
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        list_frame = tk.Frame(paned)
        paned.add(list_frame, width=600)
        # 修改列名显示
        cols = ("checkbox", "filename", "ai_score", "folder", "full_path")
        self.tree = ttk.Treeview(list_frame, columns=cols, show='headings')
        headers = [("✓", 40), ("文件名", 200), ("含手率", 80), ("父文件夹", 120), ("完整路径", 150)]
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

        self.preview_frame = tk.Frame(paned, bg="#eeeeee")
        paned.add(self.preview_frame)
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="#eeeeee")
        self.preview_scroll = tk.Scrollbar(self.preview_frame, orient="vertical", command=self.preview_canvas.yview)
        self.preview_content = tk.Frame(self.preview_canvas, bg="#eeeeee")
        self.preview_win = self.preview_canvas.create_window((0,0), anchor="nw", window=self.preview_content)
        self.preview_content.bind("<Configure>", lambda e: self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all")))
        self.preview_canvas.bind("<Configure>", lambda e: self.preview_canvas.itemconfig(self.preview_win, width=self.preview_canvas.winfo_width()))
        self.preview_content.bind("<MouseWheel>", lambda e: self.preview_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.preview_canvas.pack(side="left", fill="both", expand=True)
        self.preview_scroll.pack(side="right", fill="y")
        self.preview_canvas.configure(yscrollcommand=self.preview_scroll.set)

        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_var = tk.StringVar(value="准备就绪")
        tk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

    # ----------------- 逻辑部分 -----------------

    def apply_threshold_selection(self):
        try:
            thresh = self.threshold_var.get()
        except:
            return
        count_checked = 0
        for iid in self.tree.get_children():
            score_str = self.tree.item(iid, 'values')[2]
            if "%" in score_str:
                try:
                    score = float(score_str.replace("%", ""))
                    should_check = score < thresh
                    self.checkbox_vars[iid].set(should_check)
                    self.update_checkbox_display(iid)
                    if should_check: count_checked += 1
                except:
                    pass
        self.status_var.set(f"筛选更新：已勾选 {count_checked} 个含手率低于 {thresh}% 的视频")

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

    def stop_task(self):
        if not self.is_running: return
        if messagebox.askyesno("确认终止", "确定要停止当前的 AI 扫描任务吗？"):
            self.stop_flag = True
            self.pause_event.set() 
            self.status_var.set("正在停止任务...")

    def _set_ui_state_running(self, is_running):
        self.is_running = is_running
        if is_running:
            self.btn_start_ai.config(state=tk.DISABLED)
            self.btn_pause.config(state=tk.NORMAL, text="⏸", bg="SystemButtonFace")
            self.btn_stop.config(state=tk.NORMAL)
            # 锁定帧数选择
            # self.path_var.set(self.path_var.get())
        else:
            self.btn_start_ai.config(state=tk.NORMAL)
            self.btn_pause.config(state=tk.DISABLED, text="⏸", bg="SystemButtonFace")
            self.btn_stop.config(state=tk.DISABLED)
            self.pause_event.set()
            self.stop_flag = False

    def select_folder(self):
        path = filedialog.askdirectory()
        if path: self.path_var.set(path)

    def search_files(self):
        self.status_var.set("搜索中...")
        self.progress['mode'] = 'indeterminate'
        self.progress.start()
        threading.Thread(target=self._search_thread, daemon=True).start()

    def _search_thread(self):
        target = self.path_var.get()
        if not target: return
        self.root.after(0, lambda: [self.tree.delete(i) for i in self.tree.get_children()])
        self.checkbox_vars.clear()
        for root, file in self.file_manager.scan_directory(target):
            self.root.after(0, self._add_item, root, file)
        self.root.after(0, lambda: [self.progress.stop(), self.status_var.set("搜索完成")])

    def _add_item(self, root, file):
        item_id = self.tree.insert('', 'end', values=("", file, "--", os.path.basename(root), os.path.join(root, file)))
        self.checkbox_vars[item_id] = tk.BooleanVar(value=False)
        self.update_checkbox_display(item_id)

    def start_batch_ai_scan(self):
        if not self.detector.available:
            return messagebox.showerror("错误", "AI模型未加载")
        items = self.tree.get_children()
        if not items: return
        
        # [逻辑] 获取当前用户设置的帧数，批量扫描也用这个帧数！
        try:
            scan_frames = int(self.preview_count_var.get())
        except:
            scan_frames = 3

        current_conf = self.conf_scale.get()
        msg = f"准备运行 AI 初筛。\n\n• 采样帧数: {scan_frames}\n• 灵敏度: {current_conf}\n\n确认开始？"
        if not messagebox.askyesno("确认", msg): return

        self.stop_flag = False
        self.pause_event.set()
        self._set_ui_state_running(True)
        self.progress['mode'] = 'determinate'
        self.progress['maximum'] = len(items)
        # 将 scan_frames 传入线程
        threading.Thread(target=self._ai_scan_thread, args=(items, scan_frames), daemon=True).start()

    def _ai_scan_thread(self, items, scan_frames):
        thresh = self.threshold_var.get()
        ai_conf = self.conf_scale.get()
        
        processed_count = 0
        
        for i, iid in enumerate(items):
            if self.stop_flag:
                self.root.after(0, lambda: self.status_var.set("任务已终止"))
                break
            self.pause_event.wait()

            path = self.tree.item(iid, 'values')[4]
            try:
                # 使用界面上选择的 scan_frames (1-30)
                _, ratio = self.video_processor.extract_preview_data(path, scan_frames, 100, ai_conf)
                
                is_waste = ratio < thresh
                self.root.after(0, lambda id=iid, r=ratio, chk=is_waste: self._update_ai_result(id, r, chk))
            except Exception as e:
                print(f"Error analyzing {path}: {e}")

            processed_count += 1
            self.root.after(0, lambda v=processed_count: self.progress.configure(value=v))

        self.root.after(0, lambda: self._set_ui_state_running(False))
        final_msg = f"分析完成！\n\n已处理: {processed_count}/{len(items)}\n现在请使用【阈值重选】功能筛选废片。"
        self.root.after(0, lambda: messagebox.showinfo("结果", final_msg))

    def _update_ai_result(self, iid, ratio, check):
        if not self.tree.exists(iid): return 
        vals = self.tree.item(iid, 'values')
        self.tree.item(iid, values=(vals[0], vals[1], f"{ratio:.1f}%", vals[3], vals[4]))
        self.checkbox_vars[iid].set(check)
        self.update_checkbox_display(iid)

    def on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel: return
        path = self.tree.item(sel[-1], 'values')[4]
        self.update_preview(path)

    def update_preview(self, path):
        if self.current_filepath == path: return
        self.current_filepath = path
        for w in self.preview_content.winfo_children(): w.destroy()
        tk.Label(self.preview_content, text="YOLO 分析中...", bg="#eeeeee").pack(pady=20)
        
        try: cnt = int(self.preview_count_var.get())
        except: cnt = 3
        
        ai_conf = self.conf_scale.get()
        w = max(self.preview_frame.winfo_width(), 400)
        threading.Thread(target=self._preview_thread, args=(path, cnt, w, ai_conf), daemon=True).start()

    def _preview_thread(self, path, cnt, w, ai_conf):
        data, ratio = self.video_processor.extract_preview_data(path, cnt, w, ai_conf)
        self.root.after(0, lambda: self._render_preview(data, ratio, w))
        self.root.after(0, lambda: self._sync_list_score(path, ratio))

    def _sync_list_score(self, target_path, ratio):
        for iid in self.tree.get_children():
            vals = self.tree.item(iid, 'values')
            if vals[4] == target_path:
                new_vals = (vals[0], vals[1], f"{ratio:.1f}%", vals[3], vals[4])
                self.tree.item(iid, values=new_vals)
                break

    def _render_preview(self, data, ratio, current_width):
        for w in self.preview_content.winfo_children(): w.destroy()
        head = tk.Frame(self.preview_content, bg="#eeeeee")
        head.pack(fill=tk.X, padx=5, pady=5)
        color = "green" if ratio >= self.threshold_var.get() else "red"
        tk.Label(head, text=f"YOLO 识别率: {ratio:.1f}%", fg=color, font=("bold", 12), bg="#eeeeee").pack(side=tk.LEFT)
        tk.Label(head, text="(含彩色骨架)", font=("Arial", 8), fg="gray", bg="#eeeeee").pack(side=tk.RIGHT)
        
        num_frames = len(data)
        if num_frames == 0: return
        
        # [升级] 适配 30 帧的列数逻辑
        if current_width < 380: cols = 1
        elif num_frames <= 1: cols = 1
        elif num_frames <= 4: cols = 2
        elif num_frames <= 9: cols = 3
        elif num_frames <= 16: cols = 4
        elif num_frames <= 25: cols = 5
        else: cols = 6 # Max 30 frames
        
        for i in range(0, num_frames, cols):
            row = tk.Frame(self.preview_content, bg="#eeeeee")
            row.pack(fill=tk.X)
            for j in range(cols):
                if i+j < num_frames:
                    d = data[i+j]
                    f = tk.Frame(row, bg="white", bd=1, relief="solid")
                    f.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.BOTH)
                    label_txt = f"{d['label']} {'(有人)' if d['has_hand'] else ''}"
                    tk.Label(f, text=label_txt, fg="green" if d['has_hand'] else "black", bg="white").pack()
                    l = tk.Label(f, image=d['img_tk'], bg="white")
                    l.image = d['img_tk']
                    l.pack()
                    tk.Label(f, text=d['time'], bg="white").pack()

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

    def _get_checked_paths(self):
        return [self.tree.item(i, 'values')[4] for i, v in self.checkbox_vars.items() if v.get()]

    def delete_selected_files(self):
        paths = self._get_checked_paths()
        if not paths: return messagebox.showwarning("提示", "未勾选文件")
        if messagebox.askyesno("确认", f"删除 {len(paths)} 个文件？"):
            self.file_manager.delete_files(paths)
            self.search_files()

    def delete_selected_folders(self):
        checked_ids = [i for i, v in self.checkbox_vars.items() if v.get()]
        if not checked_ids: return messagebox.showwarning("提示", "未勾选任何文件")
        folders = set()
        for iid in checked_ids:
            full_path = self.tree.item(iid, 'values')[4]
            folders.add(os.path.dirname(full_path))
        if messagebox.askyesno("重大警告", f"将删除 {len(folders)} 个文件夹及其内部所有内容！\n确认继续？"):
            self.file_manager.delete_folders(list(folders))
            self.search_files()

if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedApp(root)
    root.mainloop()