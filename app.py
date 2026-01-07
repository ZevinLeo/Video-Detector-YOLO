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
# 模块 1: AI 智能引擎 (保持 GPU 检测功能)
# =========================================================================

class YoloDetector:
    def __init__(self):
        self.available = False
        self.model = None
        self.device = 'cpu'
        self.gpu_info = "正在检测计算设备..."
        self._try_load_model()

    def _try_load_model(self):
        try:
            from ultralytics import YOLO
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                self.device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                cuda_ver = torch.version.cuda
                self.gpu_info = f"🚀 计算设备: {gpu_name} | CUDA: {cuda_ver} (加速中)"
            else:
                self.device = 'cpu'
                self.gpu_info = "🐢 计算设备: CPU (未检测到 GPU，运行较慢)"

            model_name = 'yolov8n-pose.pt'
            # 尝试加载模型
            if os.path.exists(os.path.join("models", model_name)):
                self.model = YOLO(os.path.join("models", model_name))
            else:
                self.model = YOLO(model_name)
            
            self.available = True
        except Exception as e:
            print(f"YOLO 加载失败: {e}")
            self.gpu_info = f"⚠️ AI 引擎加载失败: {str(e)}"
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
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        is_narrow_mode = target_width < 380 
        
        if is_narrow_mode or grid_count <= 1: max_w = int(target_width * 0.95)
        elif grid_count <= 4:  max_w = int(target_width * 0.46)
        elif grid_count <= 9:  max_w = int(target_width * 0.30)
        elif grid_count <= 16: max_w = int(target_width * 0.22)
        elif grid_count <= 25: max_w = int(target_width * 0.18)
        else:                  max_w = int(target_width * 0.15)

        max_w = min(max_w, 500) 
        max_h = int(max_w * 0.75)
        scale = min(max_w/width, max_h/height)
        new_w, new_h = int(width * scale), int(height * scale)
        img = Image.fromarray(frame_rgb)
        return ImageTk.PhotoImage(img.resize((new_w, new_h), Image.Resampling.LANCZOS))

# =========================================================================
# 模块 3: 全功能 UI (升级: 稳定性控制 + 删除不刷新)
# =========================================================================

class UnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("全能版 MP4 AI智能筛选器 (Stable) - 作者: 倪梓纹")
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
        
        # 1. 扫描区
        path_group = tk.LabelFrame(top_frame, text="获取文件列表", padx=5, pady=5)
        path_group.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        self.path_var = tk.StringVar()
        # 保存控件引用以便禁用
        self.entry_path = tk.Entry(path_group, textvariable=self.path_var, width=20)
        self.entry_path.pack(side=tk.LEFT)
        self.btn_select = tk.Button(path_group, text="📂", command=self.select_folder)
        self.btn_select.pack(side=tk.LEFT)
        self.btn_scan = tk.Button(path_group, text="扫描", command=self.search_files, bg="#4CAF50", fg="white")
        self.btn_scan.pack(side=tk.LEFT, padx=5)

        # 2. AI 设置区
        ai_group = tk.LabelFrame(top_frame, text="AI 设置与扫描", padx=5, pady=5)
        ai_group.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        tk.Label(ai_group, text="帧数:").pack(side=tk.LEFT)
        self.preview_count_var = tk.StringVar(value="3")
        self.combo_frames = ttk.Combobox(ai_group, textvariable=self.preview_count_var, values=[str(i) for i in range(1, 31)], width=3)
        self.combo_frames.pack(side=tk.LEFT, padx=(0,10))

        tk.Label(ai_group, text="AI灵敏度:").pack(side=tk.LEFT)
        self.conf_scale = tk.Scale(ai_group, from_=0.01, to=0.9, resolution=0.01, orient=tk.HORIZONTAL, length=80)
        self.conf_scale.set(0.15) 
        self.conf_scale.pack(side=tk.LEFT, padx=(0,10))

        self.btn_start_ai = tk.Button(ai_group, text="▶ 运行AI初筛", command=self.start_batch_ai_scan, bg="#2196F3", fg="white")
        self.btn_start_ai.pack(side=tk.LEFT, padx=2)
        self.btn_pause = tk.Button(ai_group, text="⏸", command=self.toggle_pause, state=tk.DISABLED, width=3)
        self.btn_pause.pack(side=tk.LEFT, padx=1)
        self.btn_stop = tk.Button(ai_group, text="⏹", command=self.stop_task, state=tk.DISABLED, bg="#ffcccb", width=3)
        self.btn_stop.pack(side=tk.LEFT, padx=1)

        # 3. 筛选删除区
        del_group = tk.LabelFrame(top_frame, text="筛选阈值/删除", padx=5, pady=5, fg="red")
        del_group.pack(side=tk.LEFT, padx=5, fill=tk.Y)

        filter_frame = tk.Frame(del_group)
        filter_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        tk.Label(filter_frame, text="含手率 <").pack(side=tk.LEFT)
        self.threshold_var = tk.IntVar(value=20)
        self.entry_thresh = tk.Entry(filter_frame, textvariable=self.threshold_var, width=3)
        self.entry_thresh.pack(side=tk.LEFT)
        tk.Label(filter_frame, text="%").pack(side=tk.LEFT)
        self.btn_reselect = tk.Button(filter_frame, text="⚡重选", command=self.apply_threshold_selection, bg="#FF9800", fg="white", font=("Arial", 8))
        self.btn_reselect.pack(side=tk.LEFT, padx=5)

        action_frame = tk.Frame(del_group)
        action_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        self.btn_del_files = tk.Button(action_frame, text="🗑 删文件", command=self.delete_selected_files, bg="#f44336", fg="white")
        self.btn_del_files.pack(side=tk.LEFT, padx=2)
        self.btn_del_folders = tk.Button(action_frame, text="📂 删文件夹", command=self.delete_selected_folders, bg="#D32F2F", fg="white")
        self.btn_del_folders.pack(side=tk.LEFT, padx=2)

        # 主列表和预览区
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        list_frame = tk.Frame(paned)
        paned.add(list_frame, width=600)
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
        self.preview_canvas.create_window((0,0), anchor="nw", window=self.preview_content)
        self.preview_content.bind("<Configure>", lambda e: self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all")))
        self.preview_canvas.bind("<Configure>", lambda e: self.preview_canvas.itemconfig(self.preview_canvas.find_all()[0], width=self.preview_canvas.winfo_width()))
        self.preview_canvas.pack(side="left", fill="both", expand=True)
        self.preview_scroll.pack(side="right", fill="y")
        self.preview_canvas.configure(yscrollcommand=self.preview_scroll.set)

        # 底部状态栏
        bottom_bar = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)
        status_color = "#2E7D32" if "加速中" in self.detector.gpu_info else "black"
        self.gpu_status_var = tk.StringVar(value=self.detector.gpu_info)
        tk.Label(bottom_bar, textvariable=self.gpu_status_var, fg=status_color, font=("Segoe UI", 9, "bold"), padx=10).pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="准备就绪")
        tk.Label(bottom_bar, textvariable=self.status_var, padx=10).pack(side=tk.RIGHT)
        self.progress = ttk.Progressbar(bottom_bar, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)

    # ----------------- 核心控制逻辑 (修改重点) -----------------

    def _toggle_inputs(self, enable):
        """[新增] 统一开启或禁用所有交互控件"""
        state = tk.NORMAL if enable else tk.DISABLED
        
        # 禁用/开启 扫描相关
        self.btn_select.config(state=state)
        self.btn_scan.config(state=state)
        self.entry_path.config(state=state)
        
        # 禁用/开启 AI 设置
        self.combo_frames.config(state="readonly" if enable else tk.DISABLED)
        self.conf_scale.config(state=state)
        self.btn_start_ai.config(state=state)
        
        # 禁用/开启 筛选删除 (注意：删除时这些也要禁掉)
        self.btn_reselect.config(state=state)
        self.entry_thresh.config(state=state)
        self.btn_del_files.config(state=state)
        self.btn_del_folders.config(state=state)

    def _set_ui_state_busy(self, is_ai_running=False):
        """任务开始时的 UI 状态：锁定所有，根据类型决定是否开放暂停/停止"""
        self.is_running = True
        self._toggle_inputs(False) # 先全锁
        
        if is_ai_running:
            # 如果是 AI 任务，解锁暂停和停止
            self.btn_pause.config(state=tk.NORMAL, text="⏸", bg="SystemButtonFace")
            self.btn_stop.config(state=tk.NORMAL)
        else:
            # 如果是扫描或删除任务，暂停停止也不让用
            self.btn_pause.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.DISABLED)

    def _set_ui_state_idle(self):
        """任务结束时的 UI 状态：解锁所有"""
        self.is_running = False
        self.stop_flag = False
        self.pause_event.set()
        self._toggle_inputs(True)
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.DISABLED)

    # ----------------- 业务逻辑 -----------------

    def select_folder(self):
        path = filedialog.askdirectory()
        if path: self.path_var.set(path)

    def search_files(self):
        self.status_var.set("正在扫描文件...")
        self.progress['mode'] = 'indeterminate'
        self.progress.start()
        # 锁定 UI
        self._set_ui_state_busy(is_ai_running=False)
        threading.Thread(target=self._search_thread, daemon=True).start()

    def _search_thread(self):
        target = self.path_var.get()
        if not target: 
            self.root.after(0, self._set_ui_state_idle)
            return

        # 清空列表
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
        if not self.detector.available:
            return messagebox.showerror("错误", "AI模型未加载")
        items = self.tree.get_children()
        if not items: return
        
        try: scan_frames = int(self.preview_count_var.get())
        except: scan_frames = 3
        current_conf = self.conf_scale.get()

        if not messagebox.askyesno("确认", f"开始 AI 初筛 (帧数:{scan_frames})？"): return

        self.stop_flag = False
        self.pause_event.set()
        # 锁定 UI (允许暂停停止)
        self._set_ui_state_busy(is_ai_running=True)
        self.progress['mode'] = 'determinate'
        self.progress['maximum'] = len(items)
        threading.Thread(target=self._ai_scan_thread, args=(items, scan_frames), daemon=True).start()

    def _ai_scan_thread(self, items, scan_frames):
        thresh = self.threshold_var.get()
        ai_conf = self.conf_scale.get()
        
        for i, iid in enumerate(items):
            if self.stop_flag:
                self.root.after(0, lambda: self.status_var.set("任务已终止"))
                break
            self.pause_event.wait()

            path = self.tree.item(iid, 'values')[4]
            try:
                _, ratio = self.video_processor.extract_preview_data(path, scan_frames, 100, ai_conf)
                is_waste = ratio < thresh
                self.root.after(0, lambda id=iid, r=ratio, chk=is_waste: self._update_ai_result(id, r, chk))
            except: pass
            
            self.root.after(0, lambda v=i+1: self.progress.configure(value=v))

        self.root.after(0, self._set_ui_state_idle)
        self.root.after(0, lambda: messagebox.showinfo("结果", "AI 分析完成"))

    def _update_ai_result(self, iid, ratio, check):
        if not self.tree.exists(iid): return 
        vals = self.tree.item(iid, 'values')
        self.tree.item(iid, values=(vals[0], vals[1], f"{ratio:.1f}%", vals[3], vals[4]))
        self.checkbox_vars[iid].set(check)
        self.update_checkbox_display(iid)

    # --- 筛选与删除 (核心修改：多线程删除 + 仅更新 UI 不重扫) ---

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
        self.status_var.set(f"已勾选 {count} 个含手率 < {thresh}% 的文件")

    def _get_checked_items(self):
        """返回被勾选的 (iid, filepath) 列表"""
        return [(i, self.tree.item(i, 'values')[4]) for i, v in self.checkbox_vars.items() if v.get()]

    def delete_selected_files(self):
        checked = self._get_checked_items()
        if not checked: return messagebox.showwarning("提示", "未勾选文件")
        
        if not messagebox.askyesno("确认", f"删除 {len(checked)} 个文件？\n(删除后将从列表中移除，不自动刷新)"): return
        
        # 锁定 UI (不允许暂停)
        self._set_ui_state_busy(is_ai_running=False)
        self.status_var.set("正在删除文件...")
        
        # 开启线程执行删除，避免界面卡死
        threading.Thread(target=self._delete_files_thread, args=(checked,), daemon=True).start()

    def _delete_files_thread(self, checked_items):
        paths = [p for _, p in checked_items]
        # 执行物理删除
        success_count, errors = self.file_manager.delete_files(paths)
        
        # UI 更新逻辑移回主线程
        self.root.after(0, lambda: self._post_delete_cleanup(checked_items, success_count, errors))

    def _post_delete_cleanup(self, checked_items, count, errors):
        # 仅移除被删除的条目，不重新扫描硬盘
        for iid, path in checked_items:
            # 简单检查：如果在 errors 里说明没删掉，就不移除 UI
            # 这里简化逻辑：只要尝试删了，就从 UI 移除，或者只移除真正成功的
            # 为了准确，我们假设没有报错的都成功了
            if not any(os.path.basename(path) in e for e in errors):
                if self.tree.exists(iid):
                    self.tree.delete(iid)
                    del self.checkbox_vars[iid]

        # 恢复 UI
        self._set_ui_state_idle()
        
        msg = f"成功删除 {count} 个文件。"
        if errors: msg += f"\n失败 {len(errors)} 个 (可能被占用)。"
        messagebox.showinfo("删除结果", msg)
        self.status_var.set(msg.split('\n')[0])

    def delete_selected_folders(self):
        # 文件夹删除逻辑稍微复杂，因为 Treeview 是按文件列出的
        # 这里为了稳定，我们先找到所有勾选文件对应的父文件夹
        checked = self._get_checked_items()
        if not checked: return messagebox.showwarning("提示", "未勾选任何文件")
        
        folders = set()
        for _, path in checked:
            folders.add(os.path.dirname(path))
            
        if not messagebox.askyesno("警告", f"将删除 {len(folders)} 个文件夹及其内部所有内容！\n确认继续？"): return

        self._set_ui_state_busy(is_ai_running=False)
        self.status_var.set("正在删除文件夹...")
        
        threading.Thread(target=self._delete_folders_thread, args=(list(folders),), daemon=True).start()

    def _delete_folders_thread(self, folders):
        count, errors = self.file_manager.delete_folders(folders)
        self.root.after(0, lambda: self._post_folder_delete_cleanup(folders, count, errors))

    def _post_folder_delete_cleanup(self, deleted_folders, count, errors):
        # 遍历 Treeview，如果文件的父文件夹在已删除列表中，则移除该行
        # 需要把 deleted_folders 里的路径标准化，防止路径斜杠不一致
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

    # --- 杂项 ---

    def stop_task(self):
        """[修复补回] 终止当前任务"""
        if not self.is_running: return
        
        # 弹出确认框
        if messagebox.askyesno("确认终止", "确定要停止当前的 AI 扫描任务吗？"):
            self.stop_flag = True
            self.pause_event.set()  # 确保如果处于暂停状态也能立刻唤醒线程去结束
            self.status_var.set("正在停止任务...")
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
            ai_conf = self.conf_scale.get()
            data, ratio = self.video_processor.extract_preview_data(path, cnt, 400, ai_conf)
            self.root.after(0, lambda: self._render_preview(data, ratio))
        except: pass

    def _render_preview(self, data, ratio):
        for w in self.preview_content.winfo_children(): w.destroy()
        tk.Label(self.preview_content, text=f"AI 识别率: {ratio:.1f}%", font=("bold",12)).pack()
        f_container = tk.Frame(self.preview_content)
        f_container.pack()
        cols = 3
        if len(data) > 9: cols = 4
        if len(data) > 16: cols = 5
        
        for i, d in enumerate(data):
            f = tk.Frame(f_container, bd=1, relief="solid", padx=2, pady=2)
            f.grid(row=i//cols, column=i%cols, padx=2, pady=2)
            tk.Label(f, image=d['img_tk']).pack()
            tk.Label(f, text=d['time']).pack()
            f.image = d['img_tk']

if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    root = tk.Tk()
    app = UnifiedApp(root)
    root.mainloop()