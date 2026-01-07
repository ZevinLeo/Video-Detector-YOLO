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
# 模块 1: AI 智能引擎 (支持 CUDA 11.8+ 自动适配)
# =========================================================================

class YoloDetector:
    def __init__(self):
        self.available = False
        self.model = None
        self.device = 'cpu'
        self.gpu_info = "计算设备: CPU (未检测到显卡驱动)"
        self._try_load_model()

    def _try_load_model(self):
        try:
            from ultralytics import YOLO
            
            # --- 兼容性检测逻辑 ---
            if torch.cuda.is_available():
                # 设置为并行加速模式，提高 11.8+ 环境下的推理速度
                torch.backends.cudnn.benchmark = True
                self.device = 'cuda'
                
                # 获取显卡详细信息
                gpu_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                self.gpu_info = f"计算设备: {gpu_name} | CUDA: {cuda_version} (加速中)"
            else:
                self.device = 'cpu'
                self.gpu_info = "计算设备: CPU (GPU 不可用)"

            # 加载模型 (默认从当前目录或 models 目录寻找)
            model_path = 'yolov8n-pose.pt' 
            # 如果你有 models 文件夹逻辑，可以改为 os.path.join(base_path, 'models', 'yolov8n-pose.pt')
            self.model = YOLO(model_path) 
            self.available = True
            
        except Exception as e:
            print(f"YOLO 加载失败: {e}")
            self.gpu_info = f"计算异常: {str(e)}"
            self.available = False

    def process_frame(self, frame, conf_threshold=0.15):
        if not self.available or frame is None:
            return False, frame

        # 推理阶段
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

                    # 判定逻辑：手腕或肘部
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
# 模块 2: 核心处理类
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
        for filepath in file_paths:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    success_count += 1
            except: pass
        return success_count

    @staticmethod
    def delete_folders(folder_paths):
        success_count = 0
        for folder_path in folder_paths:
            try:
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                    success_count += 1
            except: pass
        return success_count

class VideoProcessor:
    def __init__(self, detector):
        self.detector = detector

    def extract_preview_data(self, filepath, count, target_width, ai_conf):
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened(): return [], 0.0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        indices = [int(total_frames * i / (count - 1)) if count > 1 else total_frames // 2 for i in range(count)]

        frames_data = []
        hand_detected_count = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames - 1))
            ret, frame = cap.read()
            if not ret: continue

            has_hand, annotated_frame = self.detector.process_frame(frame, conf_threshold=ai_conf)
            if has_hand: hand_detected_count += 1
            
            img_tk = self._resize_for_tk(annotated_frame, target_width, count)
            time_str = f"{int((idx/fps)//60):02d}:{int((idx/fps)%60):02d}"
            
            frames_data.append({"label": f"帧", "time": time_str, "img_tk": img_tk, "has_hand": has_hand})
        
        cap.release()
        ratio = (hand_detected_count / len(frames_data)) * 100 if frames_data else 0.0
        return frames_data, ratio

    def _resize_for_tk(self, frame_bgr, target_width, grid_count):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        
        if grid_count <= 1: max_w = int(target_width * 0.95)
        elif grid_count <= 9:  max_w = int(target_width * 0.30)
        elif grid_count <= 16: max_w = int(target_width * 0.22)
        else: max_w = int(target_width * 0.15)

        scale = min(max_w/width, (max_w*0.75)/height)
        new_w, new_h = int(width * scale), int(height * scale)
        img = Image.fromarray(frame_rgb)
        return ImageTk.PhotoImage(img.resize((new_w, new_h), Image.Resampling.LANCZOS))

# =========================================================================
# 模块 3: 界面展示 (新增 GPU 状态栏)
# =========================================================================

class UnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MP4 智能筛选器 (YOLO Pro) - 作者: 倪梓纹")
        self.root.geometry("1400x900")
        
        self.current_filepath = None
        self.checkbox_vars = {}
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.stop_flag = False
        self.is_running = False
        
        # 初始化 AI 引擎
        self.detector = YoloDetector()
        self.file_manager = FileManager()
        self.video_processor = VideoProcessor(self.detector)
        
        self._init_ui()
        self._configure_styles()

    def _init_ui(self):
        # 顶部工具栏
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(fill=tk.X)
        
        # 1. 扫描区
        path_group = tk.LabelFrame(top_frame, text="1. 扫描", padx=5, pady=5)
        path_group.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        self.path_var = tk.StringVar()
        tk.Entry(path_group, textvariable=self.path_var, width=20).pack(side=tk.LEFT)
        tk.Button(path_group, text="📂", command=self.select_folder).pack(side=tk.LEFT)
        tk.Button(path_group, text="扫描", command=self.search_files, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)

        # 2. AI 区
        ai_group = tk.LabelFrame(top_frame, text="2-4. AI 初筛", padx=5, pady=5)
        ai_group.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        tk.Label(ai_group, text="帧数:").pack(side=tk.LEFT)
        self.preview_count_var = tk.StringVar(value="3")
        ttk.Combobox(ai_group, textvariable=self.preview_count_var, values=[str(i) for i in range(1, 31)], width=3).pack(side=tk.LEFT, padx=5)
        
        tk.Label(ai_group, text="灵敏度:").pack(side=tk.LEFT)
        self.conf_scale = tk.Scale(ai_group, from_=0.01, to=0.9, resolution=0.01, orient=tk.HORIZONTAL, length=80)
        self.conf_scale.set(0.15)
        self.conf_scale.pack(side=tk.LEFT, padx=5)

        self.btn_start_ai = tk.Button(ai_group, text="▶ 运行AI", command=self.start_batch_ai_scan, bg="#2196F3", fg="white")
        self.btn_start_ai.pack(side=tk.LEFT, padx=5)
        self.btn_pause = tk.Button(ai_group, text="⏸", command=self.toggle_pause, state=tk.DISABLED, width=3)
        self.btn_pause.pack(side=tk.LEFT, padx=2)
        self.btn_stop = tk.Button(ai_group, text="⏹", command=self.stop_task, state=tk.DISABLED, bg="#ffcccb", width=3)
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        # 3. 筛选删除区
        del_group = tk.LabelFrame(top_frame, text="5-6. 筛选删除", padx=5, pady=5, fg="red")
        del_group.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        tk.Label(del_group, text="含手率 <").pack(side=tk.LEFT)
        self.threshold_var = tk.IntVar(value=20)
        tk.Entry(del_group, textvariable=self.threshold_var, width=3).pack(side=tk.LEFT)
        tk.Button(del_group, text="⚡重选", command=self.apply_threshold_selection, bg="#FF9800", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(del_group, text="🗑 删文件", command=self.delete_selected_files, bg="#f44336", fg="white").pack(side=tk.LEFT, padx=2)

        # 中间内容区 (PanedWindow)
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 列表
        list_frame = tk.Frame(paned)
        paned.add(list_frame, width=500)
        self.tree = ttk.Treeview(list_frame, columns=("checkbox", "filename", "score", "folder", "path"), show='headings')
        for col, txt, w in [("checkbox","✓",40), ("filename","文件名",150), ("score","含手率",80), ("folder","父文件夹",100), ("path","全路径",10)]:
            self.tree.heading(col, text=txt)
            self.tree.column(col, width=w)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<Button-1>", self.on_tree_click)

        # 预览区
        self.preview_frame = tk.Frame(paned, bg="#eeeeee")
        paned.add(self.preview_frame)
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="#eeeeee")
        self.preview_content = tk.Frame(self.preview_canvas, bg="#eeeeee")
        self.preview_canvas.create_window((0,0), anchor="nw", window=self.preview_content)
        self.preview_canvas.pack(side="left", fill="both", expand=True)
        self.preview_content.bind("<Configure>", lambda e: self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all")))

        # --- 底部状态栏 (双栏显示) ---
        bottom_bar = tk.Frame(self.root, relief=tk.SUNKEN, bd=1)
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 左侧：GPU 状态 (重点功能)
        self.gpu_status_var = tk.StringVar(value=self.detector.gpu_info)
        gpu_lbl = tk.Label(bottom_bar, textvariable=self.gpu_status_var, fg="#2E7D32", font=("Consolas", 9), padx=10)
        gpu_lbl.pack(side=tk.LEFT)
        
        # 中间：进度条
        self.progress = ttk.Progressbar(bottom_bar, mode='determinate', length=300)
        self.progress.pack(side=tk.RIGHT, padx=20, pady=2)
        
        # 右侧：任务状态
        self.status_var = tk.StringVar(value="系统准备就绪")
        tk.Label(bottom_bar, textvariable=self.status_var, padx=10).pack(side=tk.RIGHT)

    # ----------------- 逻辑函数 -----------------

    def _configure_styles(self):
        style = ttk.Style()
        self.tree.tag_configure('checked_item', background='#E3F2FD')

    def select_folder(self):
        path = filedialog.askdirectory()
        if path: self.path_var.set(path)

    def search_files(self):
        self.status_var.set("搜索中...")
        target = self.path_var.get()
        if not target: return
        [self.tree.delete(i) for i in self.tree.get_children()]
        self.checkbox_vars.clear()
        for root, file in self.file_manager.scan_directory(target):
            iid = self.tree.insert('', 'end', values=("", file, "--", os.path.basename(root), os.path.join(root, file)))
            self.checkbox_vars[iid] = tk.BooleanVar(value=False)
        self.status_var.set(f"搜索完成，共 {len(self.checkbox_vars)} 个文件")

    def start_batch_ai_scan(self):
        items = self.tree.get_children()
        if not items: return
        self._set_ui_state_running(True)
        self.progress['maximum'] = len(items)
        threading.Thread(target=self._ai_scan_thread, args=(items,), daemon=True).start()

    def _ai_scan_thread(self, items):
        ai_conf = self.conf_scale.get()
        scan_cnt = int(self.preview_count_var.get())
        thresh = self.threshold_var.get()
        
        for i, iid in enumerate(items):
            if self.stop_flag: break
            self.pause_event.wait()
            
            path = self.tree.item(iid, 'values')[4]
            try:
                _, ratio = self.video_processor.extract_preview_data(path, scan_cnt, 100, ai_conf)
                is_waste = ratio < thresh
                self.root.after(0, lambda id=iid, r=ratio, chk=is_waste: self._update_ai_result(id, r, chk))
            except: pass
            self.root.after(0, lambda v=i+1: self.progress.configure(value=v))

        self.root.after(0, lambda: self._set_ui_state_running(False))
        self.root.after(0, lambda: messagebox.showinfo("提示", "扫描任务结束"))

    def _update_ai_result(self, iid, ratio, check):
        if not self.tree.exists(iid): return
        vals = self.tree.item(iid, 'values')
        self.tree.item(iid, values=("✓" if check else "", vals[1], f"{ratio:.1f}%", vals[3], vals[4]))
        self.checkbox_vars[iid].set(check)

    def apply_threshold_selection(self):
        thresh = self.threshold_var.get()
        for iid in self.tree.get_children():
            score_str = self.tree.item(iid, 'values')[2]
            if "%" in score_str:
                score = float(score_str.replace("%", ""))
                self.checkbox_vars[iid].set(score < thresh)
                vals = self.tree.item(iid, 'values')
                self.tree.item(iid, values=("✓" if score < thresh else "",) + vals[1:])

    def _set_ui_state_running(self, running):
        self.is_running = running
        state = tk.DISABLED if running else tk.NORMAL
        self.btn_start_ai.config(state=state)
        self.btn_pause.config(state=tk.NORMAL if running else tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL if running else tk.DISABLED)
        if not running: self.stop_flag = False

    def toggle_pause(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.btn_pause.config(text="▶", bg="yellow")
        else:
            self.pause_event.set()
            self.btn_pause.config(text="⏸", bg="SystemButtonFace")

    def stop_task(self):
        self.stop_flag = True
        self.pause_event.set()

    def on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel: return
        path = self.tree.item(sel[0], 'values')[4]
        if self.current_filepath == path: return
        self.current_filepath = path
        threading.Thread(target=self._preview_thread, args=(path,), daemon=True).start()

    def _preview_thread(self, path):
        cnt = int(self.preview_count_var.get())
        ai_conf = self.conf_scale.get()
        data, ratio = self.video_processor.extract_preview_data(path, cnt, self.preview_frame.winfo_width(), ai_conf)
        self.root.after(0, lambda: self._render_preview(data, ratio))

    def _render_preview(self, data, ratio):
        for w in self.preview_content.winfo_children(): w.destroy()
        tk.Label(self.preview_content, text=f"AI 识别率: {ratio:.1f}%", font=("bold",12)).pack()
        f_container = tk.Frame(self.preview_content)
        f_container.pack()
        cols = 4 if len(data) > 16 else 3
        for i, d in enumerate(data):
            f = tk.Frame(f_container, bd=1, relief="solid", padx=2, pady=2)
            f.grid(row=i//cols, column=i%cols, padx=2, pady=2)
            tk.Label(f, image=d['img_tk']).pack()
            tk.Label(f, text=d['time']).pack()
            f.image = d['img_tk']

    def on_tree_click(self, event):
        if self.tree.identify_column(event.x) == "#1":
            iid = self.tree.identify_row(event.y)
            if iid:
                curr = self.checkbox_vars[iid].get()
                self.checkbox_vars[iid].set(not curr)
                vals = self.tree.item(iid, 'values')
                self.tree.item(iid, values=("✓" if not curr else "",) + vals[1:])

    def on_header_click(self, event):
        if self.tree.identify_column(event.x) == "#1":
            all_v = all(v.get() for v in self.checkbox_vars.values())
            for iid, v in self.checkbox_vars.items():
                v.set(not all_v)
                vals = self.tree.item(iid, 'values')
                self.tree.item(iid, values=("✓" if not all_v else "",) + vals[1:])

    def delete_selected_files(self):
        paths = [self.tree.item(i, 'values')[4] for i, v in self.checkbox_vars.items() if v.get()]
        if not paths: return
        if messagebox.askyesno("确认", f"确定删除选中的 {len(paths)} 个文件吗？"):
            self.file_manager.delete_files(paths)
            self.search_files()

if __name__ == "__main__":
    # 强制开启高 DPI 支持 (Windows 10/11 界面不模糊)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    
    root = tk.Tk()
    app = UnifiedApp(root)
    root.mainloop()