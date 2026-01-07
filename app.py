import os
import sys

# =========================================================================
# 0. 核心环境引导 (必须放在 import torch 之前)
# =========================================================================
def force_load_internal_cuda():
    """
    强制 Windows 优先加载软件内部打包的 CUDA/cuDNN 库，
    防止与用户系统环境变量中的 CUDA 版本冲突 (DLL Hell)。
    """
    if sys.platform == 'win32':
        # 获取运行时临时目录 (PyInstaller 解压目录) 或 当前脚本目录
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        
        # 定义需要强制加载的 DLL 路径 (适配 torch 和 nvidia 库)
        paths_to_add = [
            base_path,
            os.path.join(base_path, 'torch', 'lib'),
            os.path.join(base_path, 'nvidia', 'cuda_runtime', 'bin'),
            os.path.join(base_path, 'nvidia', 'cudnn', 'bin'),
            os.path.join(base_path, 'nvidia', 'cublas', 'bin'),
        ]

        # 将路径加入 DLL 搜索目录 (Python 3.8+ 必须)
        for p in paths_to_add:
            if os.path.exists(p):
                try:
                    os.add_dll_directory(p)
                except Exception:
                    pass
        
        # 同时修改 PATH 环境变量 (双重保险)
        os.environ['PATH'] = ';'.join(paths_to_add) + ';' + os.environ['PATH']

# 执行环境修复
force_load_internal_cuda()

# =========================================================================
# 正常导入其他库
# =========================================================================
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import torch # 必须在 force_load_internal_cuda 之后导入
import time

# =========================================================================
# 常量定义
# =========================================================================
CHECKED_ICON = "☑"
UNCHECKED_ICON = "☐"

# =========================================================================
# 模块 1: AI 智能引擎
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

    def get_model_classes(self, model_name):
        path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(path): path = model_name
        
        if model_name in self.models:
            return self.models[model_name].names
        
        try:
            from ultralytics import YOLO
            temp_model = YOLO(path)
            return temp_model.names
        except Exception:
            return {}

    def load_models(self, target_model_names):
        try:
            from ultralytics import YOLO
            current_keys = set(self.models.keys())
            target_keys = set(target_model_names)
            
            for name in (current_keys - target_keys):
                del self.models[name]
                print(f"已卸载: {name}")
                
            for name in (target_keys - current_keys):
                path = os.path.join(self.model_dir, name)
                if not os.path.exists(path): path = name
                if os.path.exists(path):
                    model = YOLO(path)
                    self.models[name] = model
                    print(f"已加载: {name}")
                else:
                    print(f"❌ 找不到: {name}")

            if self.models:
                return True, f"{self.gpu_info} | 加载: {len(self.models)}"
            else:
                return False, "未加载任何模型"
        except Exception as e:
            return False, str(e)

    def process_frame(self, frame, conf_threshold=0.15, draw=True, class_filters=None):
        if not self.models or frame is None:
            return False, frame

        has_target = False
        annotated_frame = frame.copy()

        for name, model in self.models.items():
            target_classes = None
            if class_filters and name in class_filters:
                selected_ids = class_filters[name]
                if selected_ids is not None:
                    target_classes = selected_ids
            
            results = model(frame, device=self.device, verbose=False, conf=conf_threshold, classes=target_classes)
            
            if results:
                r = results[0]
                if (len(r.boxes) > 0 or 
                   (r.keypoints is not None and len(r.keypoints.conf) > 0) or
                   (r.masks is not None)):
                    
                    has_target = True
                    if draw:
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

    def extract_preview_data(self, filepath, count, target_width, ai_conf, draw_skeleton, class_filters):
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

            has_target, annotated_frame = self.detector.process_frame(
                frame, conf_threshold=ai_conf, draw=draw_skeleton, class_filters=class_filters
            )
            
            if has_target: target_detected_count += 1
            
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
        self.root.title("智能视频筛选器V2.0           作者：倪梓纹")
        self.root.geometry("1400x950")
        
        self.current_filepath = None
        self.checkbox_vars = {}
        self.cached_preview_data = [] 
        self.cached_ratio = 0.0
        
        self.selected_models = set()
        self.active_class_filters = {}
        
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.stop_flag = False
        self.is_running = False
        
        self.detector = YoloDetector()
        self.file_manager = FileManager()
        self.video_processor = VideoProcessor(self.detector)
        
        self._init_ui()
        self._configure_styles()
        
        self._initial_scan()
        self.preview_canvas.bind("<Configure>", self._on_window_resize)

    def _initial_scan(self):
        if not os.path.exists("models"): os.makedirs("models")
        files = [f for f in os.listdir("models") if f.endswith(".pt")]
        if files:
            self.selected_models.add(files[0])
            self.status_var.set(f"默认选中: {files[0]} (点击管理按钮加载)")
            self.detector.load_models(list(self.selected_models))
            self._update_model_status_label()

    def _configure_styles(self):
        style = ttk.Style()
        style.map("Treeview", background=[("selected", "#3a7ebf")], foreground=[("selected", "white")])
        self.tree.tag_configure('checked_item', background='#E3F2FD', foreground='black')
        self.tree.tag_configure('normal_item', background='white', foreground='black')

    def _init_ui(self):
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(fill=tk.X)
        
        # --- 区域 1: 扫描 ---
        path_group = tk.LabelFrame(top_frame, text="1. 扫描设置", padx=10, pady=5)
        path_group.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        self.path_var = tk.StringVar()
        self.entry_path = tk.Entry(path_group, textvariable=self.path_var, width=18)
        self.entry_path.pack(side=tk.LEFT, padx=2)
        self.btn_select = tk.Button(path_group, text="📂", command=self.select_folder)
        self.btn_select.pack(side=tk.LEFT, padx=2)
        self.btn_scan = tk.Button(path_group, text="🔍 扫描", command=self.search_files, bg="#4CAF50", fg="white", font=("Arial", 9, "bold"))
        self.btn_scan.pack(side=tk.LEFT, padx=5)

        # --- 区域 2: AI 参数 ---
        ai_group = tk.LabelFrame(top_frame, text="2-4. AI 智能参数", padx=10, pady=5)
        ai_group.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        # 行1
        f_row1 = tk.Frame(ai_group)
        f_row1.pack(side=tk.TOP, fill=tk.X, pady=2)
        self.btn_manage_models = tk.Button(f_row1, text="⚙️ 模型与类别管理", command=self.open_model_manager, bg="#E3F2FD", font=("Arial", 9, "bold"))
        self.btn_manage_models.pack(side=tk.LEFT, padx=(0, 10))
        self.lbl_model_status = tk.Label(f_row1, text="已选: 0", fg="gray")
        self.lbl_model_status.pack(side=tk.LEFT)

        # 行2
        f_row2 = tk.Frame(ai_group)
        f_row2.pack(side=tk.TOP, fill=tk.X, pady=5)
        tk.Label(f_row2, text="帧数:").pack(side=tk.LEFT)
        self.preview_count_var = tk.StringVar(value="3")
        self.combo_frames = ttk.Combobox(f_row2, textvariable=self.preview_count_var, values=[str(i) for i in range(1, 31)], width=3)
        self.combo_frames.pack(side=tk.LEFT, padx=(0,10))
        tk.Label(f_row2, text="灵敏度:").pack(side=tk.LEFT)
        self.conf_var = tk.DoubleVar(value=0.15)
        self.conf_scale = tk.Scale(f_row2, variable=self.conf_var, from_=0.01, to=0.95, resolution=0.01, orient=tk.HORIZONTAL, length=100, width=15, showvalue=0)
        self.conf_scale.pack(side=tk.LEFT, padx=2)
        self.spin_conf = tk.Spinbox(f_row2, textvariable=self.conf_var, from_=0.01, to=0.95, increment=0.01, width=4, format="%.2f")
        self.spin_conf.pack(side=tk.LEFT)

        # 行3
        f_row3 = tk.Frame(ai_group)
        f_row3.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.draw_labels_var = tk.BooleanVar(value=True)
        self.chk_draw = tk.Checkbutton(f_row3, text="显示识别框", variable=self.draw_labels_var)
        self.chk_draw.pack(side=tk.LEFT)
        tk.Frame(f_row3, width=20).pack(side=tk.LEFT)
        self.btn_start_ai = tk.Button(f_row3, text="▶ 运行", command=self.start_batch_ai_scan, bg="#2196F3", fg="white", font=("Arial", 9, "bold"), width=8)
        self.btn_start_ai.pack(side=tk.LEFT, padx=2)
        self.btn_pause = tk.Button(f_row3, text="⏸", command=self.toggle_pause, state=tk.DISABLED, width=3)
        self.btn_pause.pack(side=tk.LEFT, padx=2)
        self.btn_stop = tk.Button(f_row3, text="⏹", command=self.stop_task, state=tk.DISABLED, bg="#ffcccb", width=3)
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        # --- 区域 3: 筛选 ---
        del_group = tk.LabelFrame(top_frame, text="5. 结果处理", padx=10, pady=5, fg="red")
        del_group.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        f_del1 = tk.Frame(del_group)
        f_del1.pack(side=tk.TOP, pady=5)
        tk.Label(f_del1, text="出现率 <").pack(side=tk.LEFT)
        self.threshold_var = tk.IntVar(value=20)
        self.entry_thresh = tk.Entry(f_del1, textvariable=self.threshold_var, width=4)
        self.entry_thresh.pack(side=tk.LEFT, padx=2)
        tk.Label(f_del1, text="%").pack(side=tk.LEFT)
        
        self.btn_reselect = tk.Button(f_del1, text="⚡增量筛选", command=self.apply_threshold_selection, bg="#FF9800", fg="white")
        self.btn_reselect.pack(side=tk.LEFT, padx=5)
        
        self.btn_clear_sel = tk.Button(f_del1, text="❌ 清空", command=self.clear_all_selection, width=6)
        self.btn_clear_sel.pack(side=tk.LEFT, padx=2)

        f_del2 = tk.Frame(del_group)
        f_del2.pack(side=tk.TOP, pady=5)
        self.btn_del_files = tk.Button(f_del2, text="🗑 删文件", command=self.delete_selected_files, bg="#f44336", fg="white")
        self.btn_del_files.pack(side=tk.LEFT, padx=5)
        self.btn_del_folders = tk.Button(f_del2, text="📂 删文件夹", command=self.delete_selected_folders, bg="#D32F2F", fg="white")
        self.btn_del_folders.pack(side=tk.LEFT, padx=5)

        # --- 主内容区 ---
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        list_frame = tk.Frame(paned)
        paned.add(list_frame, width=600)
        cols = ("checkbox", "filename", "ai_score", "folder", "full_path")
        self.tree = ttk.Treeview(list_frame, columns=cols, show='headings')
        
        # 优化列显示：checkbox 列居中
        headers = [("✓", 40), ("文件名", 200), ("出现率", 80), ("父文件夹", 120), ("完整路径", 150)]
        self.tree.heading("checkbox", text="✓")
        self.tree.column("checkbox", width=40, anchor="center") # 居中对齐
        
        for col, (txt, w) in zip(cols[1:], headers[1:]):
            self.tree.heading(col, text=txt)
            self.tree.column(col, width=w)
            
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # === 事件绑定 ===
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select_preview)
        self.tree.bind("<ButtonRelease-1>", self.on_tree_click_release)

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

        # 预览区滚轮支持 (带边界检查)
        def _preview_scroll(event):
            current = self.preview_canvas.yview()
            scroll_unit = int(-1 * (event.delta / 120))
            if scroll_unit == 0: return
            if scroll_unit < 0 and current[0] <= 0: return # 顶端禁止上滑
            if scroll_unit > 0 and current[1] >= 1: return # 底端禁止下滑
            self.preview_canvas.yview_scroll(scroll_unit, "units")
            
        self.preview_canvas.bind("<MouseWheel>", _preview_scroll)
        self.preview_content.bind("<MouseWheel>", _preview_scroll)

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

    # ----------------- 弹窗管理逻辑 -----------------

    def _create_scrollable_canvas(self, parent_frame):
        """通用方法：创建一个带滚动条的 Canvas 区域"""
        # 1. 滚动条 (先 pack 避免被挤出)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 2. Canvas
        canvas = tk.Canvas(parent_frame, bg="white")
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 3. 关联
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.configure(command=canvas.yview)
        
        # 4. 内部内容 Frame
        scrollable_frame = tk.Frame(canvas, bg="white")
        
        # 5. 绑定配置事件
        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        scrollable_frame.bind("<Configure>", _on_frame_configure)
        
        # 6. 创建窗口
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # 7. 绑定滚轮 + 边界检查
        def _on_mousewheel(event):
            current = canvas.yview()
            scroll_unit = int(-1 * (event.delta / 120))
            if scroll_unit == 0: return
            if scroll_unit < 0 and current[0] <= 0: return
            if scroll_unit > 0 and current[1] >= 1: return
            canvas.yview_scroll(scroll_unit, "units")
        
        canvas.bind("<MouseWheel>", _on_mousewheel)
        scrollable_frame.bind("<MouseWheel>", _on_mousewheel)
        
        return canvas, scrollable_frame, _on_mousewheel

    def open_model_manager(self):
        top = tk.Toplevel(self.root)
        top.title("模型加载与类别配置中心")
        top.geometry("900x600")
        
        paned = tk.PanedWindow(top, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === 左侧：模型库 ===
        frame_left = tk.LabelFrame(paned, text="1. 模型库 (勾选加载)", padx=5, pady=5)
        paned.add(frame_left, width=300)
        
        # 使用通用方法创建滚动区域
        _, content_l, wheel_func = self._create_scrollable_canvas(frame_left)
        
        # === 右侧：类别筛选 ===
        frame_right = tk.LabelFrame(paned, text="2. 类别筛选 (配置选中模型的类别)", padx=5, pady=5)
        paned.add(frame_right, width=500)
        
        self.lbl_right_header = tk.Label(frame_right, text="请先在左侧点击模型名称...", font=("Arial", 10, "bold"), fg="gray")
        self.lbl_right_header.pack(fill=tk.X, pady=5)
        
        # 右侧也需要一个容器来放搜索栏和列表
        self.frame_classes_container = tk.Frame(frame_right)
        self.frame_classes_container.pack(fill=tk.BOTH, expand=True)
        
        tk.Button(top, text="保存并应用配置", command=lambda: self._save_manager_config(top), bg="#4CAF50", fg="white", width=20, font=("bold", 10)).pack(pady=10)

        self.temp_model_vars = {}
        self.temp_class_vars = {} 
        self.current_editing_model = None
        
        # 填充左侧列表
        files = [f for f in os.listdir("models") if f.endswith(".pt")]
        if not files: tk.Label(content_l, text="未找到 .pt 文件").pack(pady=20)
        
        for f in files:
            is_checked = f in self.selected_models
            var = tk.BooleanVar(value=is_checked)
            self.temp_model_vars[f] = var
            row = tk.Frame(content_l, bd=1, relief=tk.RIDGE, bg="white")
            row.pack(fill=tk.X, pady=2)
            
            # 绑定滚轮到所有子控件
            row.bind("<MouseWheel>", wheel_func)
            
            chk = tk.Checkbutton(row, variable=var, bg="white")
            chk.pack(side=tk.LEFT)
            chk.bind("<MouseWheel>", wheel_func)
            
            btn = tk.Button(row, text=f, anchor="w", relief=tk.FLAT, bg="white", command=lambda m=f: self._load_classes_to_right_panel(m))
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
            btn.bind("<MouseWheel>", wheel_func)

    def _load_classes_to_right_panel(self, model_name):
        self.current_editing_model = model_name
        self.lbl_right_header.config(text=f"正在配置: [{model_name}] 的检测类别", fg="blue")
        
        # 清空右侧容器
        for w in self.frame_classes_container.winfo_children(): w.destroy()
        
        # 读取类别数据
        loading_lbl = tk.Label(self.frame_classes_container, text="读取元数据...")
        loading_lbl.pack(pady=20)
        self.root.update()
        
        classes = self.detector.get_model_classes(model_name)
        loading_lbl.destroy()
        
        if not classes:
            tk.Label(self.frame_classes_container, text="无法读取类别").pack()
            return

        # --- 初始化变量逻辑 ---
        if model_name not in self.temp_class_vars:
            self.temp_class_vars[model_name] = {}
            saved_ids = self.active_class_filters.get(model_name, None)
            for cid in classes.keys():
                is_on = (saved_ids is None) or (cid in saved_ids)
                self.temp_class_vars[model_name][cid] = tk.BooleanVar(value=is_on)

        # ================= UI 布局构建 =================
        
        # 1. 搜索栏区域
        search_frame = tk.Frame(self.frame_classes_container, pady=5)
        search_frame.pack(fill=tk.X)
        tk.Label(search_frame, text="🔍 搜索类别: ").pack(side=tk.LEFT)
        
        search_var = tk.StringVar()
        entry_search = tk.Entry(search_frame, textvariable=search_var)
        entry_search.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 2. 工具栏区域 (全选/全不选)
        tool_frame = tk.Frame(self.frame_classes_container, pady=2)
        tool_frame.pack(fill=tk.X)
        
        def toggle_all(state):
            for v in self.temp_class_vars[model_name].values(): v.set(state)
        
        tk.Button(tool_frame, text="全选", command=lambda: toggle_all(True), width=8, bg="#e8f5e9").pack(side=tk.LEFT, padx=2)
        tk.Button(tool_frame, text="全不选", command=lambda: toggle_all(False), width=8, bg="#ffebee").pack(side=tk.LEFT, padx=2)

        # 3. 滚动列表区域 (使用通用方法)
        list_container = tk.Frame(self.frame_classes_container)
        list_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        _, scrollable_frame, wheel_func = self._create_scrollable_canvas(list_container)

        # ================= 渲染与搜索逻辑 =================
        
        def refresh_list(*args):
            # 1. 清除旧列表
            for widget in scrollable_frame.winfo_children():
                widget.destroy()
            
            keyword = search_var.get().lower()
            row, col = 0, 0
            
            # 2. 遍历并筛选
            for cid, cname in classes.items():
                display_text = f"{cid}: {cname}"
                
                # 搜索过滤
                if keyword and (keyword not in str(cid) and keyword not in cname.lower()):
                    continue

                var = self.temp_class_vars[model_name][cid]
                
                # 创建复选框
                chk = tk.Checkbutton(scrollable_frame, text=display_text, variable=var, anchor="w", bg="white")
                chk.grid(row=row, column=col, sticky="ew", padx=5, pady=2)
                
                # 绑定滚轮事件到每个组件上
                chk.bind("<MouseWheel>", wheel_func)
                
                # 双列布局
                col += 1
                if col > 1: # 2列
                    col = 0
                    row += 1
            
            # 配置列权重
            scrollable_frame.grid_columnconfigure(0, weight=1)
            scrollable_frame.grid_columnconfigure(1, weight=1)

        # 监听搜索框输入
        search_var.trace("w", refresh_list)
        
        # 初始化显示
        refresh_list()

    def _save_manager_config(self, window):
        new_selected = set()
        for name, var in self.temp_model_vars.items():
            if var.get(): new_selected.add(name)
        
        if len(new_selected) > 3:
            messagebox.showwarning("警告", "建议最多勾选 3 个模型！")
        
        self.selected_models = new_selected
        self._update_model_status_label()
        
        for m_name, id_map in self.temp_class_vars.items():
            selected_ids = [cid for cid, v in id_map.items() if v.get()]
            all_ids = list(id_map.keys())
            
            if len(selected_ids) == len(all_ids):
                self.active_class_filters[m_name] = None
            elif len(selected_ids) == 0:
                self.active_class_filters[m_name] = []
            else:
                self.active_class_filters[m_name] = selected_ids
                
        if self.selected_models:
            self.status_var.set("正在应用配置...")
            threading.Thread(target=self._reload_models_async, daemon=True).start()
        
        window.destroy()

    def _reload_models_async(self):
        success, msg = self.detector.load_models(list(self.selected_models))
        self.root.after(0, lambda: self.gpu_status_var.set(msg))
        self.root.after(0, lambda: self.status_var.set("配置已更新"))

    def _update_model_status_label(self):
        self.lbl_model_status.config(text=f"已选: {len(self.selected_models)}")

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
        self.btn_manage_models.config(state=state)
        self.combo_frames.config(state="readonly" if enable else tk.DISABLED)
        self.conf_scale.config(state=state)
        self.spin_conf.config(state=state)
        self.chk_draw.config(state=state)
        self.btn_start_ai.config(state=state)
        self.btn_reselect.config(state=state)
        self.btn_clear_sel.config(state=state)
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
        # 默认使用 UNCHECKED_ICON
        item_id = self.tree.insert('', 'end', values=(UNCHECKED_ICON, file, "--", os.path.basename(root), os.path.join(root, file)))
        self.checkbox_vars[item_id] = tk.BooleanVar(value=False)

    def start_batch_ai_scan(self):
        if not self.selected_models:
            return messagebox.showerror("错误", "请先点击 [⚙️ 模型与类别管理] 选择至少一个模型！")

        items = self.tree.get_children()
        if not items: return
        
        try: scan_frames = int(self.preview_count_var.get())
        except: scan_frames = 3
        draw_labels = self.draw_labels_var.get()
        
        model_str = "\n   - ".join(self.selected_models)
        if not messagebox.askyesno("确认运行", f"将使用以下模型检测：\n   - {model_str}\n\n标注: {'开启' if draw_labels else '关闭'}"): return

        self.stop_flag = False
        self.pause_event.set()
        self._set_ui_state_busy(is_ai_running=True)
        self.progress['mode'] = 'determinate'
        self.progress['maximum'] = len(items)
        
        threading.Thread(target=self._ai_scan_thread, args=(items, scan_frames, draw_labels, list(self.selected_models), self.active_class_filters), daemon=True).start()

    def _ai_scan_thread(self, items, scan_frames, draw_labels, selected_models, class_filters):
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
                _, ratio = self.video_processor.extract_preview_data(path, scan_frames, 100, ai_conf, draw_labels, class_filters)
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
        # 保持第一列为当前视觉状态
        current_icon = vals[0]
        self.tree.item(iid, values=(current_icon, vals[1], f"{ratio:.1f}%", vals[3], vals[4]))
        
        # 自动勾选逻辑（如果你希望 AI 自动勾选，也要同步更新图标）
        self.checkbox_vars[iid].set(check)
        self.update_checkbox_display(iid)

    def apply_threshold_selection(self):
        try: thresh = self.threshold_var.get()
        except: return
        count_new = 0
        count_total = 0
        for iid in self.tree.get_children():
            score_str = self.tree.item(iid, 'values')[2]
            if "%" in score_str:
                try:
                    score = float(score_str.replace("%", ""))
                    if score < thresh:
                        if not self.checkbox_vars[iid].get():
                            self.checkbox_vars[iid].set(True)
                            self.update_checkbox_display(iid)
                            count_new += 1
                    
                    if self.checkbox_vars[iid].get():
                        count_total += 1
                except: pass
        self.status_var.set(f"当前已选 {count_total} 个 (本次新增 {count_new} 个)")

    def clear_all_selection(self):
        for iid in self.checkbox_vars:
            self.checkbox_vars[iid].set(False)
            self.update_checkbox_display(iid)
        self.status_var.set("已清空所有选择")

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

    # =========================================================
    # 核心修复区：新的点击判定逻辑
    # =========================================================
    
    def on_tree_click_release(self, event):
        """鼠标抬起时触发，精准判断点击了哪里"""
        region = self.tree.identify("region", event.x, event.y)
        column = self.tree.identify_column(event.x)
        
        # 1. 判定表头点击 (Heading)
        if region == "heading":
            if column == "#1": # 只有点第一列的表头才全选
                # 检查当前是否全选了
                all_checked = all(v.get() for v in self.checkbox_vars.values())
                new_state = not all_checked # 反转状态
                
                for iid, var in self.checkbox_vars.items():
                    var.set(new_state)
                    self.update_checkbox_display(iid)
        
        # 2. 判定单元格点击 (Cell)
        elif region == "cell":
            row_id = self.tree.identify_row(event.y)
            if not row_id: return
            
            if column == "#1": # 只有点第一列才切换勾选
                current_val = self.checkbox_vars[row_id].get()
                self.checkbox_vars[row_id].set(not current_val)
                self.update_checkbox_display(row_id)
            else:
                # 点击其他列：不做勾选操作
                pass

    def update_checkbox_display(self, iid):
        """根据 checkbox_vars 的状态更新图标"""
        if iid not in self.checkbox_vars: return
        v = self.checkbox_vars[iid].get()
        vals = self.tree.item(iid, 'values')
        
        # 使用常量图标
        icon = CHECKED_ICON if v else UNCHECKED_ICON
        
        # 更新第一列，保持其他列不变
        new_vals = (icon,) + vals[1:]
        
        # 同时应用背景色样式
        tag = 'checked_item' if v else 'normal_item'
        self.tree.item(iid, values=new_vals, tags=(tag,))

    def on_tree_select_preview(self, event):
        sel = self.tree.selection()
        if not sel: return
        path = self.tree.item(sel[-1], 'values')[4] # 假设路径在第5列
        
        # 避免重复加载同一文件
        if self.current_filepath == path: return
        
        self.current_filepath = path
        threading.Thread(target=self._preview_thread, args=(path,), daemon=True).start()

    def _preview_thread(self, path):
        try:
            cnt = int(self.preview_count_var.get())
            ai_conf = self.conf_var.get()
            draw = self.draw_labels_var.get()
            data, ratio = self.video_processor.extract_preview_data(path, cnt, 400, ai_conf, draw, self.active_class_filters)
            self.cached_preview_data = data
            self.cached_ratio = ratio
            self.root.after(0, lambda: self._render_preview_init())
        except: pass

    def _render_preview_init(self):
        self._reflow_grid(self.preview_frame.winfo_width())

if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    root = tk.Tk()
    app = UnifiedApp(root)
    root.mainloop()