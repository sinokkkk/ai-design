"""交通标志检测与识别 —— 图形界面。

用法：
    E:/Anaconda/envs/torch-xpu/python.exe gui/app.py
"""
import ctypes
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# Windows 高 DPI：必须在创建 Tk 之前声明，否则系统会把窗口位图放大导致字体发虚
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gui.worker import Recognizer  # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("交通标志检测与识别")

        # 按实际 DPI 放大字号和窗口尺寸（96 DPI 时为 1.0）
        dpi = self.root.winfo_fpixels("1i")
        self.scale = dpi / 96.0
        self.root.tk.call("tk", "scaling", dpi / 72.0)
        self.root.geometry(f"{int(1180 * self.scale)}x{int(720 * self.scale)}")

        self.recognizer = None
        self.image_path = None
        self.video_path = None
        self.result_img = None
        self.display_img = None
        self.busy = False
        self.queue = queue.Queue()

        self._build_ui()
        self.root.after(100, self._poll_queue)

    def px(self, n):
        """把 96 DPI 下的像素值换算成当前 DPI 下的像素值。"""
        return int(n * self.scale)

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=self.px(8))
        top.pack(fill="x")

        ttk.Button(top, text="选择图片", command=self.choose_image).pack(side="left", padx=self.px(4))
        ttk.Button(top, text="选择视频", command=self.choose_video).pack(side="left", padx=self.px(4))
        ttk.Button(top, text="开始识别", command=self.start).pack(side="left", padx=self.px(4))
        ttk.Button(top, text="保存结果", command=self.save_result).pack(side="left", padx=self.px(4))

        ttk.Label(top, text="置信度").pack(side="left", padx=(self.px(16), self.px(4)))
        self.conf_var = tk.DoubleVar(value=0.25)
        ttk.Scale(top, from_=0.05, to=0.9, variable=self.conf_var,
                  orient="horizontal", length=self.px(140)).pack(side="left")
        self.conf_label = ttk.Label(top, text="0.25")
        self.conf_label.pack(side="left", padx=self.px(4))
        self.conf_var.trace_add("write", lambda *_: self.conf_label.config(
            text=f"{self.conf_var.get():.2f}"))

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=self.px(8), pady=self.px(4))

        self.canvas = tk.Canvas(body, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        right = ttk.Frame(body, width=self.px(280))
        right.pack(side="right", fill="y", padx=(self.px(8), 0))
        right.pack_propagate(False)

        ttk.Label(right, text="识别结果", font=("Microsoft YaHei", 10, "bold")).pack(anchor="w")
        self.result_box = tk.Listbox(right, font=("Consolas", 9))
        self.result_box.pack(fill="both", expand=True, pady=(self.px(4), self.px(8)))

        self.summary = ttk.Label(right, text="尚未识别")
        self.summary.pack(anchor="w")

        self.status = ttk.Label(self.root, text="就绪", relief="sunken", anchor="w", padding=self.px(4))
        self.status.pack(fill="x", side="bottom")

    def _set_status(self, text):
        self.status.config(text=text)

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片", " ".join(f"*{e}" for e in IMG_EXTS)), ("全部", "*.*")],
        )
        if not path:
            return
        self.image_path = path
        self.video_path = None
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("错误", "无法读取这张图片")
            return
        self.result_img = img
        self._show(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.result_box.delete(0, "end")
        self.summary.config(text="已加载图片，点开始识别")
        self._set_status(f"已加载 {Path(path).name}")

    def choose_video(self):
        path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("视频", " ".join(f"*{e}" for e in VIDEO_EXTS)), ("全部", "*.*")],
        )
        if not path:
            return
        self.video_path = path
        self.image_path = None
        self._set_status(f"已选择 {Path(path).name}，点开始识别")

    def start(self):
        if self.busy:
            return
        if not self.image_path and not self.video_path:
            messagebox.showinfo("提示", "先选择一张图片或一个视频")
            return

        self.busy = True
        self.result_box.delete(0, "end")
        conf = self.conf_var.get()
        target = self.image_path or self.video_path
        kind = "image" if self.image_path else "video"
        self._set_status("模型加载中…")

        threading.Thread(target=self._run_job, args=(target, kind, conf), daemon=True).start()

    def _run_job(self, path, kind, conf):
        try:
            if self.recognizer is None:
                self.recognizer = Recognizer()
            t0 = time.time()

            if kind == "image":
                img = cv2.imread(path)
                out, items = self.recognizer.run(img, conf=conf)
                self.queue.put(("image", out, items, time.time() - t0))
            else:
                out_path, frames, last_items = self._run_video(path, conf)
                self.queue.put(("video", out_path, last_items, time.time() - t0, frames))
        except Exception as e:
            self.queue.put(("error", str(e)))

    def _run_video(self, path, conf):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("无法打开视频")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_dir = ROOT / "output"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"pred_{Path(path).stem}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_idx = 0
        last_items = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            out, items = self.recognizer.run(frame, conf=conf)
            writer.write(out)
            if items:
                last_items = items
            if frame_idx % 25 == 0:
                self.root.after(0, lambda i=frame_idx, t=total: self._set_status(
                    f"处理中 {i}/{t if t else '?'}"))
        cap.release()
        writer.release()
        return out_path, frame_idx, last_items

    def _poll_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == "error":
                    self.busy = False
                    self._set_status("出错")
                    messagebox.showerror("运行出错", msg[1])
                elif msg[0] == "image":
                    _, out, items, elapsed = msg
                    self.result_img = out
                    self._show(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                    self._fill_results(items)
                    self.busy = False
                    self._set_status(f"识别完成，{len(items)} 个标志，耗时 {elapsed:.2f}s")
                elif msg[0] == "video":
                    _, out_path, items, elapsed, frames = msg
                    cap = cv2.VideoCapture(str(out_path))
                    ok, last = cap.read()
                    cap.release()
                    if ok:
                        self.result_img = last
                        self._show(cv2.cvtColor(last, cv2.COLOR_BGR2RGB))
                    self._fill_results(items)
                    self.busy = False
                    self._set_status(
                        f"视频处理完成 {frames} 帧，耗时 {elapsed:.1f}s，已存 {out_path.name}")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _fill_results(self, items):
        self.result_box.delete(0, "end")
        if not items:
            self.summary.config(text="未检测到标志")
            return
        for i, it in enumerate(items, 1):
            self.result_box.insert(
                "end", f"{i}. {it['label']}")
            self.result_box.insert(
                "end", f"   det {it['det']:.2f}  cls {it['cls']:.2f}")
        self.summary.config(text=f"共 {len(items)} 个标志")

    def save_result(self):
        if self.result_img is None:
            messagebox.showinfo("提示", "还没有可保存的结果")
            return
        out_dir = ROOT / "output"
        out_dir.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=str(out_dir), defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if not path:
            return
        cv2.imwrite(path, self.result_img)
        self._set_status(f"已保存 {Path(path).name}")

    def _show(self, rgb_array):
        self.display_img = rgb_array
        self._redraw()

    def _redraw(self):
        if self.display_img is None:
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        h, w = self.display_img.shape[:2]
        k = min(cw / w, ch / h)
        nw, nh = max(int(w * k), 1), max(int(h * k), 1)
        img = Image.fromarray(self.display_img).resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_img, anchor="center")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
