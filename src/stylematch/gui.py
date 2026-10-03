"""Small desktop window built on tkinter (ships with most Python installs)."""

import sys

from .core import compare


def launch() -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox, scrolledtext
    except ImportError:
        print(
            "stylematch: tkinter is not available in this Python.\n"
            "On Debian/Ubuntu install it with: sudo apt install python3-tk\n"
            "The command line version still works: stylematch a.txt b.txt",
            file=sys.stderr,
        )
        return 1

    try:
        root = tk.Tk()
    except tk.TclError as err:
        print(
            f"stylematch: could not open a window ({err}).\n"
            "Are you on a machine with no display? Use the command line instead.",
            file=sys.stderr,
        )
        return 1

    root.title("stylematch")
    root.geometry("900x660")
    root.configure(bg="#f0f0f0")

    tk.Label(root, text="stylematch: AI text check and style comparison", font=("Arial", 18, "bold"),
             bg="#f0f0f0").pack(pady=10)

    frame = tk.Frame(root, bg="#f0f0f0")
    frame.pack(pady=10, padx=20)
    boxes = []
    for col, name in enumerate(("Text sample 1", "Text sample 2")):
        tk.Label(frame, text=name, font=("Arial", 12, "bold"),
                 bg="#f0f0f0").grid(row=0, column=col, padx=10)
        box = scrolledtext.ScrolledText(frame, width=40, height=15,
                                        font=("Arial", 10), wrap=tk.WORD)
        box.grid(row=1, column=col, padx=10)
        boxes.append(box)

    result_label = tk.Label(root, text="Results will appear here", font=("Arial", 11),
                            bg="white", justify=tk.LEFT, anchor="w", padx=20, pady=20,
                            relief=tk.RAISED, borderwidth=2)

    def run_compare():
        text1 = boxes[0].get("1.0", tk.END).strip()
        text2 = boxes[1].get("1.0", tk.END).strip()
        if not text1 or not text2:
            messagebox.showerror("Error", "Please enter text in both boxes")
            return
        result = compare(text1, text2)
        if result.warnings:
            messagebox.showwarning("Warning", result.warnings[0])
        result_label.config(text=(
            f"Similarity score: {result.score:.4f}\n\n"
            f"Reading: {result.verdict}\n\n"
            f"Words in text 1: {result.words_1}\n"
            f"Words in text 2: {result.words_2}"
        ))

    def run_detect():
        import threading

        texts = [b.get("1.0", tk.END).strip() for b in boxes]
        if not any(texts):
            messagebox.showerror("Error", "Please enter text in at least one box")
            return
        try:
            from .detect import MissingDependencyError, detect
        except ImportError as err:  # pragma: no cover
            messagebox.showerror("Error", str(err))
            return
        result_label.config(text="Checking... the first run downloads the model, so it can take a while.")
        out = {}

        def work():
            try:
                out["lines"] = [
                    f"Text {i + 1}: AI score {r.score:.3f} - {r.verdict} ({r.device})"
                    for i, t in enumerate(texts) if t
                    for r in [detect(t)]
                ]
            except MissingDependencyError as err:
                out["error"] = str(err)
            except Exception as err:  # keep the window alive whatever happens
                out["error"] = f"{type(err).__name__}: {err}"

        worker = threading.Thread(target=work, daemon=True)
        worker.start()

        def poll():
            if worker.is_alive():
                root.after(200, poll)
            elif "error" in out:
                result_label.config(text=out["error"])
            else:
                result_label.config(text="\n\n".join(out["lines"]) + "\n\nA statistical guess, not proof.")

        poll()

    def clear():
        for box in boxes:
            box.delete("1.0", tk.END)
        result_label.config(text="Results will appear here")

    buttons = tk.Frame(root, bg="#f0f0f0")
    buttons.pack(pady=10)
    tk.Button(buttons, text="Detect AI", command=run_detect, font=("Arial", 12, "bold"),
              bg="#1565c0", fg="white", width=15).grid(row=0, column=0, padx=10)
    tk.Button(buttons, text="Compare style", command=run_compare, font=("Arial", 12, "bold"),
              bg="#2e7d32", fg="white", width=15).grid(row=0, column=1, padx=10)
    tk.Button(buttons, text="Clear", command=clear, font=("Arial", 12, "bold"),
              bg="#c62828", fg="white", width=15).grid(row=0, column=2, padx=10)
    result_label.pack(pady=10, padx=20, fill=tk.X)

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(launch())
