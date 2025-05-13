import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

class TTreeViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("TTree Event Viewer")

        self.data = None
        self.unique_event_ids = []

        # --- File & Tree Inputs ---
        frame_top = tk.Frame(root)
        frame_top.pack(pady=5)

        tk.Label(frame_top, text="ROOT file:").grid(row=0, column=0, sticky="e")
        self.file_entry = tk.Entry(frame_top, width=40)
        self.file_entry.insert(0, "many_muons.root")
        self.file_entry.grid(row=0, column=1)

        tk.Label(frame_top, text="Tree name:").grid(row=1, column=0, sticky="e")
        self.tree_entry = tk.Entry(frame_top, width=40)
        self.tree_entry.insert(0, "mu_ndlar/hits")
        self.tree_entry.grid(row=1, column=1)

        load_btn = tk.Button(frame_top, text="Load Data", command=self.load_data)
        load_btn.grid(row=0, column=2, rowspan=2, padx=10)

        # --- Event Selection ---
        self.event_var = tk.StringVar()
        self.event_dropdown = ttk.Combobox(root, textvariable=self.event_var, state="disabled")
        self.event_dropdown.pack(pady=5)
        self.event_dropdown.bind("<<ComboboxSelected>>", self.update_plots)

        # --- Plot Area ---
        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

    def load_data(self):
        filename = self.file_entry.get()
        treename = self.tree_entry.get()

        try:
            file = uproot.open(filename)
            tree = file[treename]
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file or tree:\n{e}")
            return

        try:
            self.data = tree.arrays(["event_id", "x", "y", "z", "totN", "Q"], library="np")
            self.unique_event_ids = np.unique(self.data["event_id"])
            self.event_dropdown["values"] = self.unique_event_ids.astype(str)
            self.event_dropdown["state"] = "readonly"
            self.event_var.set(str(self.unique_event_ids[0]))
            self.update_plots()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load branches:\n{e}")

    def update_plots(self, event=None):
        if self.data is None:
            return

        event_id = int(self.event_var.get())
        mask = self.data["event_id"] == event_id
        x = self.data["x"][mask]
        y = self.data["y"][mask]
        z = self.data["z"][mask]
        totN = self.data["totN"][mask]

        # Clear old plots
        for ax in self.axs:
            ax.clear()

        for t in [1, 2]:
            t_mask = totN == t
            label = f"totN={t}"
            self.axs[0].scatter(x[t_mask], y[t_mask], label=label)
            self.axs[1].scatter(z[t_mask], y[t_mask], label=label)
            self.axs[2].scatter(x[t_mask], z[t_mask], label=label)

        self.axs[0].set_xlabel("X"); self.axs[0].set_ylabel("Y")
        self.axs[1].set_xlabel("Z"); self.axs[1].set_ylabel("Y")
        self.axs[2].set_xlabel("X"); self.axs[2].set_ylabel("Z")
        for ax in self.axs:
            ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

# --- Run GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    app = TTreeViewer(root)
    root.mainloop()

