import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_splitter_gui import ModelSplitterGUI

if __name__ == "__main__":
    import tkinter as tk
    root = tk.Tk()
    app = ModelSplitterGUI(root)
    root.mainloop()
