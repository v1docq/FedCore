import sys
import os
import json
from typing import List, Dict, Any
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np
from PIL import Image, ImageTk

from model_analyzer import ModelAnalyzer
from model_splitter import ModelSplitter
from model_exporter import ModelExporter

class ModelSplitterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Splitter Tool")
        self.root.geometry("1200x800")
        
        # Переменные для хранения путей
        self.model_path = tk.StringVar()
        self.arch_path = tk.StringVar()
        self.export_dir = tk.StringVar()
        
        # Переменные для хранения данных
        self.model = None
        self.device_arch = None
        self.analysis_result = None
        self.splitter = None
        self.exporter = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Создание основной структуры интерфейса
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Конфигурация сетки
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="Model Splitter Tool", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Загрузка модели
        model_frame = ttk.LabelFrame(main_frame, text="Model", padding="10")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(model_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(model_frame, text="Browse", command=self.browse_model).grid(row=0, column=2)
        
        # Загрузка архитектуры
        arch_frame = ttk.LabelFrame(main_frame, text="Device Architecture", padding="10")
        arch_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        arch_frame.columnconfigure(1, weight=1)
        
        ttk.Label(arch_frame, text="Architecture File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(arch_frame, textvariable=self.arch_path, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(arch_frame, text="Browse", command=self.browse_arch).grid(row=0, column=2)
        
        # Выбор директории экспорта
        export_frame = ttk.LabelFrame(main_frame, text="Export Directory", padding="10")
        export_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        export_frame.columnconfigure(1, weight=1)
        
        ttk.Label(export_frame, text="Export Path:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(export_frame, textvariable=self.export_dir, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(export_frame, text="Browse", command=self.browse_export_dir).grid(row=0, column=2)
        
        # Кнопки управления
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=(10, 20))
        
        ttk.Button(button_frame, text="Load Model", command=self.load_model).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Analyze Model", command=self.analyze_model).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Split Model", command=self.split_model).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Export Parts", command=self.export_parts).grid(row=0, column=3, padx=5)
        ttk.Button(button_frame, text="Save Results", command=self.save_results).grid(row=0, column=4, padx=5)
        
        # Вкладки для отображения результатов
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        main_frame.rowconfigure(5, weight=1)
        
        # Вкладка структуры модели
        self.model_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.model_frame, text="Model Structure")
        
        # Вкладка результатов разделения
        self.split_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.split_frame, text="Split Results")
        
        # Вкладка графа модели
        self.graph_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.graph_frame, text="Model Graph")
        
        # Создание контейнеров для визуализации
        self.create_model_structure_view()
        self.create_split_results_view()
        self.create_graph_view()
        
        # Статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def create_model_structure_view(self):
        # Создание виджета для отображения структуры модели
        tree_frame = ttk.Frame(self.model_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview для отображения структуры
        columns = ('Layer', 'Type', 'Supported', 'Module')
        self.model_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=20)
        
        for col in columns:
            self.model_tree.heading(col, text=col)
            self.model_tree.column(col, width=150)
        
        # Скроллбары
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.model_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.model_tree.xview)
        self.model_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.model_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_split_results_view(self):
        # Создание виджета для отображения результатов разделения
        results_frame = ttk.Frame(self.split_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Текстовое поле для результатов
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=20)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_graph_view(self):
        # Создание виджета для отображения графа модели
        graph_frame = ttk.Frame(self.graph_frame)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создание matplotlib фигуры
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def browse_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch files", "*.pt *.pth"), ("All files", "*.*")]
        )
        if file_path:
            self.model_path.set(file_path)
            
    def browse_arch(self):
        file_path = filedialog.askopenfilename(
            title="Select Architecture File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self.arch_path.set(file_path)
            
    def browse_export_dir(self):
        dir_path = filedialog.askdirectory(
            title="Select Export Directory"
        )
        if dir_path:
            self.export_dir.set(dir_path)
            
    def load_model(self):
        model_path = self.model_path.get()
        arch_path = self.arch_path.get()
        
        if not model_path or not arch_path:
            messagebox.showerror("Error", "Please select both model and architecture files")
            return
            
        try:
            # Загрузка модели с проверкой типа
            print(f"Loading model from: {model_path}")
            self.model = torch.load(model_path, map_location='cpu')
            
            # Проверяем, является ли модель torch script
            if hasattr(self.model, 'forward'):
                print("Model loaded successfully")
            else:
                print("Model loaded but might be in unexpected format")
                
            self.status_var.set(f"Model loaded from {model_path}")
            
            # Загрузка архитектуры устройства
            with open(arch_path, 'r') as f:
                self.device_arch = json.load(f)
            self.status_var.set(f"Architecture loaded from {arch_path}")
            
            # Создаем экземпляры классов
            self.splitter = ModelSplitter(self.device_arch)
            self.exporter = ModelExporter(self.device_arch)
            
            messagebox.showinfo("Success", "Model and architecture loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load files: {str(e)}")
            self.status_var.set("Error loading files")
            
    def analyze_model(self):
        if not self.model or not self.device_arch:
            messagebox.showerror("Error", "Please load model and architecture first")
            return
            
        try:
            analyzer = ModelAnalyzer(self.device_arch)
            self.analysis_result = analyzer.get_model_parts_info(self.model)
            
            # Обновление вкладки структуры модели
            self.update_model_structure_view()
            
            # Обновление вкладки результатов
            self.update_split_results_view()
            
            # Обновление графа модели
            self.update_model_graph()
            
            self.status_var.set("Model analyzed successfully")
            messagebox.showinfo("Success", "Model analyzed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze model: {str(e)}")
            self.status_var.set("Error analyzing model")
            
    def update_model_structure_view(self):
        # Очистка существующих данных
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
            
        # Добавление новых данных
        if self.analysis_result:
            layers = self.analysis_result['model_layers']
            for i, layer in enumerate(layers):
                supported = "Yes" if layer['supported'] else "No"
                self.model_tree.insert('', tk.END, values=(
                    layer['name'],
                    layer['type'],
                    supported,
                    type(layer['module']).__name__
                ))
                
    def update_split_results_view(self):
        # Очистка существующего текста
        self.results_text.delete(1.0, tk.END)
        
        if self.analysis_result:
            # Формирование текста результатов
            result_text = f"Model Analysis Results\n"
            result_text += f"{'=' * 50}\n\n"
            
            result_text += f"Total Layers: {self.analysis_result['total_layers']}\n"
            result_text += f"Supported Layers: {self.analysis_result['supported_layers']}\n"
            result_text += f"Unsupported Layers: {self.analysis_result['unsupported_layers']}\n"
            result_text += f"Split Points: {self.analysis_result['split_points']}\n\n"
            
            result_text += f"Parts Information:\n"
            result_text += f"{'=' * 30}\n"
            
            for part in self.analysis_result['parts_info']:
                part_type = "NPU" if part['is_npu_part'] else "CPU"
                result_text += f"\nPart {part['part_index']} ({part_type}):\n"
                result_text += f"  Layers: {part['start_layer']}-{part['end_layer']}\n"
                result_text += f"  Total Layers: {part['layers_count']}\n"
                result_text += f"  Supported: {part['supported_layers']}\n"
                result_text += f"  Unsupported: {part['unsupported_layers']}\n"
                
            self.results_text.insert(1.0, result_text)
            
    def update_model_graph(self):
        # Очистка графика
        self.ax.clear()
        
        if self.analysis_result:
            # Создание графа модели
            G = nx.DiGraph()
            
            # Добавление узлов
            layers = self.analysis_result['model_layers']
            for i, layer in enumerate(layers):
                color = 'green' if layer['supported'] else 'red'
                G.add_node(i, label=f"{layer['name']}\n{layer['type']}", 
                          color=color, layer=layer)
            
            # Добавление ребер (последовательность слоев)
            for i in range(len(layers) - 1):
                G.add_edge(i, i + 1)
            
            # Позиционирование узлов
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Рисование узлов
            node_colors = [G.nodes[i]['color'] for i in G.nodes()]
            nx.draw_networkx_nodes(G, pos, ax=self.ax, node_color=node_colors, 
                                 node_size=800, alpha=0.8)
            
            # Рисование ребер
            nx.draw_networkx_edges(G, pos, ax=self.ax, arrows=True, 
                                 arrowstyle='->', arrowsize=20)
            
            # Рисование меток
            labels = {i: G.nodes[i]['label'] for i in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, ax=self.ax, font_size=8)
            
            self.ax.set_title("Model Architecture Graph")
            self.ax.axis('off')
            
        self.canvas.draw()
        
    def split_model(self):
        if not self.model or not self.analysis_result:
            messagebox.showerror("Error", "Please analyze model first")
            return
            
        try:
            # Разделение модели
            parts = self.splitter.split_model(self.model, self.analysis_result)
            self.status_var.set(f"Model split into {len(parts)} parts")
            messagebox.showinfo("Success", f"Model split into {len(parts)} parts successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to split model: {str(e)}")
            self.status_var.set("Error splitting model")
            
    def export_parts(self):
        if not self.model or not self.analysis_result:
            messagebox.showerror("Error", "Please analyze model first")
            return
            
        if not self.export_dir.get():
            messagebox.showerror("Error", "Please select export directory")
            return
            
        try:
            # Разделение модели
            parts = self.splitter.split_model(self.model, self.analysis_result)
            
            # Экспорт частей
            exported_files = self.exporter.export_parts(parts, self.export_dir.get())
            
            self.status_var.set(f"Exported {len(exported_files)} files")
            messagebox.showinfo("Success", f"Exported {len(exported_files)} files successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export parts: {str(e)}")
            self.status_var.set("Error exporting parts")
            
    def save_results(self):
        if not self.analysis_result:
            messagebox.showerror("Error", "No results to save")
            return
            
        try:
            # Сохранение результатов в файл
            save_path = filedialog.asksaveasfilename(
                title="Save Results",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")]
            )
            
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump(self.analysis_result, f, indent=2)
                messagebox.showinfo("Success", f"Results saved to {save_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

def main():
    root = tk.Tk()
    app = ModelSplitterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
