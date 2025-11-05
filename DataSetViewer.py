from tkinter import *
from tkinter import ttk
import pandas as pd

class DataSetViewer:
    def __init__(self):
        pass

    def create_ui(self):
        self.root = Tk()
        self.root.title("Dataset Viewer - House Pricing Prediction")
        self.root.geometry("800x600")
        main_panel = PanedWindow(self.root)
        main_panel['bg']="yellow"
        main_panel.pack(fill=BOTH, expand=True)

        #Define columns
        columns = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                   'Avg. Area Number of Bedrooms','Price', 'Area Population']

        self.tree = ttk.Treeview(main_panel, columns=columns, show="headings")

        self.tree.heading("Avg. Area Income", text="Avg. Area Income")
        self.tree.heading("Avg. Area House Age", text="Avg. Area House Age")
        self.tree.heading("Avg. Area Number of Rooms", text="Avg. Area Number of Rooms")
        self.tree.heading("Avg. Area Number of Bedrooms", text="Avg. Area Number of Bedrooms")
        self.tree.heading("Area Population", text="Area Population")
        self.tree.heading("Price", text="Price")
        #self.tree.grid(row=0,colunmn=0,sticky="nsew"
        self.tree.pack(side=LEFT,fill=BOTH,expand=TRUE)
        scrollbar = ttk.Scrollbar(main_panel, orient=VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        #self.tree.grid(row=0,colunmn=0,sticky="ns"
        scrollbar.pack(side=RIGHT, fill=BOTH, expand=True)

    def show_ui(self):
        self.root.mainloop()

    def show_data_listview(self, fileName):
        df = pd.read_csv(fileName)
        for i in range(0, len(df)):
            values = (df.iloc[i][0], df.iloc[i][1], df.iloc[i][2],
                      df.iloc[i][3], df.iloc[i][4], df.iloc[i][5])
            print(values)
            self.tree.insert("", END, values=values)

