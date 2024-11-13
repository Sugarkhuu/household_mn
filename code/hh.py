import pandas as pd
import os

datadir = "C:\\Users\\radnaa\\Downloads\\"
filename = '03_livestock.dta'
for filename in os.listdir(datadir):
    if filename.endswith('.dta'):
        # Read the .dta file
        df = pd.read_stata(os.path.join(datadir, filename))
        # df = df.iloc[:1000,:]
        # Extract file name without extension
        file_name_without_extension = os.path.splitext(filename)[0]
        
        # Save as Excel file
        excel_file_path = os.path.join(datadir, f"{file_name_without_extension}.xlsx")
        df.to_excel(excel_file_path, index=False)
        print(f"{filename} converted to {file_name_without_extension}.xlsx")
