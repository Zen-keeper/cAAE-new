import os,glob
import pandas as pd
import xlwt

path_all = "D:\work\ADNI_AV45_TI_HC_MCI_AD\ADNI_CN"
ps = glob.glob(path_all+"\\new*\\_co\\*\\mask*")

csv_file = r"D:\work\AD_V3\ADNIMERGE2.xlsx"
df = pd.read_excel(csv_file, sheet_name="ADNIMERGE")
data = df.values
names = df.values[:, 1]
dates = df.values[:, 6]
file = xlwt.Workbook()
sheet = file.add_sheet('nc_train', cell_overwrite_ok=True)
hang = 0
for p in ps:
    tems = p.split("\\")
    obj = tems[6]
    name = tems[7]
    date = name[9:17]
    try:
        for j in range(int(len(names))):
            # print(pd.Timestamp(date) == dates[j])
            if (obj == names[j] and pd.Timestamp(date).year == dates[j].year and abs(
                    pd.Timestamp(date).month - dates[j].month) < 2):
                sheet.write(hang, 0, obj)
                sheet.write(hang, 1, tems[-1])
                for col in range(2, 113):
                    sheet.write(hang, col, data[j][col])
    except BaseException:
        continue
    hang = hang + 1

file.save("nc_train_inf.xls")


