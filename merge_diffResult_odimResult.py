import pandas as pd

diff_file = '/home/x1112480/ODIM/Results_Excel/diff_model_result.csv'
# odim_file = '/home/x1112480/ODIM/Results_Excel/odim_result.csv'
odim_file = '/home/x1112480/ODIM/Results_Excel/odim_result_std.csv'
diff_result = pd.read_csv(diff_file, index_col = 0)
odim_result = pd.read_csv(odim_file, index_col = 0)

index_split = odim_result.index.str.split('_')
index_refine = []
for i in range(len(index_split)):
    indice = index_split[i]
    if len(indice) > 1:
        index_refine.append(indice[1].lower())
    else:
        index_refine.append(indice[0].lower())
odim_result.index = index_refine

odim_result.columns = odim_result.columns.str.lower()
merge_result = diff_result.join(odim_result, how='outer')

# merge_result.to_csv('/home/x1112480/ODIM/Results_Excel/merge_diff_odim.csv')
merge_result.to_csv('/home/x1112480/ODIM/Results_Excel/merge_diff_odim_std.csv')
