import csv

def read_csv(file_path, encoding='gbk'):
    data = []
    with open(file_path, 'r', newline='', encoding=encoding) as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return data


def update_seventh_column(file_path, encoding='gbk'):
    csv_data = read_csv(file_path, encoding)

    # 确保文件不为空
    if not csv_data:
        print("文件为空.")
        return

    # 获取第六列标题
    if len(csv_data[0]) >= 6:
        column_name_sixth = list(csv_data[0].keys())[5]  # 5表示第六列的索引
    else:
        print("CSV文件没有足够的列.")
        return

    # 获取第七列标题
    if csv_data[0] and len(csv_data[0]) >= 7:
        column_name_seventh = list(csv_data[0].keys())[6]  # 6表示第七列的索引
    else:
        print("CSV文件没有足够的列1.")
        return

    # 更新第七列的值
    for row in csv_data:
        if row[column_name_sixth] == "正常":
            row[column_name_seventh] = '0'
        elif row[column_name_sixth] == "低能放电":
            row[column_name_seventh] = '1'
        elif row[column_name_sixth] == "高能放电":
            row[column_name_seventh] = '2'
        elif row[column_name_sixth] == "中低温过热":
            row[column_name_seventh] = '3'
        elif row[column_name_sixth] == "高温过热":
            row[column_name_seventh] = '4'
        elif row[column_name_sixth] == "中温过热":
            row[column_name_seventh] = '5'
        else:
            row[column_name_seventh] = '6'

    # 写回CSV文件
    with open(file_path, 'w', newline='', encoding=encoding) as file:
        csv_writer = csv.DictWriter(file, fieldnames=csv_data[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(csv_data)


# 示例用法
csv_file_path = 'a.csv'  # 请替换成你的CSV文件路径
update_seventh_column(csv_file_path)
