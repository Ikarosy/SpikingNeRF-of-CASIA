import os
import fileinput

# 定义文件夹路径
folder_path = './'  # 请将 'your_folder_path' 替换为实际文件夹的路径

# 要替换的语句
# old_statement = 'unsqueeze_masking_fine = False'
# new_statement = 'condensing_fine = True'

# old_statement = '_unsqueeze_masking{}'
# new_statement = '_condensing{}'

# old_statement = ', unsqueeze_masking_fine,'
# new_statement = ', condensing_fine,'

# old_statement = 'unsqueeze_masking=unsqueeze_masking_fine,'
# new_statement = 'condensing=condensing_fine,'



# 要添加的语句
# additional_statement = ''
# additional_statement = 'fine_train = dict(\n N_iters=60000,\n)'

i = 0
# 遍历文件夹中的所有.py文件
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.py') and not '_default' in file and not '_lg' in file and not 'easy_order' in file and not 'tensorf' in file:
            file_path = os.path.join(root, file)
            i += 1
            print('changing the {}-th file'.format(i), file_path)
            # 用fileinput库打开文件以便进行逐行替换
            with fileinput.FileInput(file_path, inplace=True) as f:
                for line in f:
                    # 使用replace方法替换特定语句
                    new_line = line.replace(old_statement, new_statement)
                    print(new_line, end='')
            # with open(file_path, 'a') as f:
            #     f.write('\n' + additional_statement)


print("替换完成")