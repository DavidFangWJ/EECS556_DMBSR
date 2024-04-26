import re

def extract_psnr_iter_lines(input_file_path, output_file_path):
    iter_psnr_pattern = r"iter:\s+([0-9,]+), Average PSNR : (\d+\.\d+)dB"
    iter_per_epoch_pattern = r"iters:\s+([0-9,]+)"

    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            if re.search(iter_psnr_pattern, line) or re.search(iter_per_epoch_pattern, line):
                output_file.write(line)

# 使用示例
input_log_file = 'train_low.log'  # 替换为你的输入日志文件路径
output_log_file = 'extracted_'+input_log_file  # 替换为你想要保存的输出日志文件路径
extract_psnr_iter_lines(input_log_file, output_log_file)