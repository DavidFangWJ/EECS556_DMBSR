import matplotlib.pyplot as plt
import re

def extract_data_from_log(file_path):
    iter_psnr_pattern = r"iter:\s+([0-9,]+), Average PSNR : (\d+\.\d+)dB"
    iter_per_epoch_pattern = r"iters:\s+([0-9,]+)"
    iters, psnrs1, psnrs2 = [], [], []
    iters_per_epoch = None

    with open(file_path[0], 'r') as file:
        for line in file:
            iter_match = re.search(iter_psnr_pattern, line)
            epoch_match = re.search(iter_per_epoch_pattern, line)

            if iter_match:
                iter_num = int(iter_match.group(1).replace(',', ''))
                iters.append(iter_num)
                psnrs1.append(float(iter_match.group(2)))

            if epoch_match and not iters_per_epoch:
                iters_per_epoch = int(epoch_match.group(1).replace(',', ''))

    with open(file_path[1], 'r') as file:
        for line in file:
            iter_match = re.search(iter_psnr_pattern, line)
            epoch_match = re.search(iter_per_epoch_pattern, line)

            if iter_match:
                iter_num = int(iter_match.group(1).replace(',', ''))
                psnrs2.append(float(iter_match.group(2)))

            if epoch_match and not iters_per_epoch:
                iters_per_epoch = int(epoch_match.group(1).replace(',', ''))

    # Calculate epoch marks
    epoch_marks = []
    if iters_per_epoch:
        num_epochs = iters[-1] // iters_per_epoch
        epoch_marks = [i * iters_per_epoch for i in range(1, num_epochs + 1)]

    return iters, psnrs1, psnrs2, epoch_marks

def plot_psnr_vs_iters(iters, psnrs1, psnrs2, epoch_marks, log_file_path):
    output_path='psnr_vs_iters_'+log_file_path[0]+log_file_path[1]+'.png'
    fig, ax1 = plt.subplots()
    ax1.plot(iters, psnrs1, label='PSNR with npls')
    ax1.plot(iters, psnrs2, label='PSNR w/o npls')
    
    # Marking each epoch with vertical lines
    for mark in epoch_marks:
        ax1.axvline(x=mark, color='r', linestyle='--')

    # Adding an additional x-axis on top to indicate epochs
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())  # ensure the limits of x-axis are aligned
    ax2.set_xticks(epoch_marks)
    ax2.set_xticklabels([f"{i+1}" for i in range(len(epoch_marks))])
    ax2.set_xlabel("Epochs", labelpad=10)
    ax2.tick_params(axis='x', which='major', pad=15)

    # Annotating the largest PSNR value
    max_psnr1 = max(psnrs1)
    max_psnr_iter1 = iters[psnrs1.index(max_psnr1)]
    max_psnr2 = max(psnrs2)
    max_psnr_iter2 = iters[psnrs2.index(max_psnr2)]
    ax1.scatter(max_psnr_iter1, max_psnr1, color='green', marker='o', label=f'Max PSNR ({max_psnr1} dB)')
    ax1.annotate(f'MAX: {max_psnr1} dB', (max_psnr_iter1, max_psnr1), textcoords="offset points", xytext=(0,15), ha='center', color="green")
    ax1.scatter(max_psnr_iter2, max_psnr2, color='purple', marker='o', label=f'Max PSNR ({max_psnr2} dB)')
    ax1.annotate(f'MAX: {max_psnr2} dB', (max_psnr_iter2, max_psnr2), textcoords="offset points", xytext=(0,15), ha='center', color="purple")

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR vs Iterations & Epochs \n with and without Npls in high config', pad=10)
    ax1.legend()
    plt.tight_layout()
    plt.savefig(output_path)

# Example usage
plt.rcParams.update({'font.size': 14})
log_file_path = ['extracted_mytrain_npls_high.log', 'extracted_train_high.log']  # Replace with your log file path
iters, psnrs1, psnrs2, epoch_marks = extract_data_from_log(log_file_path)
plot_psnr_vs_iters(iters, psnrs1, psnrs2, epoch_marks, log_file_path)