import os
import matplotlib.pyplot as plt 
#绘制学习曲线并将图像保存在指定的目录下
def plot_learning_curves(loss, val_mae, dir_to_save):
    # plot learning curves
    fig = plt.figure(figsize=(16, 9))
    #图像大小（16,9） 1 绘制训练损失
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(loss, label='train loss', color='tab:blue')
    ax1.legend(loc = 'upper right')
    #2 绘制验证平均绝对误差
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(val_mae, label='val mae', color='tab:orange')
    ax2.legend(loc = 'upper right')
    # ax2.set_ylim((0,50))
    fig.savefig(os.path.join(dir_to_save, 'learning_curves.png'), bbox_inches='tight', dpi = 300)
    plt.close()
    