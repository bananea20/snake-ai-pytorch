import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


# 这段代码的作用是使用matplotlib绘制游戏得分图，
# 其中scores和mean_scores是得分数据。
# plt.ion()开启交互模式，
# display.clear_output()清除之前的图形，
# display.display(plt.gcf())显示最新的图形。
# 接下来使用plt绘制得分图，设置标题和轴标签，使
# 用plot函数绘制scores和mean_scores的折线图，
# 使用text函数在图中标出最后得分和平均得分。
# 最后使用plt.show()函数显示图像并使用plt.pause()函数使图像暂停0.1秒，以允许更新图像。