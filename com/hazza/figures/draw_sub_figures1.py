import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import NullLocator

matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['font.family'] = 'Arial'

#plt.figure(figsize=(30, 10), dpi= 100)

fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, figsize=(24, 8), dpi=100)

fig.text(0.5, 0.01, r'$\mathbf{Prior\ Self-Experience}$', fontsize=12, ha='center')

plt.subplots_adjust(wspace =0.5, hspace=0.5)

ratio = 0.025

#图1
y1, y2 = 8.9, 9.8
ytotal = 12
delta = ytotal * ratio
ax1 = plt.subplot(141)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# y轴
plt.ylim((0, ytotal))
plt.yticks((0, 3, 6, 9, 12))
plt.ylabel(' ')
plt.text(-1, 4.8, r"$\mathbf{\mathit{M}}$", fontsize=12, rotation=90, verticalalignment='bottom')
plt.text(-1, 5.2, "Target Act Score", fontsize=10, rotation=90, verticalalignment='bottom')
# plt.ylabel(r'$\mathbf{\mathit{\LARGE\mathrm K}}\ Target\ Act\ Score$', fontsize=12)
# x轴
plt.xlabel(' ')
plt.xlim(-0.75, 1.75)
x = ['Yes', 'No']

# 小竖线1
plt.axhline(y=y1 - delta, xmin=0.29, xmax=0.31, linewidth=1, color='black')
plt.axhline(y=y1 + delta, xmin=0.29, xmax=0.31, linewidth=1, color='black')
plt.axvline(x=0, ymin=(y1-delta)/ytotal, ymax=(y1 + delta)/ytotal, linewidth=1, color='black')
# 小竖线2
plt.axhline(y=y2 - delta, xmin=0.69, xmax=0.71, linewidth=1, color='black')
plt.axhline(y=y2 + delta, xmin=0.69, xmax=0.71, linewidth=1, color='black')
plt.axvline(x=1, ymin=(y2-delta)/ytotal, ymax=(y2 + delta)/ytotal, linewidth=1, color='black')
# 中间横线
plt.axhline(y=ytotal, xmin=0.3, xmax=0.7, linewidth=1, color = 'black')
plt.axvline(x=0, ymin=(ytotal - delta)/ytotal, ymax=1, linewidth=1, color='black')
plt.axvline(x=1, ymin=(ytotal - delta)/ytotal, ymax=1, linewidth=1, color='black')
ax1.text(0.5, ytotal, '*', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.bar(x, height=[y1, y2], color=["silver", "white"], edgecolor=["black", "black"], width=0.7, linewidth=1)

#图2

y1, y2 = 2.45, 3.11
ytotal = 4
delta = ytotal * ratio
ax1 = plt.subplot(142)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# y轴
plt.ylim((0, ytotal))
plt.yticks((0, 1, 2, 3, 4))
plt.ylabel(' ')
plt.text(-1, 1.6, r"$\mathbf{\mathit{M}}$", fontsize=12, rotation=90, verticalalignment='bottom')
plt.text(-1, 1.73, "Serial Order Score", fontsize=10, rotation=90, verticalalignment='bottom')
# x轴
plt.xlim(-0.75, 1.75)
x = ['Yes', 'No']

# 小竖线1
plt.axhline(y=y1 - delta, xmin=0.29, xmax=0.31, linewidth=1, color='black')
plt.axhline(y=y1 + delta, xmin=0.29, xmax=0.31, linewidth=1, color='black')
plt.axvline(x=0, ymin=(y1-delta)/ytotal, ymax=(y1 + delta)/ytotal, linewidth=1, color='black')
# 小竖线2
plt.axhline(y=y2 - delta, xmin=0.69, xmax=0.71, linewidth=1, color='black')
plt.axhline(y=y2 + delta, xmin=0.69, xmax=0.71, linewidth=1, color='black')
plt.axvline(x=1, ymin=(y2-delta)/ytotal, ymax=(y2 + delta)/ytotal, linewidth=1, color='black')
# 中间横线
plt.axhline(y=ytotal, xmin=0.3, xmax=0.7, linewidth=1, color = 'black')
plt.axvline(x=0, ymin=(ytotal - delta)/ytotal, ymax=1, linewidth=1, color='black')
plt.axvline(x=1, ymin=(ytotal - delta)/ytotal, ymax=1, linewidth=1, color='black')
ax1.text(0.5, ytotal, '***', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.bar(x, height=[y1, y2], color=["silver", "white"], edgecolor=["black", "black"], width=0.7, linewidth=1)


#图3
y1, y2 = 2.15, 2.82
ytotal = 4
delta = ytotal * ratio
ax1 = plt.subplot(143)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# y轴
plt.ylim((0, ytotal))
plt.yticks((0, 1, 2, 3, 4))
plt.ylabel(' ')
plt.text(-1, 1.6, r"$\mathbf{\mathit{M}}$", fontsize=12, rotation=90, verticalalignment='bottom')
plt.text(-1, 1.73, "First Act Score", fontsize=10, rotation=90, verticalalignment='bottom')
# x轴
plt.xlim(-0.75, 1.75)
x = ['Yes', 'No']

# 小竖线1
plt.axhline(y=y1 - delta, xmin=0.29, xmax=0.31, linewidth=1, color='black')
plt.axhline(y=y1 + delta, xmin=0.29, xmax=0.31, linewidth=1, color='black')
plt.axvline(x=0, ymin=(y1-delta)/ytotal, ymax=(y1 + delta)/ytotal, linewidth=1, color='black')
# 小竖线2
plt.axhline(y=y2 - delta, xmin=0.69, xmax=0.71, linewidth=1, color='black')
plt.axhline(y=y2 + delta, xmin=0.69, xmax=0.71, linewidth=1, color='black')
plt.axvline(x=1, ymin=(y2-delta)/ytotal, ymax=(y2 + delta)/ytotal, linewidth=1, color='black')
# 中间横线
plt.axhline(y=ytotal, xmin=0.3, xmax=0.7, linewidth=1, color = 'black')
plt.axvline(x=0, ymin=(ytotal - delta)/ytotal, ymax=1, linewidth=1, color='black')
plt.axvline(x=1, ymin=(ytotal - delta)/ytotal, ymax=1, linewidth=1, color='black')
ax1.text(0.5, ytotal, '***', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.bar(x, height=[y1, y2], color=["silver", "white"], edgecolor=["black", "black"], width=0.7, linewidth=1)



#图4
y1, y2 = 3.63, 3.42
ytotal = 4
delta = ytotal * ratio
ax1 = plt.subplot(144)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# y轴
plt.ylim((0, ytotal))
plt.yticks((0, 1, 2, 3, 4))
plt.ylabel(' ')
plt.text(-1, 1.6, r"$\mathbf{\mathit{M}}$", fontsize=12, rotation=90, verticalalignment='bottom')
plt.text(-1, 1.73, "Outcome Score", fontsize=10, rotation=90, verticalalignment='bottom')
# x轴
plt.xlim(-0.75, 1.75)
x = ['Yes', 'No']

# 小竖线1
plt.axhline(y=y1 - delta, xmin=0.29, xmax=0.31, linewidth=1, color='black')
plt.axhline(y=y1 + delta, xmin=0.29, xmax=0.31, linewidth=1, color='black')
plt.axvline(x=0, ymin=(y1-delta)/ytotal, ymax=(y1 + delta)/ytotal, linewidth=1, color='black')
# 小竖线2
plt.axhline(y=y2 - delta, xmin=0.69, xmax=0.71, linewidth=1, color='black')
plt.axhline(y=y2 + delta, xmin=0.69, xmax=0.71, linewidth=1, color='black')
plt.axvline(x=1, ymin=(y2-delta)/ytotal, ymax=(y2 + delta)/ytotal, linewidth=1, color='black')
# 中间横线
plt.axhline(y=ytotal, xmin=0.3, xmax=0.7, linewidth=1, color = 'black')
plt.axvline(x=0, ymin=(ytotal - delta)/ytotal, ymax=1, linewidth=1, color='black')
plt.axvline(x=1, ymin=(ytotal - delta)/ytotal, ymax=1, linewidth=1, color='black')
ax1.text(0.5, ytotal + delta, r'$\it{ns}$', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.bar(x, height=[y1, y2], color=["silver", "white"], edgecolor=["black", "black"], width=0.7, linewidth=1)

plt.show()