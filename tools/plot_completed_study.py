"""Publication-style, unsmoothed plots of the completed ten-round study."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
root=Path('docs/experiments/ten-rounds')
font=FontProperties(fname=findfont(FontProperties(family=['Arial Unicode MS','Noto Sans CJK SC','DejaVu Sans'])))
plt.rcParams.update({'font.family':font.get_name(),'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'font.size':10,'savefig.dpi':180})
data=json.loads((root/'final-test.json').read_text())
rounds=[json.loads((root/f'r{n:02d}/results.json').read_text()) for n in range(1,11)]
numbers=np.arange(1,11)
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
csvrows=[]
for col,suite in enumerate(['standard','elongated']):
 for row,(metric,scale,unit) in enumerate([('horizon_throughput_bps',1e6,'吞吐量（Mbps）'),('horizon_coverage',.01,'覆盖率（%）')]):
  ax=axes[row,col]
  values=np.array([[t['suites'][suite]['results']['trained']['summary'][metric]['mean']/scale for t in r['trials']] for r in data['rounds']])
  mean=values.mean(axis=1);std=values.std(axis=1,ddof=1)
  ax.plot(numbers,mean,'o-',color='#15658c',label='学习策略：三种子均值',lw=2)
  ax.fill_between(numbers,mean-std,mean+std,color='#15658c',alpha=.14,label='±1 样本标准差')
  for policy,label,style,color in [('waypoint','无感知随机航点','--','#a86b24'),('greedy','GP-UCB 贪心',':','#5e6670'),('random','随机动作','-.','#9b9b9b'),('stationary','原地不动',(0,(5,2,1,2)),'#7e4e74')]:
   baseline=[r['trials'][0]['suites'][suite]['results'][policy]['summary'][metric]['mean']/scale for r in data['rounds']]
   ax.plot(numbers,baseline,linestyle=style,label=label,color=color)
  for n,a,b in zip(numbers,mean,std):csvrows.append(dict(round=int(n),suite=suite,metric=metric,unit=unit,mean=float(a),std_training_seeds=float(b)))
  ax.set_title(('标准未见场景' if suite=='standard' else '细长地图：分布外测试')+' · '+unit)
  ax.set_ylabel(unit);ax.set_xlabel('实验轮次');ax.set_xticks(numbers);ax.grid(axis='y',alpha=.18);ax.set_ylim(bottom=0)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
fig.suptitle('十轮小步优化：统一 64 秒窗口的留出测试',fontsize=16)
fig.text(.5,.085,'每轮 3 个训练种子 × 32,768 步；每个测试类别 8 个配对场景。第 10 轮后统一评估，无平滑。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.12,1,.94))
fig.savefig(root/'ten-rounds-test.png');fig.savefig(root/'ten-rounds-test.pdf');plt.close(fig)
fig,axes=plt.subplots(2,5,figsize=(16,7),sharex=True,sharey=True)
curverows=[]
for r,ax in zip(rounds,axes.ravel()):
 curves=[]
 for trial in r['trials']:
  rows=[x for x in trial['curve'] if 'evaluation' in x]
  steps=np.array([x['steps'] for x in rows]);values=np.array([x['evaluation']['horizon_throughput_bps']['mean']/1e6 for x in rows])
  curves.append(values);ax.plot(steps,values,lw=.7,color='#999999',alpha=.5)
  for step,value in zip(steps,values):curverows.append(dict(round=r['round'],seed=trial['seed'],steps=int(step),validation_throughput_mbps=float(value)))
 values=np.array(curves);mean=values.mean(axis=0);std=values.std(axis=0,ddof=1)
 ax.plot(steps,mean,color='#15658c',lw=1.6);ax.fill_between(steps,mean-std,mean+std,color='#15658c',alpha=.14)
 ax.set_title(f"第 {r['round']} 轮");ax.set_xlabel('环境步数');ax.set_ylabel('验证吞吐量（Mbps）');ax.grid(alpha=.15);ax.set_ylim(bottom=0)
fig.suptitle('完整训练过程：检查点选择集，三种子均值及标准差',fontsize=15)
fig.text(.5,.02,'灰线为单个训练种子；蓝线与阴影为均值 ± 样本标准差；未平滑，未用测试集挑曲线。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(root/'training-curves.png');fig.savefig(root/'training-curves.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,(metric,scale,label) in zip(axes,[('horizon_throughput_bps',1e6,'吞吐量（Mbps）'),('horizon_coverage',.01,'覆盖率（%）')]):
 for kind,name,color in [('best','按选择集选中的模型','#15658c'),('last','训练结束模型','#a86b24')]:
  mean=np.array([r['aggregate'][kind][metric]['mean']/scale for r in rounds]);std=np.array([r['aggregate'][kind][metric]['std_training_seeds']/scale for r in rounds])
  ax.plot(numbers,mean,'o-',color=color,label=name);ax.fill_between(numbers,mean-std,mean+std,color=color,alpha=.10)
 ax.set_xticks(numbers);ax.set_xlabel('实验轮次');ax.set_ylabel(label);ax.grid(alpha=.15);ax.set_ylim(bottom=0);ax.legend(frameon=False)
fig.suptitle('独立调参验证集：最佳与最后模型，均值 ± 三种子标准差')
fig.tight_layout();fig.savefig(root/'tuning-best-last.png');plt.close(fig)
for name,rows in [('test-metrics.csv',csvrows),('training-curves.csv',curverows)]:
 with (root/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print('Saved PNG, PDF and CSV in',root)

fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,suite,title in zip(axes,['standard','elongated'],['标准未见场景','细长地图']):
 for kind,name,color in [('best','所选检查点','#15658c'),('last','训练结束模型','#a86b24')]:
  if kind=='best':values=np.array([[t['suites'][suite]['results']['trained']['summary']['horizon_throughput_bps']['mean']/1e6 for t in r['trials']] for r in data['rounds']])
  else:values=np.array([[t['last_suites'][suite]['summary']['horizon_throughput_bps']['mean']/1e6 for t in r['trials']] for r in data['rounds']])
  mean=values.mean(axis=1);std=values.std(axis=1,ddof=1)
  ax.plot(numbers,mean,'o-',color=color,label=name);ax.fill_between(numbers,mean-std,mean+std,color=color,alpha=.10)
 ax.set(title=title,xlabel='实验轮次',ylabel='吞吐量（Mbps）',xticks=numbers);ax.grid(alpha=.15);ax.legend(frameon=False)
fig.suptitle('补充诊断：留出测试上的所选与最终模型，未据此重选')
fig.tight_layout();fig.savefig(root/'test-best-last.png');plt.close(fig)
