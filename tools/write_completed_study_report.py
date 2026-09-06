from pathlib import Path
import json
import numpy as np
root=Path('docs/experiments/ten-rounds')
final=json.loads((root/'final-test.json').read_text());selected=final['selected_by_tuning_validation']
rounds=[json.loads((root/f'r{n:02d}/results.json').read_text()) for n in range(1,11)]
def stats(n,suite,policy,metric,scale=1):
 a=np.array([t['suites'][suite]['results'][policy]['summary'][metric]['mean']/scale for t in final['rounds'][n-1]['trials']])
 return float(a.mean()),float(a.std(ddof=1))
s='''# 十轮小步优化：未知聚类环境中的 UAV 部署\n\n## 已完成的实验\n\n十轮均已完整运行，三种子各 32,768 步，共 30 次 MPS 训练、983,040 环境步。每轮训练前后均有 Git 记录，完整逐更新日志、每场景评估、模型哈希和源提交已保留。此前因任务澄清而停止或被替代的固定布局/航程预算诊断单独存档，不计入这十轮。\n\n场景包含随机矩形尺寸、随机 UAV 起点，以及位置、尺度、方向、权重、数量均变化的人群簇。只保留地图边界、最低高度和机间防碰撞；当前固定合法高度、二维移动，无障碍物、无航程预算，每回合完整运行 64 秒。Actor 不读取真实用户位置或通信服务指标。\n\n实验协议见 [protocol.md](protocol.md)。训练与模型选择场景、调参验证场景、最终测试场景互相分开。最终测试在十轮结束后统一进行，没有据此进一步调整算法。\n\n## 折线图\n\n![十轮留出测试](ten-rounds-test.png)\n\n误差带为三个训练种子的场景均值之间的样本标准差。每类测试包含八个配对场景；细长地图的长宽比分布没有用于训练。图中保留随机动作、原地不动、GP-UCB 贪心和不使用感知的随机航点基线，不只展示对学习策略有利的对照。\n\n![完整训练曲线](training-curves.png)\n\n上图使用每 1024 个环境步的检查点选择集评估，无平滑、无曲线筛选。灰线是三个种子，蓝线是均值。它描述训练进程，不能替代最终测试。\n\n![最佳与最后模型](tuning-best-last.png)\n\n![留出测试中的最佳与最后模型](test-best-last.png)\n\n补充诊断同时评估所有轮次的最终模型，主版本仍按原定验证规则选择，不据此重新挑模型。\n\n## 十轮改动与验证记录\n\n'''
for r in rounds:
 n=r['round'];title=(root/f'r{n:02d}.md').read_text().splitlines()[0].lstrip('# ')
 a=r['aggregate']['best'];s+=f"- [{title}](r{n:02d}.md)：调参验证吞吐量 **{a['horizon_throughput_bps']['mean']/1e6:.3f} ± {a['horizon_throughput_bps']['std_training_seeds']/1e6:.3f} Mbps**，覆盖率 {a['horizon_coverage']['mean']*100:.2f}%。\n"
s+=f'\n## 独立测试结论\n\n按调参验证主指标预先选出的版本是**第 {selected} 轮**。以下报告该版本，不根据测试表现重选轮次。\n\n'
for suite,title in [('standard','标准未见场景'),('elongated','未见细长地图')]:
 m,sd=stats(selected,suite,'trained','horizon_throughput_bps',1e6);cov,csd=stats(selected,suite,'trained','horizon_coverage',.01)
 initial,_=stats(1,suite,'trained','horizon_throughput_bps',1e6)
 s+=f'- **{title}：**吞吐量 {m:.3f} ± {sd:.3f} Mbps，覆盖率 {cov:.2f}% ± {csd:.2f} 个百分点；吞吐量相对第 1 轮变化 {m-initial:+.3f} Mbps。\n'
 for policy,label in [('stationary','原地不动'),('random','随机动作'),('greedy','GP-UCB 贪心'),('waypoint','无感知随机航点')]:
  baseline,_=stats(selected,suite,policy,'horizon_throughput_bps',1e6)
  s+=f'  {label}为 {baseline:.3f} Mbps；学习策略相对它的差值为 {m-baseline:+.3f} Mbps。\n'
s+='''\n不能把十轮调参当作严格消融：后续轮次的父配置由验证结果自适应选择，完整依赖关系见逐轮文档。三个训练种子也不足以作广泛的统计显著性声明。\n\n## 关于收敛与未知环境\n\n训练奖励、策略损失或最佳检查点的单次高分，都不足以证明收敛。这里保留最后模型、完整曲线和跨种子结果供判断；有限实验没有建立理论收敛证明，也没有证明数学意义上的任意地图与任意人群分布都有效。\n\n本轮信道与接入参数仍是探索性模型，没有校准真实灾区。安全计数表示被拒绝的移动，不是已经发生的实际碰撞。已知飞行边界与未知人群位置要区分；本轮不研究未知地图边界、障碍物发现或三维升降。\n\n即使服务指标改善，也不能仅凭此断言 GP 的额外因果贡献已被证明：仍需要匹配训练预算的移除/打乱信念图消融、更多独立训练种子，以及更广的聚类分布与信道校准。失败轮次保留，不能只挑最漂亮的轨迹。\n\n## 可复核产物\n\n- [最终逐场景测试数据与来源](final-test.json)。\n- [测试指标 CSV](test-metrics.csv)与[完整验证曲线 CSV](training-curves.csv)。\n- [留出测试 PDF](ten-rounds-test.pdf)与[训练曲线 PDF](training-curves.pdf)，适合论文或汇报导出。\n- [标准地图真实回放](../../../output/fixed-window/final-test/replay-standard/report.html)与[细长地图真实回放](../../../output/fixed-window/final-test/replay-elongated/report.html)，均展示按验证选择版本的种子 7，没有按测试挑选最漂亮的种子。\n- 每轮 rXX/results.json 包含完整配置、每个种子的来源与检查点 SHA-256、最佳/最终模型的逐场景验证结果及所有更新曲线；同目录 seed*-metrics.jsonl.gz 保留完整训练回合与更新日志。\n\n检查点、原始评估与回放保存在本机 output/fixed-window/。代码、配置、全部精简评估和无损训练日志进入 Git，未向远端推送。\n'''
(root/'README.md').write_text(s)
print('Selected by validation:',selected)
