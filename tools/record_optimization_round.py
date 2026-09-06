from pathlib import Path
import json,sys
import numpy as np
def record_round(n):
    root=Path('docs/experiments/ten-rounds');r=json.loads((root/f'r{n:02d}/results.json').read_text());p=root/f'r{n:02d}.md'
    s=p.read_text()+'\n## 完整实验结果\n\n'
    s+=f"训练源提交：`{r['source']['commit']}`。三个种子各 32,768 步，实际设备 MPS。本轮总墙钟耗时 {r['elapsed_seconds']/60:.2f} 分钟（含训练和调参评估）。\n\n"
    for kind,label in [('best','选中的最佳模型'),('last','训练结束模型')]:
     a=r['aggregate'][kind];s+=f"- **{label}：**验证吞吐量 {a['horizon_throughput_bps']['mean']/1e6:.3f} ± {a['horizon_throughput_bps']['std_training_seeds']/1e6:.3f} Mbps；覆盖率 {a['horizon_coverage']['mean']*100:.2f}% ± {a['horizon_coverage']['std_training_seeds']*100:.2f} 个百分点；每回合拒绝移动 {a['collisions']['mean']:.2f} 次。\n"
    s+='\n误差项为三个训练种子的场景均值之间的样本标准差。所有服务指标采用统一 64 秒窗口。\n\n'
    for t in r['trials']:
     e=t['evaluations'];b=e['best'];l=e['last'];cv=[x for x in t['curve'] if 'evaluation' in x];means=[x['evaluation']['horizon_throughput_bps']['mean']/1e6 for x in cv]
     s+=f"- 种子 {t['seed']}：选中 {b['checkpoint_step']} 步；调参验证最佳 {b['results']['trained']['summary']['horizon_throughput_bps']['mean']/1e6:.3f} Mbps，最后 {l['results']['trained']['summary']['horizon_throughput_bps']['mean']/1e6:.3f} Mbps。检查点选择集最初四次／最后四次评估均值 {np.mean(means[:4]):.3f}／{np.mean(means[-4:]):.3f} Mbps。\n"
    b=r['trials'][0]['evaluations']['best']['results'];s+='\n相同验证场景基线：'+ '；'.join(f"{k} {b[k]['summary']['horizon_throughput_bps']['mean']/1e6:.3f} Mbps / {b[k]['summary']['horizon_coverage']['mean']*100:.2f}%" for k in ['waypoint','greedy','random','stationary'])+'。\n'
    s+=f"\n[完整配置、逐场景结果、逐更新曲线与来源](r{n:02d}/results.json)。全部训练回合及更新日志无损保存在同目录 `seed*-metrics.jsonl.gz`；模型在 `output/fixed-window/r{n:02d}/seed*/`，最佳与最后检查点的 SHA-256 均在结果文件中。最终留出测试尚未用于此轮决策。\n"
    p.write_text(s)
    print(json.dumps(r['aggregate'],indent=2))


if __name__=="__main__":record_round(int(sys.argv[1]))
