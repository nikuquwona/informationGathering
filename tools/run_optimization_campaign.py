"""Predeclared small-factor trials; choose parents using tuning validation only."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from localgp.config import load_config
from record_optimization_round import record_round


def git(*args):subprocess.run(['git',*args],cwd=ROOT,check=True)


def main():
    candidates=[
      (2,'training','separate_grad_clip',True,'分开裁剪策略与价值网络梯度','两个独立网络共用范数上限会产生优化耦合，单独裁剪检验价值误差是否妨碍策略更新；不改变信息、网络或奖励。'),
      (3,'training','rollout_steps',256,'增加每次更新的新轨迹批量','每次更新收集四倍连续数据，减小单回合梯度噪声；eval_every 同步改为 4，保持每 1024 步评估，训练总步数、每样本优化次数不变。'),
      (4,'training','gae_lambda',.98,'延长优势递推的有效范围','提高 GAE 的 lambda，让延迟的感知收益传播得更远；代价是优势估计方差可能增加。'),
      (5,'training','entropy_coef',.003,'减小熵正则强度','随机探索可能妨碍稳定部署，降低熵系数检验探索与利用的权衡；不预设方向或目的地。'),
      (6,'training','learning_rate',1e-4,'降低学习率','减小更新幅度，检验策略与价值目标同时变化时的稳定性；不增加训练预算。'),
      (7,'environment','information_weight',.1,'降低不确定性代理奖励的权重','信息代理并不直接等于服务效用，降低该项，检验是否过度奖励探索；仍只用可获得的测量与信念。'),
      (8,'training','hidden_size',128,'增加网络隐藏宽度','检验当前模型容量是否限制对不同起点和聚类场景的适应；不改变观测信息或任务。'),
      (9,'environment','gp_length_scale',7.5,'缩短 GP 空间相关长度','原先 15 米相关长度可能平滑掉较窄的人群信号簇；减半检验感知分辨能力，使用同样训练和测试分布。'),
    ]
    for number,section,key,value,title,hypothesis in candidates:
        if (ROOT/f'docs/experiments/ten-rounds/r{number:02d}/results.json').exists():
            raise FileExistsError('This study round is already recorded; use a separate study for repetition')
        previous=[]
        for n in range(1,number):
            r=json.loads((ROOT/f'docs/experiments/ten-rounds/r{n:02d}/results.json').read_text())
            previous.append((r['aggregate']['best']['horizon_throughput_bps']['mean'],n,r))
        _,parent,best=max(previous,key=lambda x:x[0])
        cfg=json.loads(json.dumps(best['config']));old=cfg[section].get(key,'default')
        cfg[section][key]=value
        if key=='rollout_steps':cfg['training']['eval_every']=1024//value
        config=ROOT/f'configs/optimization/r{number:02d}.json'
        config.write_text(json.dumps(cfg,indent=2)+'\n');load_config(config)
        doc=ROOT/f'docs/experiments/ten-rounds/r{number:02d}.md'
        doc.write_text(f'# 第 {number} 轮：{title}\n\n父配置：第 {parent} 轮，由此前独立调参验证吞吐量均值选择。唯一主要因素：`{section}.{key}` 从 `{old}` 改为 `{value}`。\n\n假设：{hypothesis}\n\n所有种子从头训练，各 32,768 步；任务分布、用户真值信息边界和安全约束不变。\n')
        git('add',str(config),str(doc));git('commit','-m',f'experiment: round {number:02d} vary {key} from round {parent:02d}')
        print(f'ROUND {number} START parent={parent}, {key}={value}',flush=True)
        subprocess.run([sys.executable,str(ROOT/'tools/run_optimization_round.py'),str(number)],cwd=ROOT,check=True)
        record_round(number)
        current=json.loads((ROOT/f'docs/experiments/ten-rounds/r{number:02d}/results.json').read_text())
        gain=current['aggregate']['best']['horizon_throughput_bps']['mean']-best['aggregate']['best']['horizon_throughput_bps']['mean']
        decision='成为后续候选父配置' if gain>0 else '保留失败/未改善结果，后续回退到此前最优配置'
        with doc.open('a') as f:f.write(f'\n## 决策\n\n相对父配置，独立调参验证平均吞吐量变化 {gain/1e6:+.3f} Mbps；{decision}。这里只是调参依据，不是统计显著性或最终测试结论。\n')
        git('add',str(doc),str(ROOT/f'docs/experiments/ten-rounds/r{number:02d}'))
        git('commit','-m',f'experiment: record round {number:02d} complete results and parent decision')
        print(f'ROUND {number} DONE gain={gain/1e6:+.3f} Mbps',flush=True)


if __name__=='__main__':main()
