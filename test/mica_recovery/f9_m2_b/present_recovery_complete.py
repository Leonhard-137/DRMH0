#!/usr/bin/env python3
"""Refresh only the completed slow-signal recovery and disk controls.
Reuse existing weighted, joint-posterior plotting functions; never run MICA.
"""
from pathlib import Path
from datetime import datetime, timezone
import json,csv
import present_results as p
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/recovery_complete_20260905'
OUT.mkdir(parents=True,exist_ok=True)
truth=json.loads((HERE/'outputs/truth.json').read_text())
disk=json.loads((HERE/'cases/disk_only/truth.json').read_text())
cases={c['name']:c for c in truth['cases']}
records=[]
for case in p.CASES:
    for band in ['M2','B']:
        reference=disk['bands'][band] if case=='disk_only' else cases[case]['bands'][band]
        record=p.read_result(case,band,reference)
        assert record['status']=='complete',(case,band,record.get('missing'))
        records.append(record)
summary={'generated_utc':datetime.now(timezone.utc).isoformat(),'complete':len(records),'scope':10,'records':records,'notes':['Eight mixed-signal fits plus two disk-only controls.','All raw joint draws with final posterior weights, no cuts.','Native components ordered by centroid; not physical identifications.']}
(OUT/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
p.plot_sim_intervals(records,OUT)
p.plot_sim_width_area(records,OUT)
for band in ['M2','B']:p.plot_lightcurves(records,OUT,'simulation',band)
rows=[]
for r in records:
    for metric,v in r['metrics'].items():rows.append({'case':r['case'],'band':r['band'],'metric':metric,**v})
with (OUT/'posterior_intervals.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def fmt(v):return f"{v['median']:.4g} [{v['low']:.4g}, {v['high']:.4g}]"
lines=['# F9 M2/B 混合慢信号恢复：完整结果','',f"生成时间：{summary['generated_utc']}；八组混合实验及两组纯盘对照，**10/10 完成**。",'',
'固定 F9 的 W2/M2/B 实际 cadence、gap 与约 709 天观测基线。每档中慢成分 RMS 为盘光变 RMS 的 0.3，两个来源共用盘信号与噪声，仅一个随机实现。所有时间为观测者系天数。',
'真实慢响应由 W2 卷积产生：质心 60 天、宽度 30 天，或质心 800 天、宽度 400 天。独立慢信号由独立随机系数与预设低频功率谱产生，没有注入的转移函数或真总 lag。','',
'## 已经看到的结果','',
'- 60 天真实响应：两个波段的长质心和宽度均在中央 68.27% 区间内覆盖注入值。此实现显示 MICA 可以恢复这一档慢响应。',
'- 同档独立慢信号：长质心约为 M2 1413 天、B 1141 天，面积占比约三分之一。非零长分量并不能证明该慢变化是响应；该现象不等于已经定位出代码漏洞。',
'- 超长档：真实响应与独立信号的长质心、宽度、面积占比区间在 M2/B 都有大量重叠。不能用这些区间直接构造可靠来源判别；重叠不是正式的生成模型比较。',
'- 短分量：加入慢信号后中位数约为 M2 0.091–0.109 天、B 0.961–1.105 天；纯盘分别为 0.095 和 1.072 天。没有在本批 M2/B 中看到数倍放大的短质心，但这不能直接排除 F9 U/V 中的慢信号偏差。',
'- 纯盘基准已有小偏差：差分盘核相对质心参考为 M2 0.128 天、B 1.179 天。这与项目 X=2.49 的标量盘表定义不同。不能把所有恢复偏差归因于添加慢信号。','',
'## 参数对照','',
'数字为加权中位数 [中央 68.27% 区间]。小/大分量只是每个完整联合样本按质心排序的标签。','',
'| 条件 | 波段 | 小质心（天） | 大质心（天） | 大分量宽度（天） | 大分量面积占比 |',
'|---|---|---|---|---|---|']
for r in records:
    m=r['metrics'];lines.append(f"| {p.CASE_NAMES[r['case']]} | {r['band']} | "+' | '.join(fmt(m[k]) for k in ['tau_short','tau_long','w_long','fraction_long'])+' |')
lines+=['','## 图','',
'点和误差线为中位数与中央 68.27% 区间，空心菱形为有定义的注入参考。纯盘没有独立的长响应真值，独立信号也没有真总 lag。',
'![分量与总延迟](02_simulation_lags.png)','',
'![长分量宽度和面积占比](06_simulation_width_area.png)','',
'下图比较模拟观测与原生总光变重构。阴影是原生重构不确定度，不是参数中央 68% 区间；不能将可交换原生标签的分解直接命名为盘或慢响应。',
'![M2 光变拟合](04_simulation_lightcurves_M2.png)','',
'![B 光变拟合](04_simulation_lightcurves_B.png)','',
'## 使用范围','',
'仅 M2/B、一个随机实现、一个慢/盘 RMS 比；没有计算误报率、正式来源分类准确率或在同一数据上比较两类生成模型。模拟的质心先验 [-10,2000] 天、宽度先验 [0.01,1200] 天，与实数据 broad_np4 的设置不同。','',
'所有原始后验和权重保留。全支撑核矩由逐样本 Gamma 参数计算，不从 [-100,500] 天的转移函数导出窗口积分。',
'完整八组统计与原生先验见 [recovery_summary.json](../recovery_summary.json)，此次纯盘及混合结果见 [results.json](results.json)。','']
(OUT/'README.md').write_text('\n'.join(lines))
print(json.dumps({'complete':len(records),'scope':10,'output':str(OUT)},ensure_ascii=False))
