# -*- coding: utf-8 -*-
"""116 結果公布後的一鍵公平比較。
用法：python3 -I evaluate_116.py actual_116.csv [輸出.md]
actual_116.csv 欄位：群類(09/21)、學校、科系、科目（單科或 英文＋專一）、實際門檻
比較對象（都是 2026-11-13 凍結的預測）：我的 need、V22 的 v12（=75/25）、V22 的 v11、v14
規則（事先講好，不得事後更改）：
  只比兩邊都有預測、也有實際值的同一批關卡；
  準確度＝MAE、RMSE、誤差≤1 比例、誤差≥3 比例、偏差；
  逐關比較誰更準＋符號檢定；另以「校系」為單位的 bootstrap 估 MAE 差的 95% 區間；
  選校結果＝隨機程度學生各選 6 個校系（1挑戰2適中3保底），以實際門檻判定平均達標數；
  換中心的條件：門檻 MAE 差的 95% 區間不含 0，且選校達標數不輸，才換。"""
import sys, json, csv, re, collections
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tuizhen_engine as T
SUB = T.SUB
def norm(x): return re.sub(r'\s', '', x.replace('國立', '').replace('台', '臺'))
def subs_key(t):
    xs = [s for s in re.split(r'[＋+、,，\s]+', t) if s]
    for s in xs:
        if s not in SUB: raise ValueError(f'不認得的科目：{s}')
    return tuple(sorted(SUB.index(s) for s in xs))
def load_frozen():
    F = json.loads((Path(__file__).resolve().parent / 'data' / 'frozen_116_predictions.json').read_text(encoding='utf-8'))
    gates = {}
    for r in F['rows']:
        for c in r['mine']:
            k = (r['g'], norm(r['school']), norm(r['dept']), tuple(sorted(c['subjects'])))
            gates.setdefault(k, {})['mine'] = c['center']; gates[k]['lo'] = c['lo']; gates[k]['hi'] = c['hi']
        for g in (r['v22_v12'] or []):
            k = (r['g'], norm(r['school']), norm(r['dept']), tuple(sorted(SUB.index(s) for s in g['subjects'] if s in SUB)))
            d = gates.setdefault(k, {})
            for n in ('v12', 'v11', 'v14'):
                if g[n] is not None: d[n] = g[n]
    return gates
def load_actual(path):
    out = {}
    for row in csv.DictReader(open(path, encoding='utf-8-sig')):
        g = '09' if row['群類'].lstrip('0') == '9' else '21'
        out[(g, norm(row['學校']), norm(row['科系']), subs_key(row['科目']))] = float(row['實際門檻'])
    return out
def acc(e):
    return dict(n=len(e), MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())), within1=float((np.abs(e) <= 1).mean()), big3=float((np.abs(e) >= 3).mean()), bias=float(e.mean()))
def main(actual_path, out_path=None):
    G = load_frozen(); A = load_actual(actual_path)
    keys = [k for k in A if k in G and 'mine' in G[k] and 'v12' in G[k]]
    L = [f'# 116 年預測準確度比較（共同關卡 {len(keys)} 個；實際檔 {len(A)} 關，找不到預測 {len([k for k in A if k not in G])} 關）\n']
    if len(keys) < 10: L.append('共同關卡太少，結論不可靠。')
    a = np.array([A[k] for k in keys]); progs = [k[:3] for k in keys]
    models = {'我（凍結 need）': np.array([G[k]['mine'] for k in keys], float), 'V22 v12（75/25）': np.array([G[k]['v12'] for k in keys], float)}
    for n in ('v11', 'v14'):
        if all(n in G[k] for k in keys): models[f'V22 {n}'] = np.array([G[k][n] for k in keys], float)
    L.append('## 門檻準確度（預測−實際；偏差為正＝預測偏高）\n\n|模型|MAE|RMSE|誤差≤1|誤差≥3|偏差|\n|---|---|---|---|---|---|')
    for nme, p in models.items():
        r = acc(p - a); L.append(f'|{nme}|{r["MAE"]:.3f}|{r["RMSE"]:.3f}|{r["within1"]*100:.1f}%|{r["big3"]*100:.1f}%|{r["bias"]:+.2f}|')
    lo = np.array([G[k].get('lo', np.nan) for k in keys]); hi = np.array([G[k].get('hi', np.nan) for k in keys])
    ok = ~np.isnan(lo); L.append(f'\n我的預測範圍（lo～hi）涵蓋實際門檻：{np.mean((a[ok] >= lo[ok]) & (a[ok] <= hi[ok]))*100:.1f}%（{int(ok.sum())} 關）')
    m, v = models['我（凍結 need）'], models['V22 v12（75/25）']; em, ev = np.abs(m - a), np.abs(v - a)
    w, l_ = int((ev < em).sum()), int((em < ev).sum())
    p = binomtest(w, w + l_, .5).pvalue if w + l_ else 1.0
    L.append(f'\n## 逐關比較\nV22 較準 {w} 關、我較準 {l_} 關、相同 {len(keys)-w-l_} 關；符號檢定 p={p:.3f}')
    by = collections.defaultdict(list)
    for i, pk in enumerate(progs): by[pk].append(i)
    ids = list(by.values()); rng = np.random.default_rng(116); diffs = []
    for _ in range(4000):
        pick = rng.integers(0, len(ids), len(ids)); ix = np.concatenate([ids[j] for j in pick])
        diffs.append(em[ix].mean() - ev[ix].mean())
    lo95, hi95 = np.percentile(diffs, [2.5, 97.5])
    L.append(f'MAE 差（我−V22）= {em.mean()-ev.mean():+.3f}，以校系為單位 bootstrap 95% 區間 [{lo95:+.3f}, {hi95:+.3f}]（正＝V22 更準）')
    for g in ('09', '21'):
        ix = [i for i, k in enumerate(keys) if k[0] == g]
        if ix: L.append(f'- {g} 群 {len(ix)} 關：我 MAE {em[ix].mean():.3f}／V22 {ev[ix].mean():.3f}')
    # 選校結果：以實際門檻判定
    prog_names = list(by.keys()); W, Tm, Tv, Ta, st = [], [], [], [], []
    for pk in prog_names:
        st.append(len(W))
        for i in by[pk]:
            w_ = np.zeros(5, np.float32); w_[list(keys[i][3])] = 1
            W.append(w_); Tm.append(m[i]); Tv.append(v[i]); Ta.append(a[i])
    W = np.array(W); Tm, Tv, Ta, st = (np.array(x, np.float32) for x in (Tm, Tv, Ta)) if False else (np.array(Tm, np.float32), np.array(Tv, np.float32), np.array(Ta, np.float32), np.array(st))
    sd = T.SD; res = {}
    for model in ('V9常態', '均勻連結'):
        r2 = np.random.default_rng(7); S = T.students(model, 20000, r2); jit = r2.random((20000, len(prog_names)))
        act = T.passed(S, W, Ta, st)
        pm = T.choose(T.prob(S, W, Tm, st, sd), T.TIERS['挑戰1＋適中2＋保底3'], jit)
        gap = np.minimum.reduceat(S @ W.T - Tv, st, axis=1); cls = np.where(gap >= 2, 3, np.where(gap >= 0, 2, 0)).astype(np.int8)
        cls = np.where(cls == 0, np.where(gap >= -2, 1, 0), cls)       # 差≤2級分視為「積極挑戰」（簡化版 V22 分級）
        used = np.zeros(cls.shape, bool); pv = []
        for c in (1, 2, 2, 3, 3, 3):
            sc = np.abs(cls - c) * 10 - 0.001 * jit; sc = np.where(used | (cls == 0), 1e9, sc); j = sc.argmin(1); pv.append(j); used[np.arange(len(S)), j] = True
        pv = np.stack(pv, 1)
        for nme, pk in (('我（機率分級）', pm), ('V22（餘裕分級，v12中心）', pv)):
            ok_ = np.take_along_axis(act, pk, 1).sum(1); res[(model, nme)] = (ok_.mean(), (ok_ >= 1).mean())
    L.append('\n## 選校結果（隨機程度學生各選 6 個校系，用 116 實際門檻判定）\n\n|學生模型|選法|平均達標數|至少上一個|\n|---|---|---|---|')
    for (md, nme), (mu, p1) in res.items(): L.append(f'|{md}|{nme}|{mu:.3f}|{p1*100:.1f}%|')
    L.append('\n## 判讀規則（事先約定）\n換中心的條件：門檻 MAE 差的 95% 區間不含 0，且選校結果不輸，兩者都要成立才換；否則維持現狀。單一年份、校系數有限，結論僅供參考。')
    txt = '\n'.join(L)
    if out_path: Path(out_path).write_text(txt, encoding='utf-8')
    print(txt)
if __name__ == '__main__':
    if len(sys.argv) < 2: print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
