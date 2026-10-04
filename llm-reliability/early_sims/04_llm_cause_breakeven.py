"""LLM 원인 분류 정확도의 손익분기점과 호출 타이밍 정책 비교 (104주, 200개 가상 세계).

실행: early_sims 폴더 안에서 python 04_llm_cause_breakeven.py  (결과 그림은 같은 폴더에 저장)
"""
import numpy as np, sys, pickle
A, X0, W = 10.0, 8.0, 104
SD_S, SD_T, N_S, N_T = 0.20, 0.35, 8, 150          # 설문: 소량·정확 / 텍스트: 대량·부정확
LAM, KEEP = 0.98, 0.02                              # 평소 느린 망각, 폐기 시 남기는 가중치
g = np.array([1, np.log(X0)])

def make_world(seed):
    rng = np.random.default_rng(seed)
    while True:
        wk = np.sort(rng.choice(np.arange(15, 96), 4, replace=False))
        if np.all(np.diff(wk) >= 15): break
    kinds = rng.permutation(["R", "R", "M", "M"])
    b = np.full(W, 0.55); bias = np.ones(W); ver = np.zeros(W, int)
    for w, k in zip(wk, kinds):
        if k == "R": b[w:] += rng.choice([-1, 1]) * 0.06           # 실제 선호 변화 (x=8에서 약 13%)
        else: bias[w:] *= 1 + rng.choice([-1, 1]) * 0.10; ver[w:] += 1   # 텍스트 추출 버전 교체 (10% 편향)
    data = []
    for t in range(W):
        xs = np.clip(np.exp(rng.normal(np.log(4), .45, N_S)), 1, 12); xt = np.clip(np.exp(rng.normal(np.log(4), .45, N_T)), 1, 12)
        ys = np.log(A) + (b[t]-1)*np.log(xs) + rng.normal(0, SD_S, N_S)
        yt = np.log(A) + (b[t]-1)*np.log(xt) + np.log(bias[t]) + rng.normal(0, SD_T, N_T)
        data.append((np.log(xs), ys, np.log(xt), yt))
    truth = A * X0**(b-1)
    events = dict(zip(wk.tolist(), kinds.tolist()))
    return data, truth, bias, ver, events

def run(world, policy, p=1.0, seed=0):
    """policy: slow | always_real | stat | oracle | llm_alarm | llm_gated | llm_weekly"""
    data, truth, bias, ver, events = world
    rng = np.random.default_rng(seed)
    corr = {0: 1.0}; discard = -1; cp = cn = 0.0; refr = 0; calls = 0
    est = np.zeros(W); beta = None
    def act(kind, t):
        nonlocal discard
        if kind == "R": discard = max(discard, t - 3)
        elif kind == "M": corr[ver[t]] = bias[t] / bias[0] * np.exp(rng.normal(0, .01))   # 기준 텍스트로 버전 보정
    def truth_kind(t):   # 최근 12주 안에 일어난 가장 최근 사건 (오분류 후 재경보 포함)
        best = "N"
        for w, k in events.items():
            if t - 12 <= w <= t: best = k
        return best
    def llm(true_k):
        if rng.random() < p: return true_k
        return rng.choice([k for k in ("R", "M", "N") if k != true_k])
    for t in range(W):
        ls, ys, lt, yt = data[t]
        if policy == "llm_weekly":                    # 매주 맥락(로그·뉴스)을 읽음: 사건 당일 감지 가능, 대신 환각 가능
            calls += 1
            if t in events:
                if rng.random() < p: act(events[t], t)
                elif rng.random() < .5: act("M" if events[t] == "R" else "R", t)
            elif rng.random() < 0.06 * (1 - p): act(rng.choice(["R", "M"]), t)
        if policy == "llm_event":                     # 배포 로그·뉴스에 기록이 생긴 주에만 호출
            if t in events and rng.random() < 0.8:    # 사건의 80%는 로그/뉴스에 흔적을 남김
                calls += 1
                if rng.random() < p: act(events[t], t)
                elif rng.random() < .5: act("M" if events[t] == "R" else "R", t)
            elif t not in events and rng.random() < 0.10:   # 무관한 로그/뉴스가 있는 주 (10%)
                calls += 1
                if rng.random() < 0.3 * (1 - p): act(rng.choice(["R", "M"]), t)
        if beta is not None and refr <= 0:            # CUSUM 감시
            yt_c = yt - np.log(corr.get(ver[t], 1.0))
            X_s = np.column_stack([np.ones_like(ls), ls]); X_t = np.column_stack([np.ones_like(lt), lt])
            rs, rt = ys - X_s @ beta, yt_c - X_t @ beta
            w_s, w_t = 1/SD_S**2, 1/SD_T**2
            z = (w_s*rs.sum() + w_t*rt.sum()) / np.sqrt(w_s*len(rs) + w_t*len(rt))
            cp = max(0, cp + z - .5); cn = max(0, cn - z - .5)
            if max(cp, cn) > 5:
                cp = cn = 0.0; refr = 3
                # 통계적 원인 분류: 최근 3주 설문 잔차가 움직였나? (설문까지 움직이면 실제 변화)
                rs3 = np.concatenate([data[s][1] - np.column_stack([np.ones_like(data[s][0]), data[s][0]]) @ beta for s in range(max(0, t-2), t+1)])
                zs = abs(rs3.mean()) / (SD_S / np.sqrt(len(rs3)))
                stat_k = "R" if zs > 2 else "M"; ambiguous = zs < 2.0   # 설문이 확실히 움직인 경우(zs≥2, 실제 변화 정답률 83%)만 통계로 확정
                tk = truth_kind(t)
                if policy == "always_real": act("R", t)
                elif policy == "stat": act(stat_k, t)
                elif policy == "oracle": act(tk, t)
                elif policy == "llm_alarm": calls += 1; act(llm(tk), t)
                elif policy == "llm_gated":
                    if ambiguous: calls += 1; act(llm(tk), t)
                    else: act(stat_k, t)
                elif policy == "llm_weekly": act(llm(tk), t)
                elif policy == "llm_event": calls += 1; act(llm(tk), t)
        refr -= 1
        # 가중 최소제곱 재추정 (지금까지 전체 데이터)
        LX, LY, WT = [], [], []
        for s in range(t + 1):
            a, b_, c, d = data[s]; tw = LAM**(t - s) * (KEEP if s < discard else 1.0)
            LX += [a, c]; LY += [b_, d - np.log(corr.get(ver[s], 1.0))]
            WT += [np.full(len(a), tw/SD_S**2), np.full(len(c), tw/SD_T**2)]
        LX, LY, WT = map(np.concatenate, (LX, LY, WT))
        X = np.column_stack([np.ones_like(LX), LX]); Xw = X * WT[:, None]
        beta = np.linalg.solve(X.T @ Xw, Xw.T @ LY); est[t] = np.exp(g @ beta)
    err = np.abs(est - truth) / truth * 100
    return err, calls

REPS = 200; PS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
def job(args):
    pol, p = args
    r = [run(make_world(i), pol, p, seed=10_000 + i) for i in range(REPS)]
    return args, (np.array([e for e, _ in r]), np.mean([c for _, c in r]))
if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [(pol, 1.0) for pol in ("slow", "always_real", "stat", "oracle")] + \
           [(pol, p) for pol in ("llm_alarm", "llm_gated", "llm_event", "llm_weekly") for p in PS]
    with Pool(4) as pool: out = dict(pool.map(job, jobs))
    for k, (e, c) in out.items(): print(k, "err %.2f%%  calls %.1f" % (e[:, 4:].mean(), c))
    pickle.dump(dict(out=out, PS=PS), open("llm_res.pkl", "wb"))
