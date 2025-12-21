import time, numpy as np
def benchmark_latency(model, img, runs=50):
    ts=[]
    for _ in range(runs):
        t=time.time(); model.predict(img); ts.append((time.time()-t)*1000)
    return {'mean_ms':float(np.mean(ts)),'std_ms':float(np.std(ts)),'p95_ms':float(np.percentile(ts,95)),'runs':runs}
