import numpy as np

def compute_metrics(results_list, k=10):
    mrr_total, recall_total, ndcg_total = 0.0, 0.0, 0.0
    count = 0

    for results in results_list:
        relevances = [r['is_selected'] for r in results[:k]]
        
        mrr = 0.0
        for rank, rel in enumerate(relevances, 1):
            if rel:
                mrr = 1.0 / rank
                break
        
        recall = int(any(relevances))
        
        dcg = sum((rel / np.log2(i + 2)) for i, rel in enumerate(relevances))
        idcg = sum((1.0 / np.log2(i + 2)) for i in range(sum(relevances)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        mrr_total += mrr
        recall_total += recall
        ndcg_total += ndcg
        count += 1

    return {
        'MRR@{}'.format(k): mrr_total / count,
        'Recall@{}'.format(k): recall_total / count,
        'NDCG@{}'.format(k): ndcg_total / count
    }