import json
from summ_eval.bleu_metric import BleuMetric
from summ_eval.rouge_metric import RougeMetric
from summ_eval.rouge_we_metric import RougeWeMetric
from summ_eval.s3_metric import S3Metric
from summ_eval.bert_score_metric import BertScoreMetric
# from summ_eval.mover_score_metric import MoverScoreMetric
# from summ_eval.summa_qa_metric import SummaQAMetric
# from summ_eval.blanc_metric import BlancMetric
# from summ_eval.supert_metric import SupertMetric
# from summ_eval.meteor_metric import MeteorMetric
# from summ_eval.data_stats_metric import DataStatsMetric
# from summ_eval.cider_metric import CiderMetric
# from summ_eval.chrfpp_metric import ChrfppMetric
from eval.redundancy import Redundancy
from eval.relevance import Relevance

if __name__ == '__main__':
    # golden_summary_path = "../data_mx/multi_x/tokTrunc_1024_utf/testY.txt" # multiX
    # golden_summary_path = "../data_mx/multi_x/tokTrunc_1024_utf_nosep/testY.txt" # multiX nosep

    # golden_summary_path = "../data_mx/multi_news/tokTrunc_1024_utf/testY.txt" #multi News
    # golden_summary_path = "../data_mx/multi_news/tokTrunc_1024_utf_nosep/testY.txt" #multi News nosep

    golden_summary_path = "../Results/M2 - Impact of special token/Multi N/HT_docl/6 - 23/out.80000.gold"
    generated_summary_path = "../Results/M2 - Impact of special token/Multi N/HT_docl/6 - 23/out.80000.candidate"

    # generated_summary_path = "../Results/M2 - Impact of special token/Multi N/Tran_ori/test.transformer_ori.out.min_length200"

    blue = BleuMetric()
    bert = BertScoreMetric()
    rouge = RougeMetric()
    rouge_we = RougeWeMetric()
    s3 = S3Metric()
    relevance = Relevance()
    redundancy = Redundancy()
    # mover = MoverScoreMetric() # need cuda
    # summaQA = SummaQAMetric(use_gpu=False) # run python -m spacy download en
    # blanc = BlancMetric(device='cpu') # need cuda
    # supert = SupertMetric() # PYTHONPATH, numpy 1.21.5 to 1.19.5, change type of .cache/torch/<model>/module.json
    # meteor = MeteorMetric() # did not work
    # dataStat = DataStatsMetric()
    # CIDer = CiderMetric()
    # CHRF = ChrfppMetric()

    # generated = ["This is one summary", "This is another summary"]
    # golden = ["This is one reference", "This is another"]

    # generated summaries
    with open(generated_summary_path, "r", encoding='utf-8') as f:
        generated = [line.strip()[2:] for line in f]  # removing the dash in the beginning and making the list

    # golden summaries
    with open(golden_summary_path, "r", encoding='utf-8') as f:
        golden = [line.strip()[2:] for line in f]  # removing the dash in the beginning and making the list

    # print(generated[0])
    # print(golden[0])

    # generated = ["Three people are having a meeting"]
    # golden = ["3 people are having a meeting"]

    # generated = ["Three people are having a meeting"]
    # golden = ["3 people are having a discussion"]

    generated = ["Three people are having a meeting"]
    golden = ["3 people are in a discussion"]
    results_dict = {
        'rouge':  rouge.evaluate_batch(generated, golden),
        'rouge_we':  rouge_we.evaluate_batch(generated, golden),
        # 'blue': blue.evaluate_batch(generated, golden),
        'bert': bert.evaluate_batch(generated, golden),
        # 's3': s3.evaluate_batch(generated, golden),
        # 'relevance' : relevance.evaluateBatch(generated, golden),
        # 'redundancy' : redundancy.evaluateBatch(generated)
    }

    print(results_dict)
    # with open('ht_docl_mn.json', 'w') as file:
    #    json.dump(results_dict, file)
