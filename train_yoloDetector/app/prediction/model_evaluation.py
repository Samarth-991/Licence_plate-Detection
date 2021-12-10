import logging
import os
import pandas as pd
from os.path import join as join_path
from app.utils import kpi_metrics
from collections import defaultdict
from app.utils import utils
from app.config.load_config import LoadJson


class Evaluate_model:
    def __init__(self,data_txt='val.txt',metadata='metadata.pickle',kpi_name='KPI_metrics.csv'):
        self.json_pipeline = LoadJson()
        self.classes = self.json_pipeline.get_labels()

        # get inference parameters
        self.iou ,self.cnf,self.outpath = self.json_pipeline.get_inferparams()
        self.metadata_path = os.path.join(self.outpath ,metadata)

        self.kpiresult = os.path.join(self.json_pipeline.dir_path, kpi_name)

        self.val_text = os.path.join(self.json_pipeline.model_path, data_txt)
        self.val_csv = join_path(self.json_pipeline.model_path,data_txt[:-4]+'.csv')

        # define parameters
        self.gt_dict = defaultdict(list)
        self.pred_dict = defaultdict(list)
        self.result_csv = join_path(self.outpath,'predictions.csv')
        self.transform()
        self.fit()

    def transform(self):
        with open(self.val_text, 'r+')as fd:
            val_imgs = fd.readlines()
        val_imgs = list(map(lambda x: x.strip(),val_imgs))
        utils.read_and_save_ascsv(self.metadata_path,val_imgs,csv_path=self.result_csv)
        # read both Csv and create dataframe
        test_df = pd.read_csv(self.result_csv)
        val_df = pd.read_csv(self.val_csv)
        for label in self.classes:
            predcords_df = test_df.loc[test_df['label'] == label, ['xmin', 'ymin', 'xmax', 'ymax']]

            gtcords_df = val_df.loc[val_df['Label'] == label, ['xmin','ymin','xmax','ymax']]
            pred_cords = predcords_df.values.tolist()
            gt_cords = gtcords_df.values.tolist()

            self.pred_dict[label].append(pred_cords)
            self.gt_dict[label].append(gt_cords)

    def fit(self):
        kpi_dict = defaultdict(list)
        for label in self.classes:
            kpi_dict[label].append(kpi_metrics.get_metrices(label, self.gt_dict, self.pred_dict,iou_thr=self.iou))

        precisons, recalls = kpi_metrics.calc_precision_recall(kpi_dict)
        true_positives = [kpi_dict[label][0]['true_positive'] for label in self.classes]
        false_positves = [kpi_dict[label][0]['false_positive'] for label in self.classes]
        false_negatives = [kpi_dict[label][0]['false_negative'] for label in self.classes]
        df = pd.DataFrame(
            {'Part': self.classes, 'TruePositive': true_positives, 'FalsePositive': false_positves,
                                                    'FalseNegative': false_negatives, 'Precision': precisons,'Recall': recalls})
        df.to_csv(self.kpiresult,index=False)


